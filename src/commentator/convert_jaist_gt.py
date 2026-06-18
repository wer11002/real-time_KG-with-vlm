"""
convert_jaist_gt.py — Convert local JAIST SN-Short GT to human_commentary.json

Reads from local clone of Augmented_Soccer (no download needed):
  ~/work/s2616011/Augmented_Soccer/Dataset/short/<league_season>/<match>/
      1_game.json   (1st half)
      2_game.json   (2nd half)

Writes to our pipeline data directory:
  data/sn_long/<season> - <match_name>/human_commentary.json

Usage:
    python src/commentator/convert_jaist_gt.py
    python src/commentator/convert_jaist_gt.py --short-root /other/path
    python src/commentator/convert_jaist_gt.py --dry-run
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
JAIST_REPO    = Path.home() / "work" / "s2616011" / "Augmented_Soccer"
SHORT_ROOT    = JAIST_REPO / "Dataset" / "short"
PIPELINE_DATA = BASE_DIR / "data" / "sn_long"

HALF_OFFSET = 45   # half-2 gameTime minutes are 46-90; store as within-half (subtract 45)

# ── label map ─────────────────────────────────────────────────────────────
# Maps JAIST "label" field values (if present) to our event_type strings.

LABEL_MAP = {
    "goal"        : "Goal",
    "shot"        : "Shot",
    "foul"        : "Foul",
    "corner"      : "Corner",
    "free-kick"   : "Free_Kick",
    "free kick"   : "Free_Kick",
    "substitution": "Substitution",
    "offside"     : "Offside",
    "yellow card" : "YellowCard",
    "red card"    : "RedCard",
}

# ── helpers ───────────────────────────────────────────────────────────────

_EVENT_PATTERNS = [
    (re.compile(r'\bgoal\b',              re.I), "Goal"),
    (re.compile(r'\bshot\b|\bchance\b',   re.I), "Shot"),
    (re.compile(r'\bfoul\b|\bhandball\b', re.I), "Foul"),
    (re.compile(r'\bcorner\b',            re.I), "Corner"),
    (re.compile(r'\bfree.?kick\b',        re.I), "Free_Kick"),
    (re.compile(r'\bsubstitut',           re.I), "Substitution"),
    (re.compile(r'\boffside\b',           re.I), "Offside"),
    (re.compile(r'\byellow card\b',       re.I), "YellowCard"),
    (re.compile(r'\bred card\b',          re.I), "RedCard"),
]

_PLAYER_TEAM_RE = re.compile(
    r'\b([A-ZÀ-Ž][a-zà-ž]+(?:[\s\-][A-ZÀ-Ž][a-zà-ž]+)+)\s*\(([^)]{2,40})\)'
)


def _infer_event_type(text: str) -> str:
    for pat, label in _EVENT_PATTERNS:
        if pat.search(text):
            return label
    return "Unknown"


def _extract_player_team(text: str):
    m = _PLAYER_TEAM_RE.search(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def _parse_gametime(gametime: str, half: int):
    """Parse 'MM:SS' → (minute_within_half, second)."""
    parts = str(gametime).strip().split(":")
    try:
        abs_min = int(parts[0])
        second  = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0
    within = max(0, abs_min - HALF_OFFSET) if half == 2 else abs_min
    return within, second


def _get_text(ann: dict) -> str:
    """Return the best available commentary text from an annotation."""
    for key in ("short-term", "description", "query"):
        v = ann.get(key, "")
        if v and v.strip():
            return v.strip()
    return ""


def _get_event_type(ann: dict, text: str) -> str:
    raw = ann.get("label", "")
    if raw:
        return LABEL_MAP.get(raw.lower().strip(), _infer_event_type(text))
    return _infer_event_type(text)


# ── conversion ────────────────────────────────────────────────────────────

def convert_match(match_dir: Path, half: int) -> list:
    game_file = match_dir / f"{half}_game.json"
    if not game_file.exists():
        return []

    with open(game_file, encoding="utf-8") as f:
        data = json.load(f)

    # support both {"annotations": [...]} and bare list
    annotations = data.get("annotations", data) if isinstance(data, dict) else data

    entries = []
    for ann in annotations:
        gametime_raw = ann.get("gameTime") or ann.get("game_time", "0:00")
        text         = _get_text(ann)
        if not text:
            continue

        minute, second = _parse_gametime(gametime_raw, half)
        player, team   = _extract_player_team(text)
        event_type     = _get_event_type(ann, text)
        half_label     = "1st" if half == 1 else "2nd"

        entries.append({
            "gametime"  : f"{half_label} {minute:02d}:{second:02d}",
            "minute"    : minute,
            "half"      : half,
            "second"    : second,
            "event_type": event_type,
            "player"    : player,
            "team"      : team,
            "human_text": text,
        })
    return entries


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert local JAIST SN-Short GT → human_commentary.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--short-root", default=str(SHORT_ROOT),
                    help=f"Path to Dataset/short/ (default: {SHORT_ROOT})")
    ap.add_argument("--out-dir",    default=str(PIPELINE_DATA),
                    help=f"Output root (default: {PIPELINE_DATA})")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print what would be written without writing")
    args = ap.parse_args()

    short_root = Path(args.short_root)
    out_root   = Path(args.out_dir)

    if not short_root.exists():
        print(f"ERROR: short root not found: {short_root}")
        raise SystemExit(1)

    n_matches    = 0
    n_events     = 0
    type_counts  = defaultdict(int)

    for league_dir in sorted(short_root.iterdir()):
        if not league_dir.is_dir():
            continue

        # "england_epl_2014-2015" → season = "2014-2015"
        parts  = league_dir.name.rsplit("_", 1)
        season = parts[1] if len(parts) == 2 else league_dir.name

        for match_dir in sorted(league_dir.iterdir()):
            if not match_dir.is_dir():
                continue

            match_name     = match_dir.name
            out_folder     = f"{season} - {match_name}"
            out_dir        = out_root / out_folder
            out_path       = out_dir / "human_commentary.json"

            # collect both halves
            all_entries = []
            for half in (1, 2):
                all_entries.extend(convert_match(match_dir, half))

            if not all_entries:
                print(f"  [skip] no entries: {out_folder}")
                continue

            all_entries.sort(key=lambda e: (e["half"], e["minute"], e["second"]))

            if args.dry_run:
                print(f"  would write {len(all_entries):3d} events → {out_path}")
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_entries, f, indent=2, ensure_ascii=False)

            for e in all_entries:
                type_counts[e["event_type"]] += 1
            n_events  += len(all_entries)
            n_matches += 1

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    action = "Would write" if args.dry_run else "Written"
    print(f"{action} {n_matches} matches, {n_events} total events")
    print(f"Output root: {out_root}\n")
    print("Event type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<15} {count:>5}")

    if not args.dry_run:
        print("\nNext steps:")
        print("  python src/commentator/event_anchored_eval.py \\")
        print("      --data-dir data/sn_long/")
        print("  python src/commentator/evaluate_commentary.py \\")
        print("      --data-dir data/sn_long/ --ai-file ai_commentary_anchored.json")


if __name__ == "__main__":
    main()
