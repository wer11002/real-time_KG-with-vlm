"""
convert_jaist_gt.py — Convert local JAIST SN-Short GT to human_commentary.json

Reads from the locally-cloned Augmented_Soccer repo (no download):
  ~/work/s2616011/Augmented_Soccer/Dataset/short/<league_season>/<match>/
      1_game.json   (1st half SN-Short annotations)
      2_game.json   (2nd half SN-Short annotations)

Writes only to match folders that already have a downloaded video:
  data/sn_long/<season> - <match_name>/human_commentary.json

Run once after pulling new videos, or after re-cloning Augmented_Soccer.

Usage:
    python src/commentator/convert_jaist_gt.py
    python src/commentator/convert_jaist_gt.py --dry-run
    python src/commentator/convert_jaist_gt.py --short-root /other/path/Dataset/short
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
JAIST_REPO   = Path.home() / "work" / "s2616011" / "Augmented_Soccer"
SHORT_ROOT   = JAIST_REPO / "Dataset" / "short"
PIPELINE_OUT = BASE_DIR / "data" / "sn_long"

HALF_OFFSET = 45   # half-2 gameTime runs 46-90; store as within-half (subtract 45)

# ── label map (JAIST "label" field → our event_type) ──────────────────────

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

# ── text heuristic fallback ────────────────────────────────────────────────

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

def infer_event_type_from_text(text: str) -> str:
    for pat, label in _EVENT_PATTERNS:
        if pat.search(text):
            return label
    return "Unknown"

# ── gameTime parsing ───────────────────────────────────────────────────────

def parse_gametime(time_str: str):
    """Parse 'MM:SS' → (minute, second) or (None, None) on failure."""
    parts = str(time_str).strip().split(":")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None, None

# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert local JAIST SN-Short GT to human_commentary.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--short-root", default=str(SHORT_ROOT),
                    help=f"Path to Dataset/short/ (default: {SHORT_ROOT})")
    ap.add_argument("--out-dir",    default=str(PIPELINE_OUT),
                    help=f"Pipeline data root (default: {PIPELINE_OUT})")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print what would be written without writing")
    args = ap.parse_args()

    short_root   = Path(args.short_root)
    pipeline_out = Path(args.out_dir)

    if not short_root.exists():
        print(f"ERROR: short root not found: {short_root}")
        raise SystemExit(1)

    n_converted = 0
    n_skipped   = 0
    type_counts = defaultdict(int)

    for league_dir in sorted(short_root.iterdir()):
        if not league_dir.is_dir():
            continue

        # "england_epl_2014-2015" → season = "2014-2015"
        parts  = league_dir.name.rsplit("_", 1)
        season = parts[1] if len(parts) == 2 else league_dir.name

        for match_dir in sorted(league_dir.iterdir()):
            if not match_dir.is_dir():
                continue

            match_name  = match_dir.name
            out_folder  = pipeline_out / f"{season} - {match_name}"

            # Only convert if the video folder already exists (video downloaded)
            if not out_folder.exists():
                n_skipped += 1
                continue

            merged_events = []
            for half in (1, 2):
                game_file = match_dir / f"{half}_game.json"
                if not game_file.exists():
                    continue

                data        = json.load(open(game_file, encoding="utf-8"))
                annotations = data if isinstance(data, list) else \
                              data.get("annotations", [])

                for entry in annotations:
                    time_str = entry.get("gameTime") or entry.get("game_time")
                    if not time_str:
                        continue

                    m, s = parse_gametime(time_str)
                    if m is None:
                        continue

                    # within-half minute for second half
                    within_min = max(0, m - HALF_OFFSET) if half == 2 else m

                    text = (entry.get("short-term") or
                            entry.get("description") or
                            entry.get("query") or "")
                    if not text:
                        continue

                    label      = (entry.get("label") or "").lower().strip()
                    event_type = (LABEL_MAP.get(label) or
                                  infer_event_type_from_text(text))

                    merged_events.append({
                        "minute"    : within_min,
                        "half"      : half,
                        "second"    : s or 0,
                        "event_type": event_type,
                        "player"    : "",
                        "team"      : "",
                        "human_text": text.strip(),
                    })
                    type_counts[event_type] += 1

            if not merged_events:
                continue

            merged_events.sort(key=lambda e: (e["half"], e["minute"], e["second"]))

            out_file = out_folder / "human_commentary.json"
            if args.dry_run:
                print(f"  would write {len(merged_events):3d} events → {out_file}")
            else:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(merged_events, f, indent=2, ensure_ascii=False)
                print(f"  {len(merged_events):3d} events → {out_file}")

            n_converted += 1

    # ── summary ───────────────────────────────────────────────────────────
    total_events = sum(type_counts.values())
    action       = "Would write" if args.dry_run else "Written"
    print(f"\n{'═'*52}")
    print(f"Matches converted : {n_converted}")
    print(f"Matches skipped   : {n_skipped}  (no video folder in {pipeline_out})")
    print(f"Total events      : {total_events}\n")
    print("Event type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 30 // max(type_counts.values(), default=1))
        print(f"  {etype:<15} {count:>5}  {bar}")

    if not args.dry_run and n_converted:
        print("\nNext:")
        print("  python src/commentator/evaluate_commentary.py \\")
        print("      --data-dir data/sn_long/ --ai-file ai_commentary_anchored.json")


if __name__ == "__main__":
    main()
