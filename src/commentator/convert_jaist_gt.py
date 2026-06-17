"""
convert_jaist_gt.py — Download JAIST SN-Short GT and convert to our format

Downloads 1_game.json + 2_game.json (one per half) from Dataset/short/ in:
  https://github.com/JAIST-KnOWLab/Augmented_Soccer

Merges both halves into one human_commentary.json per match, saved under:
  data/sn_short/<league>/<match_slug>/human_commentary.json

Schema of each source file (Dataset/short/**/N_game.json):
  { "annotations": [
      { "gameTime": "45:55",        <- absolute match minute:second
        "game_time": "45:03",
        "query": "...",
        "short-term": "...",        <- this is our GT human_text
        "query_ano": "..." }
  ]}
  File name prefix tells us the half:  1_game.json = half 1, 2_game.json = half 2

Output format per entry:
  {
    "gametime":   "1st 45:55",    # display string
    "minute":     45,             # minute WITHIN the half (gameTime MM, adjusted)
    "half":       1,              # 1 or 2
    "second":     55,             # seconds
    "event_type": "Goal",         # inferred from query text
    "player":     "Olivier Giroud",
    "team":       "Arsenal",
    "human_text": "Goal! Olivier Giroud (Arsenal) fires..."
  }

Usage:
    python src/commentator/convert_jaist_gt.py
    python src/commentator/convert_jaist_gt.py --out-dir data/sn_short
    python src/commentator/convert_jaist_gt.py --token ghp_xxx   # GitHub PAT
    python src/commentator/convert_jaist_gt.py --dry-run         # list files only
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

# ── constants ─────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
REPO      = "JAIST-KnOWLab/Augmented_Soccer"
BRANCH    = "main"
SHORT_DIR = "Dataset/short"

RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
TREE_URL = (
    f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
)

# First half ends at minute 45; second half gameTime continues as 46-90+.
# Within-half minute for half 2 = gameTime_minute - HALF_OFFSET.
HALF_OFFSET = 45


# ── event type inference ───────────────────────────────────────────────────

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

def _infer_event_type(text: str) -> str:
    for pat, label in _EVENT_PATTERNS:
        if pat.search(text):
            return label
    return "Unknown"


# ── player / team extraction ───────────────────────────────────────────────

_PLAYER_TEAM_RE = re.compile(
    r'\b([A-ZÀ-Ž][a-zà-ž]+(?:[\s\-][A-ZÀ-Ž][a-zà-ž]+)+)\s*\(([^)]{2,40})\)'
)

def _extract_player_team(text: str):
    m = _PLAYER_TEAM_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


# ── gameTime parsing ───────────────────────────────────────────────────────

def _parse_gametime(gametime: str, half: int):
    """
    Parse 'MM:SS' absolute match time into (minute_within_half, second).

    Half 1: gameTime minutes are 0-45  → within-half minute = minute
    Half 2: gameTime minutes are 46-90 → within-half minute = minute - 45
    """
    parts = gametime.strip().split(":")
    try:
        abs_minute = int(parts[0])
        second     = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0

    within_half = abs_minute - HALF_OFFSET if half == 2 else abs_minute
    within_half = max(0, within_half)
    return within_half, second


def _gametime_display(half: int, minute: int, second: int) -> str:
    label = "1st" if half == 1 else "2nd"
    return f"{label} {minute:02d}:{second:02d}"


# ── slug helpers ───────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """'2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal' → '2015-02-21_Crystal_Palace_1_2_Arsenal'"""
    # remove the time portion "- HH-MM "
    slug = re.sub(r'\s*-\s*\d{2}-\d{2}\s*', '_', name)
    slug = re.sub(r'\s*-\s*', '_', slug)
    slug = re.sub(r'\s+', '_', slug)
    slug = re.sub(r'[^A-Za-z0-9_\-.]', '', slug)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug


# ── GitHub API helpers ─────────────────────────────────────────────────────

def _api_headers(token: Optional[str]) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _fetch_tree(token: Optional[str]) -> list:
    print("Fetching repo file tree ...", end=" ", flush=True)
    r = requests.get(TREE_URL, headers=_api_headers(token), timeout=30)
    if r.status_code == 403:
        print(
            "\nGitHub rate limit hit. Use --token <ghp_xxx> to authenticate "
            "(5000 req/hr vs 60 unauthenticated)."
        )
        sys.exit(1)
    r.raise_for_status()
    paths = [item["path"] for item in r.json().get("tree", [])
             if item["type"] == "blob"]
    print(f"{len(paths)} files")
    return paths


def _game_json_paths(all_paths: list) -> list:
    """
    Filter to Dataset/short/**/N_game.json files.
    Returns list of (repo_path, half_number).
    """
    results = []
    for p in all_paths:
        if not p.startswith(SHORT_DIR + "/"):
            continue
        fname = Path(p).name
        m = re.match(r'^(\d+)_game\.json$', fname)
        if m:
            results.append((p, int(m.group(1))))
    return results


def _download_json(path: str, token: Optional[str]) -> Optional[dict]:
    url = f"{RAW_BASE}/{quote(path, safe='/')}"
    r   = requests.get(url, headers=_api_headers(token), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


# ── conversion ─────────────────────────────────────────────────────────────

def _convert_annotations(annotations: list, half: int) -> list:
    entries = []
    for ann in annotations:
        gametime_raw = ann.get("gameTime") or ann.get("game_time", "0:00")
        query        = ann.get("query", "")
        short_text   = ann.get("short-term", query)

        if not short_text.strip():
            continue

        minute, second = _parse_gametime(gametime_raw, half)
        player, team   = _extract_player_team(query)
        event_type     = _infer_event_type(query)

        entries.append({
            "gametime"  : _gametime_display(half, minute, second),
            "minute"    : minute,
            "half"      : half,
            "second"    : second,
            "event_type": event_type,
            "player"    : player,
            "team"      : team,
            "human_text": short_text.strip(),
        })
    return entries


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download JAIST SN-Short GT → human_commentary.json per match.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--out-dir",
        default=str(BASE_DIR / "data" / "sn_short"),
        help="Output root (default: data/sn_short/)",
    )
    ap.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="GitHub PAT — avoids the 60 req/hr rate limit",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched files without downloading",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. enumerate repo
    all_paths   = _fetch_tree(args.token)
    game_files  = _game_json_paths(all_paths)
    print(f"Found {len(game_files)} N_game.json files in Dataset/short/.\n")

    if not game_files:
        print("ERROR: no *_game.json files found. Repo structure may have changed.")
        sys.exit(1)

    if args.dry_run:
        for path, half in sorted(game_files):
            print(f"  half={half}  {path}")
        sys.exit(0)

    # 2. group by match (league/match_folder)
    # key = (league, match_folder), value = {half: repo_path}
    match_map: dict = {}
    for repo_path, half in game_files:
        parts = Path(repo_path).parts
        # parts: ['Dataset', 'short', '<league>', '<match>', '<file>']
        if len(parts) < 5:
            continue
        league = parts[2]
        match  = parts[3]
        match_map.setdefault((league, match), {})[half] = repo_path

    print(f"Unique matches: {len(match_map)}\n")

    n_ok = n_failed = n_empty = 0

    for i, ((league, match_name), half_files) in enumerate(sorted(match_map.items())):
        match_slug = _slugify(match_name)
        out_dir    = out_root / league / match_slug
        print(f"[{i+1:02d}/{len(match_map):02d}] {league}/{match_slug}")

        all_entries = []
        for half in sorted(half_files):
            repo_path = half_files[half]
            try:
                data = _download_json(repo_path, args.token)
                time.sleep(0.4)
            except Exception as exc:
                print(f"  ERROR half={half}: {exc}")
                n_failed += 1
                continue

            if not data:
                print(f"  [skip] half={half} → 404")
                continue

            annotations = data.get("annotations", [])
            entries     = _convert_annotations(annotations, half)
            print(f"  half={half}: {len(annotations)} annotations → {len(entries)} entries")
            all_entries.extend(entries)

        if not all_entries:
            print("  [skip] no convertible entries\n")
            n_empty += 1
            continue

        # sort by half then minute then second
        all_entries.sort(key=lambda e: (e["half"], e["minute"], e["second"]))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "human_commentary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)

        print(f"  → {len(all_entries)} total events saved to {out_path}\n")
        n_ok += 1

    print(f"{'═'*52}")
    print(f"Done.  matches={n_ok}  empty={n_empty}  errors={n_failed}")
    print(f"Output: {out_root}\n")
    print("Next steps:")
    print("  1. Add 720p.mp4 for each match (from SoccerNet)")
    print("  2. python src/commentator/event_anchored_eval.py \\")
    print("         --data-dir data/sn_short/")
    print("  3. python src/commentator/evaluate_commentary.py \\")
    print("         --data-dir data/sn_short/ \\")
    print("         --ai-file ai_commentary_anchored.json")


if __name__ == "__main__":
    main()
