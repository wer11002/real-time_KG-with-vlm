"""
convert_jaist_gt.py — Download JAIST SN-Short GT and convert to our format

Downloads all 1_short-term.json files from:
  https://github.com/JAIST-KnOWLab/Augmented_Soccer (Dataset/short/)

Converts each annotation to our human_commentary.json format and saves under:
  data/sn_short/<match_slug>/human_commentary.json

The 720p.mp4 video for each match must be added separately (SoccerNet licence).
Once videos are in place, run event_anchored_eval.py to generate AI commentary
and evaluate against this GT.

Output format per entry:
  {
    "gametime":   "1st 08:00",     # display string
    "minute":     8,               # minute WITHIN the half (0-45+)
    "half":       1,               # 1 or 2
    "second":     0,               # seconds within the minute
    "event_type": "Shot",          # inferred from query text
    "player":     "Danny Welbeck", # extracted from "Player (Team)" pattern
    "team":       "Arsenal",       # extracted from "Player (Team)" pattern
    "human_text": "Great chance! Danny Welbeck..."  # "short-term" field
  }

Usage:
    python src/commentator/convert_jaist_gt.py
    python src/commentator/convert_jaist_gt.py --out-dir data/sn_short
    python src/commentator/convert_jaist_gt.py --token ghp_xxx   # avoid rate limit
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

RAW_BASE  = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
TREE_URL  = (
    f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
)

# Minutes >= this threshold are treated as second half.
# SoccerNet uses continuous minute count (1-90+); second half starts at 46.
HALF_BOUNDARY = 45


# ── event type inference ───────────────────────────────────────────────────

_EVENT_PATTERNS = [
    (re.compile(r'\bgoal\b',             re.I), "Goal"),
    (re.compile(r'\bshot\b|\bchance\b',  re.I), "Shot"),
    (re.compile(r'\bfoul\b|\bhandball\b',re.I), "Foul"),
    (re.compile(r'\bcorner\b',           re.I), "Corner"),
    (re.compile(r'\bfree.?kick\b',       re.I), "Free_Kick"),
    (re.compile(r'\bsubstitut',          re.I), "Substitution"),
    (re.compile(r'\boffside\b',          re.I), "Offside"),
    (re.compile(r'\byellow card\b',      re.I), "YellowCard"),
    (re.compile(r'\bred card\b',         re.I), "RedCard"),
]

def _infer_event_type(text: str) -> str:
    for pattern, label in _EVENT_PATTERNS:
        if pattern.search(text):
            return label
    return "Unknown"


# ── player / team extraction ───────────────────────────────────────────────

# Matches "Firstname Lastname (Team Name)" — handles multi-word names/teams
_PLAYER_TEAM_RE = re.compile(
    r'\b([A-ZÀ-Ž][a-zà-ž]+(?:[\s\-][A-ZÀ-Ž][a-zà-ž]+)+)\s*\(([^)]{2,40})\)'
)

def _extract_player_team(text: str):
    m = _PLAYER_TEAM_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


# ── gameTime parsing ───────────────────────────────────────────────────────

def _parse_gametime(gametime: str):
    """
    Parse JAIST gameTime string (e.g. "46:30") into (half, minute, second).

    SoccerNet uses continuous match minutes:
      minutes 1-45  → first half
      minutes 46-90 → second half (stored as minute-within-half, i.e. -45)

    Returns (half: int, minute_within_half: int, second: int)
    """
    gametime = gametime.strip()
    parts = gametime.split(":")
    try:
        total_min = int(parts[0])
        second    = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 1, 0, 0

    if total_min <= HALF_BOUNDARY:
        return 1, total_min, second
    else:
        return 2, total_min - HALF_BOUNDARY, second


def _gametime_display(half: int, minute: int) -> str:
    label = "1st" if half == 1 else "2nd"
    return f"{label} {minute:02d}:00"


# ── slug helpers ───────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a match folder name to a safe directory name."""
    # e.g. "2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal"
    #   → "2015-02-21_Crystal_Palace_1_2_Arsenal"
    slug = re.sub(r'\s*-\s*\d{2}-\d{2}\s*', '_', name)   # remove time part
    slug = re.sub(r'\s*-\s*', '_', slug)                   # remaining dashes
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
    """Return list of file path strings in the repo tree."""
    print(f"Fetching repo tree from GitHub ...", end=" ", flush=True)
    r = requests.get(TREE_URL, headers=_api_headers(token), timeout=30)
    if r.status_code == 403:
        print(
            "\nGitHub rate limit hit. Pass --token <ghp_xxx> to authenticate "
            "(5000 req/hr vs 60 unauthenticated)."
        )
        sys.exit(1)
    r.raise_for_status()
    data  = r.json()
    paths = [item["path"] for item in data.get("tree", [])
             if item["type"] == "blob"]
    print(f"{len(paths)} files")
    return paths


def _short_term_paths(all_paths: list) -> list:
    """Filter to only Dataset/short/**/1_short-term.json files."""
    return [
        p for p in all_paths
        if p.startswith(SHORT_DIR + "/") and p.endswith("1_short-term.json")
    ]


def _download_json(path: str, token: Optional[str]) -> Optional[dict]:
    url = f"{RAW_BASE}/{quote(path, safe='/')}"
    r   = requests.get(url, headers=_api_headers(token), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


# ── conversion ─────────────────────────────────────────────────────────────

def _convert_annotations(annotations: list) -> list:
    """Convert a list of JAIST annotations to our human_commentary format."""
    entries = []
    for ann in annotations:
        gametime_raw = ann.get("gameTime") or ann.get("game_time", "0:00")
        query        = ann.get("query", "")
        short_text   = ann.get("short-term", ann.get("short_term", query))

        if not short_text.strip():
            continue

        half, minute, second = _parse_gametime(gametime_raw)
        player, team         = _extract_player_team(query)
        event_type           = _infer_event_type(query)

        entries.append({
            "gametime"  : _gametime_display(half, minute),
            "minute"    : minute,
            "half"      : half,
            "second"    : second,
            "event_type": event_type,
            "player"    : player,
            "team"      : team,
            "human_text": short_text.strip(),
        })

    # sort by half, then minute, then second
    entries.sort(key=lambda e: (e["half"], e["minute"], e["second"]))
    return entries


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download JAIST SN-Short GT and convert to human_commentary.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--out-dir",
        default=str(BASE_DIR / "data" / "sn_short"),
        help="Output root directory (default: data/sn_short/)",
    )
    ap.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="GitHub personal access token (avoids 60 req/hr rate limit)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched files without downloading",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. get full repo tree
    all_paths   = _fetch_tree(args.token)
    short_paths = _short_term_paths(all_paths)
    print(f"Found {len(short_paths)} SN-Short annotation files.\n")

    if not short_paths:
        print("ERROR: no 1_short-term.json files found. "
              "The repo structure may have changed.")
        sys.exit(1)

    if args.dry_run:
        for p in short_paths:
            print(" ", p)
        sys.exit(0)

    # 2. process each file
    n_ok      = 0
    n_failed  = 0
    n_empty   = 0

    for i, repo_path in enumerate(sorted(short_paths)):
        # path e.g.:
        # Dataset/short/england_epl_2014-2015/2015-02-21 - 18-00 .../1_short-term.json
        parts        = Path(repo_path).parts  # ['Dataset', 'short', 'league', 'match', 'file']
        league_name  = parts[2] if len(parts) > 2 else "unknown_league"
        match_name   = parts[3] if len(parts) > 3 else "unknown_match"
        match_slug   = _slugify(match_name)
        out_dir      = out_root / league_name / match_slug

        print(f"[{i+1:02d}/{len(short_paths):02d}] {league_name}/{match_slug}")

        # download
        try:
            data = _download_json(repo_path, args.token)
            time.sleep(0.3)   # stay well under rate limit
        except Exception as exc:
            print(f"  ERROR downloading: {exc}")
            n_failed += 1
            continue

        if not data:
            print("  [skip] 404 or empty")
            n_failed += 1
            continue

        annotations = data.get("annotations", [])
        if not annotations:
            print("  [skip] no annotations")
            n_empty += 1
            continue

        entries = _convert_annotations(annotations)
        if not entries:
            print("  [skip] no convertible entries")
            n_empty += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "human_commentary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        print(f"  {len(entries)} annotations → {out_path}")
        n_ok += 1

    print(f"\n{'═'*50}")
    print(f"Done.  OK={n_ok}  empty={n_empty}  failed={n_failed}")
    print(f"Output root: {out_root}")
    print()
    print("Next steps:")
    print("  1. Add 720p.mp4 for each match under its folder")
    print("  2. python src/commentator/event_anchored_eval.py \\")
    print("         --data-dir data/sn_short/")
    print("  3. python src/commentator/evaluate_commentary.py \\")
    print("         --data-dir data/sn_short/ \\")
    print("         --ai-file ai_commentary_anchored.json")


if __name__ == "__main__":
    main()
