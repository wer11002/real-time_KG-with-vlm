"""
generate_espn_csv.py — Fetch ESPN play-by-play for all match folders under data/
and save one CSV per match.

Usage:
    python generate_espn_csv.py          # all match folders
    python generate_espn_csv.py --dry-run  # print what would be fetched, no requests
"""

import argparse
import csv
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR / "src" / "2_web_scraper"))
from espn_scraper import ESPNScraper

FIELDNAMES = ["Action_Type", "Time", "Player", "Team", "Yellow_Card", "Red_Card", "Full_Text"]


# ── Match folder parser (mirrors parse_match_folder in main.py) ────────────

def parse_match_folder(folder: Path):
    """
    Parse "YYYY-MM-DD - Team One - Team Two" → (date, team1, team2) or None.
    Only scans the top level of data/ (not nested SoccerNet layout).
    """
    name  = folder.name
    parts = name.split(" - ", 2)
    if len(parts) != 3:
        return None
    date, team1, team2 = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    return date.strip(), team1.strip(), team2.strip()


def slug(name: str) -> str:
    """'Nottingham Forest' → 'nottingham_forest'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def csv_path_for(date: str, team1: str, team2: str) -> Path:
    return DATA_DIR / f"{date}_{slug(team1)}_{slug(team2)}.csv"


def save_csv(events: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for e in events:
            writer.writerow({
                "Action_Type": e.get("action", ""),
                "Time"       : e.get("time_raw", f"{int(e.get('time', 0))}'"),
                "Player"     : e.get("player")    or "",
                "Team"       : e.get("team")      or "",
                "Yellow_Card": "1" if e.get("yellow") else "0",
                "Red_Card"   : "1" if e.get("red")    else "0",
                "Full_Text"  : e.get("full_text", ""),
            })


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be fetched without making requests")
    args = ap.parse_args()

    # discover match folders at the top level of data/
    folders = sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_dir() and parse_match_folder(f)
    )

    if not folders:
        print(f"No match folders found under {DATA_DIR}")
        print("Expected format: data/YYYY-MM-DD - Team1 - Team2/")
        sys.exit(0)

    print(f"Found {len(folders)} match folder(s).\n")

    found   = []
    missed  = []
    skipped = []

    for folder in folders:
        date, team1, team2 = parse_match_folder(folder)
        out_path = csv_path_for(date, team1, team2)

        if out_path.exists():
            print(f"[SKIP]  {folder.name}")
            print(f"        → {out_path.name} already exists")
            skipped.append(folder.name)
            continue

        if args.dry_run:
            print(f"[DRY]   {folder.name}")
            print(f"        → would fetch ESPN({date}, {team1!r}, {team2!r})")
            print(f"        → would save to {out_path.name}")
            continue

        scraper = ESPNScraper()
        n = scraper.find_and_load(date, team1, team2)

        if n == 0:
            print(f"[MISS]  {folder.name}")
            print(f"        → not found on ESPN")
            missed.append(folder.name)
            continue

        events = scraper.get_all_events()
        save_csv(events, out_path)
        print(f"[OK]    {folder.name}")
        print(f"        → {len(events)} events saved → {out_path.name}")
        found.append(folder.name)

    # ── also ensure the legacy blackburn CSV exists ─────────────────────────
    legacy = DATA_DIR / "blackburn_forest_2019-10-01.csv"
    blackburn_new = DATA_DIR / "2019-10-01_blackburn_rovers_nottingham_forest.csv"
    if not legacy.exists() and blackburn_new.exists():
        import shutil
        shutil.copy2(blackburn_new, legacy)
        print(f"\n[COPY]  {blackburn_new.name} → {legacy.name}  (legacy path)")

    # ── summary ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n--- dry run complete ({len(folders)} folders scanned) ---")
        return

    total = len(folders)
    print(f"\n{'─'*56}")
    print(f"  Found   : {len(found)}/{total} matches")
    if skipped:
        print(f"  Skipped : {len(skipped)}/{total} (CSV already existed)")
    if missed:
        print(f"  Missed  : {len(missed)}/{total} matches")
        for m in missed:
            print(f"    - {m}")
    print(f"{'─'*56}")


if __name__ == "__main__":
    main()
