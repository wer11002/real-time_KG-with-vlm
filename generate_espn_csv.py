"""
generate_espn_csv.py — Fetch Blackburn vs Forest 2019-10-01 from ESPN and
save to data/blackburn_forest_2019-10-01.csv for use by evaluate.py.

Run once on the server:
    python generate_espn_csv.py
"""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src" / "2_web_scraper"))
from espn_scraper import ESPNScraper

OUT_PATH = Path(__file__).resolve().parent / "data" / "blackburn_forest_2019-10-01.csv"
FIELDNAMES = ["Action_Type", "Time", "Player", "Team", "Yellow_Card", "Red_Card", "Full_Text"]


def main():
    scraper = ESPNScraper(csv_fallback=OUT_PATH)
    n = scraper.find_and_load("2019-10-01", "Blackburn", "Nottingham Forest")
    if n == 0:
        print("ERROR: no events loaded — ESPN API may be unavailable or match not found")
        sys.exit(1)

    if scraper.source == "csv":
        print("Already loaded from CSV (file exists); nothing to do.")
        return

    events = scraper.get_all_events()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for e in events:
            writer.writerow({
                "Action_Type": e.get("action", ""),
                "Time"       : e.get("time_raw", f"{int(e.get('time', 0))}'"),
                "Player"     : e.get("player") or "",
                "Team"       : e.get("team")   or "",
                "Yellow_Card": "1" if e.get("yellow") else "0",
                "Red_Card"   : "1" if e.get("red")    else "0",
                "Full_Text"  : e.get("full_text", ""),
            })

    print(f"Saved {len(events)} events → {OUT_PATH}")


if __name__ == "__main__":
    main()
