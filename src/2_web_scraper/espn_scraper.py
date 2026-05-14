"""
espn_scraper.py
---------------
Fetches live match events from ESPN's hidden API.

What's new in this version:
    - ESPN events can be marked as "consumed" after first match
    - Prevents duplicate KG nodes when two overlapping clips
      both detect the same real event
    - consume_event(time, action) marks an event as used
    - get_events_up_to() and get_events_near() skip consumed events
      by default (pass consumed=False to get all including consumed)

Usage in pipeline:
    scraper = ESPNScraper()
    scraper.find_and_load("2019-10-01", "Blackburn Rovers", "Nottingham Forest")
    
    events = scraper.get_events_up_to(minute=10.0)
    
    # after matching, mark ESPN event as consumed
    scraper.consume_event(time=9.0, action="Shot")
    
    # next call won't return the consumed event
    events = scraper.get_events_up_to(minute=10.0)

Quick test:
    python espn_scraper.py
"""

import re
import csv
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from thefuzz import fuzz


# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "blackburn_forest_2019-10-01.csv"

LEAGUES = ["eng.2", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions"]

PLAY_TYPE_MAP = {
    "shot-blocked"   : "Shot",
    "shot-wide"      : "Shot",
    "shot-saved"     : "Shot",
    "shot-on-target" : "Shot",
    "shot-off-target": "Shot",
    "shot"           : "Shot",
    "goal"           : "Goal",
    "penalty-scored" : "Goal",
    "penalty-missed" : "Shot",
    "foul"           : "Foul",
    "yellow-card"    : "Foul",
    "red-card"       : "Foul",
    "corner"         : "Corner",
    "offside"        : "Offside",
    "free-kick"      : "Free_Kick",
    "substitution"   : "Substitution",
    "kickoff"        : "Other",
    "end-period"     : "Other",
    "start-period"   : "Other",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def parse_time_min(time_str: str) -> float:
    t = re.sub(r"'", "", str(time_str).strip())
    if "+" in t:
        base, extra = t.split("+", 1)
        try: return float(base.strip()) + float(extra.strip())
        except: return 0.0
    try: return float(t.strip())
    except: return 0.0


def normalize_action(play_type: str) -> str:
    return PLAY_TYPE_MAP.get(play_type.lower().strip(), "Other")


def parse_commentary_event(item: dict) -> Optional[Dict]:
    play = item.get("play", {})
    if not play:
        return None

    clock        = play.get("clock", {})
    time_value   = clock.get("value", 0.0)
    time_display = clock.get("displayValue", "")
    period       = play.get("period", {}).get("number", 1)

    time_min = time_value / 60.0  # clock.value is cumulative seconds from kick-off

    if time_min <= 0 and not time_display:
        return None

    play_type = play.get("type", {})
    type_str  = play_type.get("type", play_type.get("text", "other"))
    action    = normalize_action(type_str)

    if action == "Other":
        return None

    participants = play.get("participants", [])
    player = None
    if participants:
        player = participants[0].get("athlete", {}).get("displayName")

    team   = play.get("team", {}).get("displayName")
    yellow = "yellow" in type_str.lower()
    red    = "red"    in type_str.lower()

    return {
        "time"      : round(time_min, 2),
        "time_raw"  : time_display or f"{int(time_min)}'",
        "player"    : player,
        "team"      : team,
        "action"    : action,
        "yellow"    : yellow,
        "red"       : red,
        "full_text" : item.get("text", play.get("text", "")),
        "period"    : period,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ESPN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════

class ESPNScraper:

    def __init__(self, csv_fallback: Path = CSV_PATH):
        self.csv_fallback  = csv_fallback
        self._events       : List[Dict] = []
        self._loaded       : bool = False
        self._source       : str  = "none"
        self._game_id      : Optional[str] = None
        self._league       : Optional[str] = None
        # dedup tracking — set of (time_rounded, action) tuples
        self._consumed     : Set[Tuple] = set()
        self._consume_count: int = 0

    @property
    def game_id(self) -> Optional[str]:
        return self._game_id

    @property
    def league(self) -> Optional[str]:
        return self._league

    # ── find + load ────────────────────────────────────────────────────────

    def _find_game(self, date_str: str, team1: str, team2: str):
        """
        Search ESPN API for a match by date and team names.
        Tries exact date first, then ±1 day if not found.
        This handles cases where folder date is off by one day
        (e.g. match played Oct 2 but folder named Oct 1).
        """
        from datetime import datetime, timedelta

        # generate date candidates: exact date first, then +1, then -1
        try:
            base_date = datetime.strptime(date_str, "%Y%m%d")
            candidates = [
                date_str,
                (base_date + timedelta(days=1)).strftime("%Y%m%d"),
                (base_date - timedelta(days=1)).strftime("%Y%m%d"),
            ]
        except ValueError:
            candidates = [date_str]

        for search_date in candidates:
            for league in LEAGUES:
                url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
                       f"/{league}/scoreboard?dates={search_date}")
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code != 200:
                        continue
                    for event in r.json().get("events", []):
                        name = event.get("name", "").lower()
                        s1   = fuzz.token_set_ratio(team1.lower(), name)
                        s2   = fuzz.token_set_ratio(team2.lower(), name)
                        if s1 > 80 and s2 > 80:
                            if search_date != date_str:
                                print(f"  [espn] found on {search_date} "
                                      f"(folder date was {date_str})")
                            return event.get("id"), league
                except Exception:
                    continue

        return None, None

    def _load_from_api(self, game_id: str, league: str) -> int:
        url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
               f"/{league}/summary?event={game_id}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return 0
            data       = r.json()
            commentary = data.get("commentary", []) or data.get("keyEvents", [])
            events = [parse_commentary_event(i) for i in commentary]
            events = [e for e in events if e is not None]
            # ESPN commentary often contains duplicate entries for the same event;
            # deduplicate by (time rounded to 6 s, action, player).
            seen   = set()
            unique = []
            for e in events:
                key = (round(e["time"] * 10), e["action"], e.get("player") or "")
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            events = unique
            if events:
                self._events = sorted(events, key=lambda e: e["time"])
                self._source = "api"
                return len(self._events)
        except Exception as e:
            print(f"  [espn] API error: {e}")
        return 0

    def _load_from_csv(self) -> int:
        if not self.csv_fallback.exists():
            return 0
        events = []
        with open(self.csv_fallback, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                time_min = parse_time_min(row.get("Time", "0"))
                player   = row.get("Player", "").strip()
                team     = row.get("Team",   "").strip()
                action   = row.get("Action_Type", "").strip()
                if not action or action == "None":
                    continue
                events.append({
                    "time"      : time_min,
                    "time_raw"  : row.get("Time", "").strip(),
                    "player"    : player if player not in ("None", "") else None,
                    "team"      : team   if team   not in ("None", "") else None,
                    "action"    : action,
                    "yellow"    : row.get("Yellow_Card", "0") == "1",
                    "red"       : row.get("Red_Card",    "0") == "1",
                    "full_text" : row.get("Full_Text", "").strip(),
                    "period"    : 1 if time_min <= 45 else 2,
                })
        self._events = sorted(events, key=lambda e: e["time"])
        self._source = "csv"
        return len(self._events)

    def find_and_load(self, date: str, team1: str, team2: str) -> int:
        date_str = date.replace("-", "")
        print(f"  [espn] searching for {team1} vs {team2} on {date}...")

        game_id, league = self._find_game(date_str, team1, team2)
        self._game_id = game_id
        self._league  = league

        if game_id:
            print(f"  [espn] found game_id={game_id} in {league}")
            n = self._load_from_api(game_id, league)
            if n > 0:
                print(f"  [espn] loaded {n} events from API ✓")
                self._loaded = True
                return n
            print(f"  [espn] API empty — falling back to CSV")
        else:
            print(f"  [espn] match not found — falling back to CSV")

        n = self._load_from_csv()
        if n > 0:
            print(f"  [espn] loaded {n} events from CSV ✓")
            self._loaded = True
        return n

    # ── consumption tracking ───────────────────────────────────────────────

    def _consume_key(self, time: float, action: str) -> Tuple:
        """
        Generate a consumption key for an event.
        Rounds to nearest 0.1 minute (6 s) — fine enough that two fouls at
        9:00 and 9:25 get different keys, while floating-point noise in ESPN
        time (e.g. 9.000 vs 9.001) still collapses to the same bucket.
        """
        time_rounded = round(time * 10) / 10  # nearest 0.1 min (6 s)
        return (time_rounded, action.strip())

    def consume_event(self, time: float, action: str):
        """
        Mark an ESPN event as consumed after it has been matched.
        Prevents duplicate KG nodes from overlapping clips.
        """
        key = self._consume_key(time, action)
        if key not in self._consumed:
            self._consumed.add(key)
            self._consume_count += 1

    def is_consumed(self, event: Dict) -> bool:
        """Check if an ESPN event has already been matched."""
        key = self._consume_key(event["time"], event["action"])
        return key in self._consumed

    def reset_consumed(self):
        """Clear all consumed events (use between matches)."""
        self._consumed.clear()
        self._consume_count = 0

    def consumed_count(self) -> int:
        return self._consume_count

    # ── query methods ──────────────────────────────────────────────────────

    def get_all_events(self, skip_consumed: bool = False) -> List[Dict]:
        self._ensure_loaded()
        if skip_consumed:
            return [e for e in self._events if not self.is_consumed(e)]
        return self._events

    def get_events_up_to(self, minute: float,
                         skip_consumed: bool = True) -> List[Dict]:
        """
        All events from 0 to minute.
        By default skips consumed events to prevent duplicate matching.
        Pass skip_consumed=False to get all events including consumed ones.
        """
        self._ensure_loaded()
        return [
            e for e in self._events
            if e["time"] <= minute
            and (not skip_consumed or not self.is_consumed(e))
        ]

    def get_events_in_window(self, from_min: float, to_min: float,
                             skip_consumed: bool = True) -> List[Dict]:
        self._ensure_loaded()
        return [
            e for e in self._events
            if from_min <= e["time"] <= to_min
            and (not skip_consumed or not self.is_consumed(e))
        ]

    def get_events_near(self, minute: float, tolerance: float = 2.0,
                        skip_consumed: bool = False) -> List[Dict]:
        self._ensure_loaded()
        return [
            e for e in self._events
            if abs(e["time"] - minute) <= tolerance
            and (not skip_consumed or not self.is_consumed(e))
        ]

    def total_events(self) -> int:
        return len(self._events)

    def source(self) -> str:
        return self._source

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call find_and_load() first.")

    def summary(self) -> str:
        lines = [f"ESPNScraper [{self._source}] — {len(self._events)} events "
                 f"({self._consume_count} consumed)"]
        available = [e for e in self._events if not self.is_consumed(e)]
        for e in available[:5]:
            player = e["player"] or "—"
            lines.append(f"  [{e['time_raw']:>6}]  {e['action']:<12}  {player}")
        if len(available) > 5:
            lines.append(f"  ... and {len(available)-5} more available")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def load_espn_events(date: str, team1: str, team2: str) -> List[Dict]:
    scraper = ESPNScraper()
    scraper.find_and_load(date, team1, team2)
    return scraper.get_all_events()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── ESPNScraper self-test ───\n")

    scraper = ESPNScraper()
    n = scraper.find_and_load(
        date  = "2019-10-01",
        team1 = "Blackburn Rovers",
        team2 = "Nottingham Forest",
    )

    print(f"\nLoaded {n} events from: {scraper.source()}\n")
    print(scraper.summary())

    print("\n── Events in first 10 minutes ──")
    for e in scraper.get_events_up_to(10):
        player = e["player"] or "—"
        print(f"  [{e['time_raw']:>6}]  {e['action']:<12}  {player:<22}  {e['team']}")

    # ── test dedup ─────────────────────────────────────────────────────────
    print("\n── Dedup test ──")
    print("  Before consuming: events near 9' (tolerance=1):")
    before = scraper.get_events_near(9.0, tolerance=1.0, skip_consumed=False)
    for e in before:
        print(f"    [{e['time_raw']:>6}]  {e['action']:<12}  {e['player']}")

    # simulate: clip 1 matches Armstrong's shot at 9'
    print("\n  → consuming Shot at 9.0 (clip 1 matched it)")
    scraper.consume_event(time=9.0, action="Shot")

    print(f"\n  After consuming: events near 9' (skip_consumed=True):")
    after = scraper.get_events_near(9.0, tolerance=1.0, skip_consumed=True)
    for e in after:
        print(f"    [{e['time_raw']:>6}]  {e['action']:<12}  {e['player']}")
    if not after:
        print("    (none — correctly blocked duplicate)")

    print(f"\n  Total consumed: {scraper.consumed_count()}")

    print("\n── Yellow cards ──")
    for e in scraper.get_all_events():
        if e["yellow"]:
            print(f"  [{e['time_raw']:>6}]  {e['player']}  🟨")

    print("\n✓ done!")