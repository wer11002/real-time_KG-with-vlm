"""
roster_lookup.py
----------------
Maps jersey numbers to player names using ESPN roster data.

Loaded once at pipeline startup.
Used by align.py after Qwen2-VL detects a jersey number.

Usage:
    lookup = RosterLookup()
    lookup.load_from_espn("2019-10-01", "Blackburn Rovers", "Nottingham Forest")

    player = lookup.find(jersey="7", team="Blackburn Rovers")
    # → "Adam Armstrong"

    player = lookup.find_by_color(jersey="7", color="blue")
    # → [{"player": "Adam Armstrong", "team": "Blackburn Rovers"}, ...]

Quick test:
    python roster_lookup.py
"""

import sys
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from thefuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "2_web_scraper"))

LEAGUES = ["eng.2", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions"]


# ═══════════════════════════════════════════════════════════════════════════
# ROSTER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

class RosterLookup:

    def __init__(self):
        self._rosters  : Dict[str, Dict[str, str]] = {}
        self._loaded   : bool = False
        self._game_id  : Optional[str] = None
        self._tier1_attempts : int = 0
        self._tier1_hits     : int = 0
        self._color_map: dict = {}

    # ── find game ──────────────────────────────────────────────────────────

    def _find_game(self, date_str: str, team1: str, team2: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Search ESPN API for a match. Tries exact date first, then ±1 day.
        Returns (game_id, league) or (None, None).
        """
        from datetime import datetime, timedelta

        try:
            base_date  = datetime.strptime(date_str, "%Y%m%d")
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
                                print(f"  [roster] found on {search_date} "
                                      f"(folder date was {date_str})")
                            return event.get("id"), league
                except Exception:
                    continue

        return None, None

    # ── load from ESPN ─────────────────────────────────────────────────────

    def load_from_espn(self, date: str, team1: str, team2: str) -> int:
        """
        Load rosters from ESPN API.
        Returns total players loaded.
        Warns explicitly if no roster data found.
        """
        date_str = date.replace("-", "")
        print(f"  [roster] loading rosters for {team1} vs {team2}...")

        game_id, league = self._find_game(date_str, team1, team2)
        if not game_id:
            print(f"  [roster] WARNING: match not found in ESPN API")
            print(f"  [roster] WARNING: Tier 1 jersey matching will not fire")
            return 0

        self._game_id = game_id
        url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
               f"/{league}/summary?event={game_id}")

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"  [roster] WARNING: API returned {r.status_code}")
                print(f"  [roster] WARNING: Tier 1 jersey matching will not fire")
                return 0

            data    = r.json()
            rosters = data.get("rosters", [])
            total   = 0

            for team_data in rosters:
                team_name = team_data.get("team", {}).get("displayName", "")
                players   = team_data.get("roster", [])
                self._rosters[team_name] = {}

                for p in players:
                    jersey  = str(p.get("jersey", "")).strip()
                    name    = p.get("athlete", {}).get("displayName", "").strip()
                    if jersey and name:
                        self._rosters[team_name][jersey] = name
                        total += 1

                print(f"  [roster] {team_name}: {len(players)} players loaded")

            if total == 0:
                print(f"  [roster] WARNING: ESPN returned roster structure but "
                      f"no players found")
                print(f"  [roster] WARNING: Tier 1 jersey matching will not fire")
            else:
                self._loaded = True

            return total

        except Exception as e:
            print(f"  [roster] ERROR: {e}")
            print(f"  [roster] WARNING: Tier 1 jersey matching will not fire")
            return 0

    def load_from_game_id(self, game_id: str, league: str) -> int:
        """
        Load rosters using a pre-resolved game_id + league.
        Skips the full scoreboard scan — call this when the caller
        (ESPNScraper) already resolved the game_id.
        """
        self._game_id = game_id
        url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
               f"/{league}/summary?event={game_id}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"  [roster] WARNING: API returned {r.status_code}")
                return 0
            data    = r.json()
            rosters = data.get("rosters", [])
            total   = 0
            for team_data in rosters:
                team_name = team_data.get("team", {}).get("displayName", "")
                players   = team_data.get("roster", [])
                self._rosters[team_name] = {}
                for p in players:
                    jersey = str(p.get("jersey", "")).strip()
                    name   = p.get("athlete", {}).get("displayName", "").strip()
                    if jersey and name:
                        self._rosters[team_name][jersey] = name
                        total += 1
                print(f"  [roster] {team_name}: {len(players)} players loaded")
            if total > 0:
                self._loaded = True
            return total
        except Exception as e:
            print(f"  [roster] ERROR: {e}")
            return 0

    def load_manual(self, rosters: Dict[str, Dict[str, str]]):
        """Load rosters manually (for testing)."""
        self._rosters = rosters
        self._loaded  = True

    def set_color_map(self, color_map: dict):
        """Store the color→team map so find_by_color() can filter by team."""
        self._color_map = color_map or {}

    # ── lookup ─────────────────────────────────────────────────────────────

    def find(self, jersey: str, team: str) -> Optional[str]:
        """
        Find player by jersey + team name (fuzzy match on team).
        Tracks hit rate for evaluation reporting.
        """
        self._tier1_attempts += 1
        jersey = str(jersey).strip().lstrip("#")

        if not self._loaded:
            print(f"  [roster] WARNING: roster not loaded — "
                  f"call load_from_espn() first")
            return None

        best_team = self._find_team(team)

        if not best_team:
            print(f"  [roster] WARNING: team '{team}' not found in roster "
                  f"(available: {list(self._rosters.keys())})")
            return None

        player = self._rosters[best_team].get(jersey)

        if player:
            self._tier1_hits += 1
        else:
            print(f"  [roster] jersey #{jersey} not found in {best_team} roster")

        return player

    def find_by_color(self, jersey: str, color: str) -> List[Dict]:
        """
        Find player by jersey + kit color.
        Uses stored color map (set via set_color_map()) to narrow by team.
        Falls back to returning all matching teams if color is unresolvable.
        """
        self._tier1_attempts += 1
        jersey      = str(jersey).strip().lstrip("#")
        color_lower = color.lower().strip() if color else ""

        resolved_team = None
        if self._color_map and color_lower:
            resolved_team = self._color_map.get(color_lower)
            if not resolved_team:
                for key, team in self._color_map.items():
                    if key and key in color_lower:
                        resolved_team = team
                        break

        results = []
        for team_name, players in self._rosters.items():
            if jersey not in players:
                continue
            if resolved_team and team_name != resolved_team:
                continue
            results.append({"player": players[jersey], "team": team_name})

        if results:
            self._tier1_hits += 1
        return results

    def get_all_players(self, team: str = None) -> List[Dict]:
        results = []
        for team_name, players in self._rosters.items():
            if team and not self._team_matches(team, team_name):
                continue
            for jersey, name in players.items():
                results.append({
                    "jersey": jersey,
                    "name"  : name,
                    "team"  : team_name,
                })
        return sorted(results,
                      key=lambda p: int(p["jersey"]) if p["jersey"].isdigit() else 99)

    def get_teams(self) -> List[str]:
        return list(self._rosters.keys())

    # ── hit rate reporting ─────────────────────────────────────────────────

    def hit_rate(self) -> float:
        """
        Returns Tier 1 jersey match hit rate.
        hit_rate = successful lookups / total lookup attempts
        """
        if self._tier1_attempts == 0:
            return 0.0
        return self._tier1_hits / self._tier1_attempts

    def hit_rate_summary(self) -> str:
        rate = self.hit_rate()
        return (f"Tier 1 jersey lookups: {self._tier1_hits}/{self._tier1_attempts} "
                f"({rate*100:.1f}% hit rate)")

    # ── helpers ────────────────────────────────────────────────────────────

    def _find_team(self, team_query: str) -> Optional[str]:
        best_score = 0
        best_team  = None
        for team_name in self._rosters:
            score = fuzz.token_set_ratio(team_query.lower(), team_name.lower())
            if score > best_score:
                best_score = score
                best_team  = team_name
        return best_team if best_score > 60 else None

    def _team_matches(self, query: str, team_name: str) -> bool:
        return fuzz.token_set_ratio(query.lower(), team_name.lower()) > 60

    def __repr__(self):
        teams = ", ".join(self._rosters.keys())
        total = sum(len(p) for p in self._rosters.values())
        return f"<RosterLookup teams=[{teams}] players={total}>"


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── RosterLookup self-test ───\n")

    lookup = RosterLookup()
    n = lookup.load_from_espn(
        date  = "2019-10-01",
        team1 = "Blackburn Rovers",
        team2 = "Nottingham Forest",
    )
    print(f"\nLoaded {n} players total")
    print(repr(lookup))

    print("\n── Full rosters ──")
    for team in lookup.get_teams():
        print(f"\n{team}:")
        for p in lookup.get_all_players(team):
            print(f"  #{p['jersey']:<3}  {p['name']}")

    print("\n── Direct jersey lookups ──")
    tests = [
        ("7",  "Blackburn Rovers",  "Adam Armstrong"),
        ("23", "Nottingham Forest", "Joe Lolley"),
        ("19", "Blackburn Rovers",  "Stewart Downing"),
        ("21", "Nottingham Forest", "Samba Sow"),
        ("26", "Blackburn",         "Darragh Lenihan"),
        ("99", "Blackburn Rovers",  None),
    ]

    all_passed = True
    for jersey, team, expected in tests:
        result = lookup.find(jersey=jersey, team=team)
        ok     = result == expected
        status = "✓" if ok else "✗"
        if not ok:
            all_passed = False
        print(f"  {status} #{jersey:<3} {team:<22} → {result}  (expected: {expected})")

    print("\n── Hit rate ──")
    print(f"  {lookup.hit_rate_summary()}")

    print("\n── Color-based lookup ──")
    candidates = lookup.find_by_color(jersey="7", color="blue")
    for c in candidates:
        print(f"  #7  {c['player']:<22} ({c['team']})")

    print("\n── Warning test (bad team name) ──")
    result = lookup.find(jersey="7", team="Manchester City")
    print(f"  Result: {result}  (expected: None with warning)")

    print("\n── Warning test (bad jersey) ──")
    result = lookup.find(jersey="99", team="Blackburn Rovers")
    print(f"  Result: {result}  (expected: None with warning)")

    print(f"\n{'✓ all tests passed!' if all_passed else '✗ some tests failed'}")