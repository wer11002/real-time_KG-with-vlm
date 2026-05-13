# FIX 007 — ESPN API Triple Scan: Share game_id Across All Three Callers

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/2_web_scraper/espn_scraper.py`
- `src/1_video_processor/roster_lookup.py`
- `src/1_video_processor/action_recognizer.py`
- `main.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

At match startup, three separate components each independently search
the ESPN API for the same game:

1. `ESPNScraper.find_and_load()` → `_find_game()` → scans 7 leagues × 3 dates
2. `RosterLookup.load_from_espn()` → `_find_game()` → scans 7 leagues × 3 dates
3. `load_colors_from_espn()` (action_recognizer.py) → scans 7 leagues × 3 dates

In the worst case (match not in the first league checked), this makes
**63 HTTP requests** (21 per caller × 3 callers) to find the same game_id.
These are synchronous, sequential calls at pipeline startup — total wall
time can reach 30–60 seconds before the first clip is processed.

Each `_find_game()` implementation is a copy of the same logic, making it
easy for them to diverge. If ESPN rate-limits the caller, subsequent scans
may fail even when the first one succeeded.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

### 1 — espn_scraper.py: expose game_id + league

Added two fields and two properties to `ESPNScraper`:

```python
self._game_id : Optional[str] = None
self._league  : Optional[str] = None

@property
def game_id(self) -> Optional[str]:
    return self._game_id

@property
def league(self) -> Optional[str]:
    return self._league
```

In `find_and_load()`, the result of `_find_game()` is stored immediately:

```python
game_id, league   = self._find_game(date_str, team1, team2)
self._game_id     = game_id
self._league      = league
```

### 2 — roster_lookup.py: add load_from_game_id()

New method that skips `_find_game()` entirely and goes straight to the
summary endpoint:

```python
def load_from_game_id(self, game_id: str, league: str) -> int:
    url = f".../summary?event={game_id}"
    # same roster-parsing logic as load_from_espn()
    # no scoreboard scan
```

### 3 — action_recognizer.py: add load_colors_from_game_id()

New function that skips the scoreboard scan:

```python
def load_colors_from_game_id(game_id, league, team1, team2):
    url = f".../summary?event={game_id}"
    # same color-extraction logic as load_colors_from_espn()
    # no scoreboard scan
```

### 4 — main.py: use shared game_id when available

```python
scraper = ESPNScraper()
n_espn  = scraper.find_and_load(match_date, team1, team2)

roster = RosterLookup()
if scraper.game_id and scraper.league:
    roster.load_from_game_id(scraper.game_id, scraper.league)
    color_map = load_colors_from_game_id(scraper.game_id, scraper.league, team1, team2)
else:
    roster.load_from_espn(match_date, team1, team2)
    color_map = load_colors_from_espn(match_date, team1, team2)
roster.set_color_map(color_map)
```

The old `load_from_espn()` and `load_colors_from_espn()` code paths are
kept as fallbacks for when ESPNScraper failed (game_id is None).

════════════════════════════════════════════════════════════════
## PART 3 — IMPACT
════════════════════════════════════════════════════════════════

| Scenario | Before (API calls) | After (API calls) |
|----------|--------------------|-------------------|
| Game in first league (eng.2) | 3 | 3 |
| Game in third league (esp.1) | 3×3=9 scoreboard + 3 summary = 12 | 3 scoreboard + 2 summary = 5 |
| Game in last league, ±1 day  | 3×21=63 scoreboard + 3 summary = 66 | 21 scoreboard + 2 summary = 23 |

In the best case the reduction is from 3 to 3 (game found immediately).
In the worst case it drops from 66 to 23 API calls. Wall time at startup
falls from potentially 60s to ~20s in the worst case.

Summary endpoint (`/summary?event={game_id}`) is called twice instead of
three times — once for roster, once for colors — because both components
extract from the same endpoint. A further optimization would cache the
summary response, but that would add complexity beyond the current scope.

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- The summary endpoint is still called separately for roster and colors
  (2 calls instead of 1). A single cached call would save one more
  request, but requires a shared data layer not present yet.
- `RosterLookup._find_game()` is still present for standalone use
  (e.g., testing roster_lookup.py directly). It's not called at match
  startup anymore.
- ESPN rate limiting is not handled with retry/backoff. If the summary
  endpoint returns 429, both roster and colors will fail silently.
