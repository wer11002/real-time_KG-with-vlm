# FIX 003 — Dedup OR-logic Silently Drops Events + find_by_color() Ignores Color

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/3_buffer_matcher/buffer.py`
- `src/1_video_processor/roster_lookup.py`
- `main.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEMS
════════════════════════════════════════════════════════════════

Two separate bugs in the dedup and matching layer that together caused
real events to be silently dropped and players to remain UNKNOWN even
when the VLM had identified their team color.

────────────────────────────────────────────────────────────────
### Bug A — OR logic in EventBuffer._find_duplicate()

In `src/3_buffer_matcher/buffer.py` line 141, the deduplication check was:

```python
time_close   = abs(e.video_time  - new.video_time)  <= self.dedup_window   # 30s
clip_overlap = abs(e.clip_start  - new.clip_start)  <= self.dedup_window   # 30s
if time_close or clip_overlap:
    return e   # treated as duplicate
```

With `clip_step = 30s` and `dedup_window = 30s`:
- Adjacent clips always have clip_start difference of exactly 30s
- 30 ≤ 30 is True, so `clip_overlap` is **always True** for adjacent clips
- With `OR`, any same-type action in adjacent clips was treated as a duplicate
  regardless of whether the action times were actually close

This means: a Foul at 10s into clip 1 (video_time=10s) and a completely
different Foul at 55s into the same clip range (video_time=55s, clip 2)
would be deduplicated because:
  - time_close: |10 - 55| = 45s > 30s  → False
  - clip_overlap: |0 - 30| = 30s ≤ 30s → True
  - `False or True` → True → silently dropped

In a 90-minute match with ~22 ESPN-recorded fouls and clip_step=30s,
approximately half the fouls in adjacent clips were being silently erased.
No error is raised — the event just never reaches the KG.

In `evaluate.py` terms: these lost events become False Negatives —
real events in the ground truth that the pipeline missed.

────────────────────────────────────────────────────────────────
### Bug B — find_by_color() ignores its color argument

In `src/1_video_processor/roster_lookup.py` lines 185–198, the method:

```python
def find_by_color(self, jersey: str, color: str) -> List[Dict]:
    jersey = str(jersey).strip().lstrip("#")
    results = []
    for team_name, players in self._rosters.items():
        if jersey in players:
            results.append({"player": players[jersey], "team": team_name})
    return results
    # ← color parameter is never used
```

The `color` argument is received but completely ignored. The method
returns every team that has a player with that jersey number, regardless
of kit color.

How this breaks player attribution (align.py, match_by_jersey()):
1. VLM detects jersey #7, team_color="blue", but team_hint=None
   (color map didn't resolve it — e.g., white kit is always None)
2. Falls through to `roster_lookup.find_by_color("7", "blue")`
3. Both Blackburn (#7 = Armstrong) and Forest (#7 = some player) are returned
4. `len(candidates) > 1` → ambiguous → falls through to ESPN time match
5. If ESPN has no nearby event → player stays "UNKNOWN"

The intended behaviour: if color="blue" and the map says blue=Blackburn,
return only Blackburn's #7. The disambiguation never happened.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIXES
════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
### Fix A — buffer.py: OR → AND (line 141)

Before:
```python
if time_close or clip_overlap:
```

After:
```python
if time_close and clip_overlap:
```

With AND:
- An event is only a duplicate if BOTH its video_time AND its clip_start
  are within the dedup_window
- This correctly identifies overlap duplicates (same real event seen in
  two overlapping clips — both close in time AND in adjacent clips)
- Different events in the same time window but at genuinely different
  times are now correctly kept

────────────────────────────────────────────────────────────────
### Fix B — roster_lookup.py: add set_color_map() + fix find_by_color()

**Change 1 — __init__: add _color_map field**
```python
self._color_map: dict = {}
```

**Change 2 — new method set_color_map()** (after load_manual()):
```python
def set_color_map(self, color_map: dict):
    """Store the color→team map so find_by_color() can filter by team."""
    self._color_map = color_map or {}
```

**Change 3 — fix find_by_color():**
```python
def find_by_color(self, jersey: str, color: str) -> List[Dict]:
    jersey      = str(jersey).strip().lstrip("#")
    color_lower = color.lower().strip() if color else ""

    resolved_team = None
    if self._color_map and color_lower:
        resolved_team = self._color_map.get(color_lower)
        if not resolved_team:
            # partial match: "blue/white" → check if "blue" is in map
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
    return results
```

If `_color_map` is empty (not set), behaviour falls back to the old
behaviour — returns all teams. No regression for callers that don't
call set_color_map().

────────────────────────────────────────────────────────────────
### Fix C — main.py: wire color_map into roster

Before:
```python
load_colors_from_espn(match_date, team1, team2)
```

After:
```python
color_map = load_colors_from_espn(match_date, team1, team2)
roster.set_color_map(color_map)
```

`load_colors_from_espn()` already returned the color_map dict
(action_recognizer.py line ~242: `return color_map`) but the
return value was being discarded. Now it is captured and passed
to the roster so find_by_color() can use it.

════════════════════════════════════════════════════════════════
## PART 3 — PROOF THIS WORKS
════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
### Bug A proof: two fouls in adjacent clips

Setup:
  - clip 1: start=0s, Foul at video_time=10.2s
  - clip 2: start=30s, Foul at video_time=55.8s
  - dedup_window = 30.0

Before fix (OR):
  time_close   = |10.2 - 55.8| = 45.6 > 30 → False
  clip_overlap = |0.0 - 30.0|  = 30.0 ≤ 30 → True
  False or True → True → second foul DROPPED  ✗

After fix (AND):
  time_close   = 45.6 > 30 → False
  clip_overlap = 30.0 ≤ 30 → True
  False and True → False → second foul KEPT  ✓

Legitimate duplicate (same real foul, seen in both clips):
  - clip 1: start=0s, Foul at video_time=14.2s
  - clip 2: start=30s, Foul at video_time=14.5s
  time_close   = |14.2 - 14.5| = 0.3 ≤ 30 → True
  clip_overlap = |0 - 30| = 30 ≤ 30 → True
  True and True → True → duplicate correctly removed  ✓

────────────────────────────────────────────────────────────────
### Bug B proof: color disambiguation trace

Setup:
  - color_map = {"blue": "Blackburn Rovers", "red": "Nottingham Forest"}
  - VLM detected jersey="7", team_color="blue/white"
  - team_hint = None (resolve_team("blue/white") returned None because
    "blue/white" is not an exact key in the map)

Before fix (color ignored):
  find_by_color("7", "blue/white")
    → scans all teams
    → Blackburn has #7 (Armstrong) → add
    → Forest has #7 (some player)  → add
    → returns 2 candidates → ambiguous → falls to ESPN time match
    → no ESPN event nearby → player = "UNKNOWN"  ✗

After fix (color used via partial match):
  find_by_color("7", "blue/white")
    → color_lower = "blue/white"
    → _color_map.get("blue/white") → None (no exact match)
    → partial: "blue" in "blue/white" → resolved_team = "Blackburn Rovers"
    → scan rosters: Forest skipped (team_name != "Blackburn Rovers")
    → Blackburn #7 → Armstrong added
    → returns 1 candidate → player = "Adam Armstrong" ✓

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- If two teams have similar kit colors that map to the same color name
  (kit color collision — Problem #5), resolved_team may still be wrong.
  Example: two teams both wearing navy blue → both map to "blue" →
  color map can only hold one. This is a separate problem requiring
  better color distance logic.

- The ESPN consume key collision (Problem #8) where two events of the
  same type within 30 seconds get the same consume key is not fixed here.

- The Tier 2 time-based fallback in align.py still has a ±2 min tolerance
  that can produce wrong player matches. This is not addressed here.

- find_by_color() falls back to returning all teams if _color_map is empty
  (e.g., ESPN API failed to return colors). This is the correct fallback —
  better to be ambiguous than to confidently return the wrong team.
