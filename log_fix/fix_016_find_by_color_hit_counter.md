# Fix 016 — find_by_color() Hit Counter Fix

**Status:** Applied  
**Files changed:** `src/1_video_processor/roster_lookup.py`

---

## Problem

`RosterLookup.find_by_color()` never incremented `_tier1_attempts` or `_tier1_hits`,
so the Tier 1 jersey hit-rate report at pipeline end showed 0/0 (or only counted
`find()` calls). This made it look like jersey matching had zero hits when
color-based lookups were succeeding.

Evaluation output showed e.g. "Brentford 0/0 jersey hit rate" because all
Brentford lookups went through `find_by_color()`.

---

## Fix

Two lines added to `find_by_color()`:

```python
self._tier1_attempts += 1   # at top of method
...
if results:
    self._tier1_hits += 1   # after building results list
```

One attempt per call; one hit if any player was found (regardless of how many
candidates). This matches the semantics of `find()` — the caller resolves
ambiguity separately.

---

## What this does not fix

Ambiguous results (len > 1) still count as a hit. If the caller discards all
candidates due to ambiguity, the hit is slightly over-counted. This is a known
approximation — precise hit-rate for color lookups would require tracking
call-through to the disambiguation step in `match_by_jersey()`.
