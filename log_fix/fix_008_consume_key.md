# FIX 008 — ESPN Consume Key Rounding Too Coarse (30 s → 6 s)

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/2_web_scraper/espn_scraper.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

The ESPN consume key prevents duplicate KG nodes when two overlapping
clips both match the same real ESPN event. The key is:

```python
def _consume_key(self, time: float, action: str) -> Tuple:
    time_rounded = round(time * 2) / 2   # nearest 0.5 min = 30 s
    return (time_rounded, action.strip())
```

Rounding to 0.5 minutes (30 seconds) means two events of the same type
that occur within 30 seconds of each other are treated as the same event:

| Event | Time (min) | Rounded | Key |
|-------|-----------|---------|-----|
| Foul A | 9.00  | 9.0 | ("9.0", "Foul") |
| Foul B | 9.25  | 9.0 | ("9.0", "Foul") ← collision |

After Foul A is consumed (matched), Foul B's key is already in `_consumed`
even though it hasn't been matched yet. On the next align pass, Foul B
appears consumed and is silently skipped.

In a typical 90-minute match, fouls cluster in bursts — two fouls within
30 seconds of each other are a common occurrence (especially near set
pieces). This bug silently drops the second foul from ever being matched
to a VLM event.

**Interaction with Fix 003 (dedup OR→AND):** Fix 003 ensures the VLM-side
buffer keeps both fouls. Without this fix, the ESPN-side would still
consume both with a single key, so the VLM's second foul would correctly
survive the buffer but find no ESPN event to match against.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

Changed rounding from 0.5 minutes (30 s) to 0.1 minutes (6 s):

Before:
```python
time_rounded = round(time * 2) / 2   # nearest 0.5 min (30 s)
```

After:
```python
time_rounded = round(time * 10) / 10  # nearest 0.1 min (6 s)
```

6 seconds is fine-grained enough that two fouls at 9:00 and 9:25 get
distinct keys — their rounded times are 9.0 and 9.4 respectively.

At the same time, 6 seconds is coarse enough that genuine floating-point
noise in ESPN time values (e.g. 9.000 vs 9.001, arising from integer
seconds ÷ 60) still collapses to the same bucket:

| Scenario | Before (0.5 min bucket) | After (0.1 min bucket) |
|----------|------------------------|------------------------|
| Foul at 9:00 → 9.000 min | key "9.0" | key "9.0" |
| Same foul, float noise 9.001 min | key "9.0" | key "9.0" (same bucket) ✓ |
| Second foul at 9:25 → 9.417 min | key "9.0" ← collision ✗ | key "9.4" ✓ |
| Two fouls at 9:00 and 9:06 | key "9.0" ← collision ✗ | key "9.0" and "9.1" ✓ |

The minimum event separation resolvable by this key is 6 seconds. Two
ESPN events of the same type within 6 seconds are vanishingly rare in
football (referees and commentary editors don't record them that closely),
so this is not a practical limitation.

════════════════════════════════════════════════════════════════
## PART 3 — WHY NOT SMALLER GRANULARITY?
════════════════════════════════════════════════════════════════

ESPN time values come from the API as integer seconds divided by 60:
  e.g., 540 seconds → 9.0 minutes exactly, no float error

The float error risk is low. But if ESPN ever returns non-integer seconds
(e.g. 540.5s → 9.008 min), 6-second buckets (0.1 min) still absorb this.
Using 1-second buckets (0.017 min) would also work but is unnecessarily
precise given ESPN time resolution.

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- Two ESPN events of the same type within 6 seconds will still share
  a consume key and the second will be silently skipped. This is
  extremely unlikely in practice but remains a theoretical gap.
- The consume mechanism only prevents duplicate ESPN→VLM matching.
  It doesn't deduplicate VLM events themselves — that is handled by
  EventBuffer (Fix 003).
