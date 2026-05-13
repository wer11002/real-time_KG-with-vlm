# FIX 001 — Timestamp Gametime Wrong (Clip Start Instead of Frame Time)

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/1_video_processor/action_recognizer.py`
- `src/3_buffer_matcher/buffer.py`
- `main.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

Every detected action was being stamped with the clip's START time,
not the time of the specific frame where the action was seen.

Example of what was happening:
- Clip starts at 540 seconds into the match
- VLM sees a shot at frame 5 out of 8 frames
- Frame 5 is at roughly 540 + (5/8 × 60) = 577.5 seconds = 9 minutes 37 seconds
- But the stored gametime was "1st 09:00" (the clip start)
- The correct gametime should be "1st 09:37"

This made every timestamp off by 0 to 60 seconds depending on
which frame the action appeared in. For a commentator output
showing [MM:SS], this is a visible and meaningful error.

────────────────────────────────────────────────────────────────
### Evidence in the code (before the fix)

In `main.py` line 226:
```python
gametime = seconds_to_gametime(start_sec, halftime_sec)
```
This computes gametime from start_sec — the clip's start, not the frame.

In `main.py` line 244:
```python
n_added = buffer.add_from_detections(detections, start_sec, gametime)
```
This passes that clip-start gametime to the buffer.

In `buffer.py` line 58–59 (make_video_event):
```python
gametime = gametime,   # <-- just stores whatever was passed in
```
No adjustment — the clip-start gametime is stored as-is.

Meanwhile in `action_recognizer.py` lines 452–453:
```python
time_in_clip = estimate_time(raw.get("frame_index"), frame_times, duration_sec)
video_time   = clip_start_sec + time_in_clip    # ← correct per-frame seconds
```
The `video_time` WAS correctly computed per frame. But it was never
converted back to a gametime string. The gametime that travelled
through the pipeline was always the clip start.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

Three files changed. Each change is described separately below.

────────────────────────────────────────────────────────────────
### Change A — action_recognizer.py

**What was added:**

A private helper function `_seconds_to_gametime()` was added
directly into action_recognizer.py. It uses the same formula
as the existing `seconds_to_gametime()` in sliding_window.py.

Why not import from sliding_window.py?
Because sliding_window.py has module-level path definitions
(MATCH_DIR, DATA_DIR, etc.) that run on import and are
hardcoded to a specific match folder. Importing it from
action_recognizer would cause side effects and make the
module less portable. Inlining the 10-line formula is safer.

```python
def _seconds_to_gametime(seconds: float, halftime_sec: float) -> str:
    if seconds < halftime_sec:
        half    = "1st"
        minutes = int(seconds // 60)
        secs    = int(seconds % 60)
    else:
        half    = "2nd"
        adj     = seconds - halftime_sec
        minutes = int(adj // 60)
        secs    = int(adj % 60)
    return f"{half} {minutes:02d}:{secs:02d}"
```

`detect_actions()` signature was changed from:
```python
def detect_actions(clip_path: str, clip_start_sec: float = 0.0) -> List[Dict]:
```
to:
```python
def detect_actions(clip_path: str, clip_start_sec: float = 0.0,
                   halftime_sec: float = 2700.0) -> List[Dict]:
```
Default is 2700.0 (45 minutes) so old callers that don't pass
halftime_sec will not break — they just get a slightly wrong
second-half boundary, same as before.

Inside the detection loop, `"gametime"` is now added to each dict:
```python
detections.append({
    ...
    "video_time"  : video_time,          # was already correct
    "gametime"    : _seconds_to_gametime(video_time, halftime_sec),  # NEW
})
```
This converts the already-correct `video_time` (absolute seconds)
into the human-readable gametime string at the right per-frame time.

────────────────────────────────────────────────────────────────
### Change B — buffer.py

**What was changed:**

In `make_video_event()`, one line changed:

Before:
```python
gametime = gametime,
```

After:
```python
gametime = detected.get("gametime", gametime),
```

This means: use the gametime from the detection dict (the per-frame
one computed in action_recognizer) if it is present. Fall back to
the clip-start gametime only if the key is missing — for example
if some other code calls make_video_event() without the new field.

────────────────────────────────────────────────────────────────
### Change C — main.py

**What was changed:**

The call to `detect_actions()` now passes `halftime_sec`:

Before:
```python
detections = detect_actions(str(clip_path), clip_start_sec=start_sec)
```

After:
```python
detections = detect_actions(str(clip_path), clip_start_sec=start_sec,
                            halftime_sec=halftime_sec)
```

`halftime_sec` is already computed earlier in `run_match()` at line 181
from the match's Labels-ball.json file:
```python
halftime_sec = get_halftime_sec(labels_path)
```
So no new computation was needed — the value already existed in scope.

════════════════════════════════════════════════════════════════
## PART 3 — PROOF THIS WORKS
════════════════════════════════════════════════════════════════

This is not guesswork. Here is a full trace of the data flow
after the fix, using a concrete example.

────────────────────────────────────────────────────────────────
### Trace: Shot detected at frame 5, clip starting at 540s

Inputs:
- clip_start_sec = 540.0
- halftime_sec   = 2764.0  (from Labels-ball.json: "1 - 46:04")
- NUM_FRAMES     = 8
- clip at 25fps  → total_frames ≈ 1500

How frame_times is actually built (extract_frames, lines 261–270):
```python
indices = np.linspace(0, 1499, 8, dtype=int)
        = [0, 214, 428, 642, 856, 1070, 1284, 1499]

frame_times = [idx / fps for idx in indices]
            = [0.0, 8.56, 17.12, 25.68, 34.24, 42.80, 51.36, 59.96]
```
Note: these are NOT uniform 7.5s gaps — they are actual frame
positions converted to seconds via real fps. The spacing is
approximately 8.5s between frames, not 7.5s.

VLM output: {"action": "Shot", "frame_index": 5, ...}

Step 1 — estimate_time():
```
idx            = 5 - 1 = 4          (frame_index is 1-based → 0-based)
frame_times[4] = 34.24s             (frame 5 is 34.24 seconds into clip)
time_in_clip   = 34.24
```

Step 2 — video_time:
```
video_time = 540.0 + 34.24 = 574.24 seconds
```

Step 3 — _seconds_to_gametime(574.24, 2764.0):
```
574.24 < 2764  → first half
minutes = int(574.24 // 60) = 9
secs    = int(574.24 % 60)  = 34
result  = "1st 09:34"
```

Step 4 — detection dict now contains:
```python
{
    "action"    : "Shot",
    "video_time": 574.24,
    "gametime"  : "1st 09:34",   ← was "1st 09:00" before the fix
    ...
}
```

Step 5 — buffer.py make_video_event():
```python
gametime = detected.get("gametime", gametime)
         = "1st 09:34"   ← uses the detection dict value, not clip start
```

Step 6 — VideoEvent stored with:
```
video_time = 574.24
gametime   = "1st 09:34"   ← correct
```

BEFORE the fix, the same event would have been stored as:
```
video_time = 574.24      (was already correct — untouched)
gametime   = "1st 09:00" (was WRONG — clip start 540s → 9 min 0 sec)
```

────────────────────────────────────────────────────────────────
### Why the default halftime_sec = 2700.0 is safe

The default 2700.0 (45:00) is only used if the caller forgets to
pass halftime_sec. In the actual pipeline, main.py always reads
the real halftime from Labels-ball.json and passes it. The default
only matters for standalone testing of action_recognizer.py.

The worst case from using the wrong halftime_sec is a wrong half
label on second-half events (e.g., "1st 47:00" instead of "2nd 01:00").
This was already the existing bug before this fix. The fix makes it
correct when halftime_sec is passed, and no worse than before when it is not.

────────────────────────────────────────────────────────────────
### Precision gain from this fix

With 8 frames across a 60-second clip at 25fps:
- Frame spacing ≈ 60 / 8 = 7.5 seconds per frame in theory
- Actual spacing from linspace: ≈ 8.5s between frames
  (linspace(0, 1499, 8) gives gaps of ~214 frames → 214/25 = 8.56s)
- Maximum error after this fix = ±8.56 / 2 ≈ ±4.3 seconds
  (the action happened somewhere in the ~8.5s window around that frame)
- Maximum error BEFORE this fix = up to 60 seconds
  (the entire clip length, if the action was in the last frame)

The fix reduces worst-case timestamp error from 60s to ~4s.
For a commentator output at [MM:SS] resolution, this is the
difference between showing [09:00] and [09:34] for the same event.

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

This fix does not address:
- The VLM's frame_index being imprecise (the VLM may report
  frame 5 even if the action is more visible in frame 4)
- The ±3.75 second residual error from frame spacing
- The 2-minute tolerance in evaluate.py (can be tightened now,
  but that is a separate change)
- Any of the other 3 problems (over-detection, dedup, color collision)
