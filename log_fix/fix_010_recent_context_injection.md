# Fix 010 — Recent Context Injection: Suppress Same-Action Re-Detection

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`, `src/3_buffer_matcher/buffer.py`, `main.py`

---

## Problem

Overlapping 60s clips (30s step) mean the VLM processes the same 30s of footage
twice. Even with dedup (Fix 003), the VLM could detect the same action in both
clips at slightly different timestamps, and the 30s dedup window might not catch
the second detection if the clip_start gap is > 30s.

---

## Fix

**buffer.py** — added `get_recent(current_sec, minutes=2.0)` to retrieve the last
2 minutes of buffer events.

**action_recognizer.py** — `detect_actions()` gains `recent_events` parameter.
When provided, formats them into a context string appended to the prompt:
> "Recent detections in the last 2 minutes: Shot at 1st 09:14, ...  
> Do NOT re-detect the same action type within 30 seconds of a prior detection."

**main.py** — before each `detect_actions()` call, fetches recent buffer events
and passes them as `recent_events`.

---

## What this does not fix

Identical action at the same time but detected in a clip > 2 minutes later
(very unlikely with 30s step). The dedup window in EventBuffer still handles
adjacent-clip overlap.
