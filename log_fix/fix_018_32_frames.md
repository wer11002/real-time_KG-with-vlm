# Fix 018 — NUM_FRAMES 16 → 32: Reduce Frame Sampling Gap

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`

---

## Problem

At NUM_FRAMES = 16 over a 60-second clip, one frame was sampled every 3.75s.
A shot on goal lasts approximately 0.5 seconds. In the worst case the entire
shot motion falls between two sampled frames and the VLM never sees it.

The two FN shots at 26.1' and 30.4' in the 35-clip evaluation were most likely
not landing on any sampled frame — not a failure of the prompt or threshold,
but a failure to capture the moment at all.

Additionally the hardcoded per-frame timestamp line in `ACTION_PROMPT`:
> "Frame 1=0s  Frame 2=4s  Frame 3=8s ..."
was wrong at 32 frames (timestamps would shift to ~1.9s intervals) and provided
no real benefit to the model.

---

## Fix

```python
NUM_FRAMES = 32   # was 16 — now ~1 frame every 1.9s
```

`ACTION_PROMPT` header updated to "These 32 frames … one frame roughly every
2 seconds." Hardcoded per-frame timestamp list removed.

---

## Expected impact

At 32 frames, a 0.5s shot spans ~0.26 frame intervals, making it virtually
certain to be captured in at least one frame. Memory/compute cost increases
linearly with frame count — monitor inference time per clip.

---

## What this does not fix

At 32 frames the model receives 4× more image tokens than at 8 frames. If the
model's effective context window is saturated, quality could plateau or degrade.
Monitor whether inference time grows proportionally or super-linearly.
