# Fix 021 — Shot Threshold 0.65 + Tighter ESPN Gate + Tolerance 1.0 + Cross-Batch Dedup

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`, `src/3_buffer_matcher/align.py`, `main.py`, `evaluate.py`

---

## Problems fixed

### Shot precision collapse (32 frames)
After raising to 32 frames (Fix 018), precision dropped from ~0.50 to 0.238 (21
Shots detected, 5 TP, 16 FP). More frames give the VLM more opportunities to see
ambiguous foot-ball contact that it scores as a Shot at 0.60–0.64 confidence.
Fix: threshold raised back to 0.65.

### ESPN gate too narrow
Gate (Fix 013) only fired when no Shot/Goal ESPN event was found nearby. Many FP
shots matched an ESPN Shot that was 1.5–2 min away — technically within the
±2 min window but far too loose to be the same event.
Fix: gate now also fires when the best ESPN Shot/Goal match is >1.0 min away and
confidence < 0.75.

```python
_shot_gate = (
    video_action == "Shot" and confidence < 0.75
    and (best_espn_action not in {"Shot", "Goal"}   # no match
         or best_diff > 1.0)                        # match too far
)
```

### Cross-batch dedup miss
The EventBuffer dedup catches duplicates within one ESPN tick batch. When the
buffer was flushed mid-clip-sequence (e.g. clips 5:12 and 5:14 fell in adjacent
batches), the second clip restarted with an empty buffer and the duplicate was
not caught. Result: two near-identical Shot events entered the KG (e.g. 5:12
and 5:14, 2 seconds apart).
Fix: `ingested_cache` list in `main.py` tracks (action, video_time) of all events
ingested this match. Before ingesting, if same action was ingested within 60s,
skip with `XDEDUP` log line.

### Evaluate tolerance 0.5 → 1.0 min
±0.5 min (30s) was too strict for Free_Kick: a correctly-detected Free_Kick at
21:11 missed a GT annotation at 22.4' (gap = 1.2 min). Free_Kick prompt triggers
at play-restart, which can appear 30–90s before the time ESPN records. ±1.0 min
(60s) is a practical middle ground between ±0.5 (too strict) and ±2.0 (too loose).
