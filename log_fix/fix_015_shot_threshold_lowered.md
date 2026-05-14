# Fix 015 — Shot Threshold Lowered 0.65 → 0.60

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`

---

## Problem

35-clip test had 2 FN Shots (at 26.1' and 30.4'). These were genuine shots that
the VLM detected but assigned confidence in the 0.60–0.64 range, which the 0.65
threshold discarded.

---

## Fix

```python
CONFIDENCE_THRESHOLDS["Shot"] = 0.60  # was 0.65
```

This threshold lowering is safe now that Fix 013 (ESPN confirmation gate) is in
place. Without the gate, lowering the threshold would have increased FPs. With the
gate, any low-confidence Shot that has a contradicting ESPN event is discarded
before entering the KG, so the FP risk from threshold reduction is bounded.

---

## Interaction with Fix 013

Threshold: 0.60 ≤ conf < 0.75 → VLM keeps the Shot  
Gate check: if ESPN has a nearby non-Shot/Goal event AND conf < 0.75 → discard  
Result: the only low-confidence Shots that enter the KG are those with NO contradicting
ESPN event — a more permissive but safer set.
