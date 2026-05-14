# Fix 013 — ESPN Confirmation Gate for Shot Events

**Status:** Applied  
**Files changed:** `src/3_buffer_matcher/align.py`, `main.py`

---

## Problem

35-clip test produced 4 FP Shot detections. Several were unmatched by ESPN
(no ESPN Shot/Goal nearby), but ESPN DID have a different event (Foul, Free_Kick,
Corner) at nearly the same time. This means the real action was something else
and the VLM misclassified it. Since unmatched events still entered the KG, these
FPs polluted the graph.

---

## Fix

In `match_by_time()`, before returning the standard `unmatched` result for Shot:

1. Search ALL ESPN events within the time tolerance window (not just Shot/Goal).
2. If the nearest event is NOT Shot or Goal AND VLM confidence < 0.75, return
   a MatchedEvent with `match_method = "gated"`.

A second guard (for when ACTION_MAP is broadened in future) checks the actual
matched ESPN event: if a candidate was found but its action is not Shot/Goal and
confidence < 0.75, also set `match_method = "gated"`.

In `main.py`, gated events are skipped entirely (not ingested into KG):
```python
if m.match_method == "gated":
    print(f"   GATED  {m.gametime}  {m.action} → conf={m.confidence:.2f} ...")
    continue
```

---

## Rationale

Threshold alone cannot distinguish a genuine low-confidence Shot from a misclassified
Foul. ESPN serves as a weak negative signal: "something happened here, but it wasn't
a Shot/Goal." Combined with confidence < 0.75, this is strong enough to discard.

Shots with confidence ≥ 0.75 are kept even if ESPN contradicts — the VLM may have
detected the shot before the referee's whistle registered.

---

## What this does not fix

FP Shots with confidence ≥ 0.75 (or where ESPN has no event at all) still enter the
KG as unmatched events. Further prompt refinement needed for those.
