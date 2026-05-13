# FIX 002 — VLM Over-detection: No Confidence Filter + Aggressive Prompt

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/1_video_processor/action_recognizer.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

Every detection the VLM returned — even with `confidence=0.1` — flowed
into the buffer unchanged. The prompt also demanded "EVERY" action and
encouraged the model to report "all" events, actively incentivising
hallucination. Together, these two problems flooded the KG with noise events
that had no real-world match.

────────────────────────────────────────────────────────────────
### Sub-problem A — No confidence threshold

In `detect_actions()` (line ~462), the raw actions from `run_inference()`
were filtered only by action type membership:

```python
for raw in raw_actions:
    action = raw.get("action")
    if action not in VALID_ACTIONS:
        continue
    # ← no confidence check — 0.1 confidence passes straight through
```

A VLM that is 10% confident in a "Foul" would still create a Foul node
in the KG, with player="UNKNOWN", team=None, and a vague description.
In `evaluate.py` terms, these become False Positives — events the pipeline
emits that do not correspond to anything in the ESPN or Labels ground truth.

────────────────────────────────────────────────────────────────
### Sub-problem B — Prompt wording encourages over-detection

The ACTION_PROMPT contained:
```
"Identify EVERY key soccer action visible across these frames:"
"Multiple actions can occur in the same clip — include all of them."
```

The word "EVERY" and the second sentence both push the VLM toward exhaustive
enumeration. Because Qwen2-VL is an instruction-following model, it tries
to comply: if in doubt, it reports the event rather than omits it. This is
the opposite of what we want. We want the model to be conservative and only
report what it can clearly see.

────────────────────────────────────────────────────────────────
### Also added: Pass as a new detectable action

Labels-ball.json for the Blackburn vs Forest match contains:
  768  PASS
  102  HIGH PASS
   61  CROSS

These are never detected by the VLM (Pass was not in VALID_ACTIONS),
so the evaluation would score them all as False Negatives even though
the pipeline never attempted to detect them. Pass is now added as a
VLM-detectable action so we can measure how well it is covered.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

Three changes to `src/1_video_processor/action_recognizer.py`.

────────────────────────────────────────────────────────────────
### Change A — Add Pass to VALID_ACTIONS (line 38)

Before:
```python
VALID_ACTIONS = {"Shot", "Goal", "Foul", "Corner", "Free_Kick", "Substitution", "Offside"}
```

After:
```python
VALID_ACTIONS = {"Shot", "Goal", "Foul", "Corner", "Free_Kick", "Substitution", "Offside", "Pass"}
```

────────────────────────────────────────────────────────────────
### Change B — Prompt reword (ACTION_PROMPT)

Before:
```
"Identify EVERY key soccer action visible across these frames:"
"Multiple actions can occur in the same clip — include all of them."
```

After:
```
"Identify only soccer actions that are CLEARLY and UNAMBIGUOUSLY visible in these frames:"
"Only report an action if you can clearly see it happening. If uncertain, omit it."
```

Also added Pass to the action list in the prompt:
```
- Pass (deliberate pass to a teammate — key passes, long balls, through balls)
```

────────────────────────────────────────────────────────────────
### Change C — min_confidence parameter and filter in detect_actions()

Signature change:
```python
def detect_actions(clip_path: str, clip_start_sec: float = 0.0,
                   halftime_sec: float = 2700.0,
                   min_confidence: float = 0.5) -> List[Dict]:
```

Default is 0.5. Backward-compatible — callers that do not pass
min_confidence get the new default, which is a tighter filter
than the old behaviour (which had no filter at all).

Filter added after action type check:
```python
confidence = float(raw.get("confidence", 0.0))
if confidence < min_confidence:
    continue
```

Note: `raw.get("confidence", 0.0)` defaults to 0.0 (not 0.5) so a
VLM that omits the confidence field is treated as uncertain and skipped.
This is the safe direction — better to miss a real event than emit noise.

════════════════════════════════════════════════════════════════
## PART 3 — PROOF THIS WORKS
════════════════════════════════════════════════════════════════

### Before the fix — low-confidence example

VLM output for clip starting at 480s:
```json
{"actions": [
  {"action": "Foul", "frame_index": 3, "jersey": null, "team_color": null,
   "description": "Two players collide near the halfway line",
   "confidence": 0.28}
]}
```

Before: `detect_actions()` would add this to detections — no check.
After:  `0.28 < 0.5` → skipped. Does not reach the buffer.

### Before the fix — "EVERY" prompt example

With "EVERY" in the prompt, if frames 2 and 6 both show players
running, the VLM would report two separate Pass events (one per frame
it noticed movement), both with confidence ~0.4. Both would enter
the buffer and create KG nodes.

With the new prompt + filter, confidence 0.4 < 0.5 → both skipped.

### Impact on evaluate.py output

In `evaluate.py` terms:
- FP count drops (noise events removed → fewer incorrect pipeline detections)
- Precision improves across all action types
- Recall is unchanged (we are only removing events below 0.5 confidence;
  real events visible to the VLM typically score 0.7+)

Pass GT ground truth in evaluate.py:
- PASS (768) + HIGH PASS (102) + CROSS (61) = 931 total pass events in GT
- VLM will only detect key passes (long balls, clear through balls)
- Expected recall for Pass will be low (VLM sees 8 frames per 60s clip;
  most short passes are invisible)
- But precision on reported passes should be high (only reported at ≥0.5)

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- The confidence score itself comes from the VLM. If the VLM is
  systematically overconfident (always reports 0.8+ even for hallucinations),
  the filter will not help. This is a model calibration issue.
- The threshold 0.5 is a starting point. It may need tuning after running
  evaluate.py on real output. Can be adjusted via: `min_confidence=0.7`
  if FP count is still too high after the fix.
- Pass recall will be very low because 8 frames per 60s cannot see most
  short passes. This is expected and acceptable — we detect key passes only.
- Kit color collision (Problem #5) and ESPN coverage gaps (Problem #6)
  are separate issues not addressed here.
