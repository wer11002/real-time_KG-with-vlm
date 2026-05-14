# Fix 009 — VLM Prompt Strictness: SYSTEM_PROMPT + Per-Action Criteria + 16 Frames

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`

---

## Problem

After the confidence filter (Fix 002), the VLM still hallucinated events — scoring
false Goals and Shots at 0.90–0.95 confidence even when no action was happening.
Root cause: the default VLM behaviour is to "say something" rather than nothing.
720p resolution was tested but made no difference; the issue was the prompt, not
resolution.

Evidence: 12-clip test → 14 detections with precision 14%. Goal hallucination at
1:53 survived 0.80 threshold.

---

## Fix

1. **SYSTEM_PROMPT rewritten** to make the empty response the default:
   > "Your default answer is `{"actions": []}`. Only override when you have
   > unmistakable visual proof of a specific action. A missed detection is
   > always better than a false one."

2. **ACTION_PROMPT** replaced hardcoded "CLEARLY visible" with per-action strict
   criteria. Each action now has ALL-conditions-must-hold rules, e.g.:
   - **SHOT**: foot/head makes CONTACT with ball AND ball is moving toward goal
   - **GOAL**: ball crosses goal line inside net, OR celebration + scoreboard change
   - **FOUL**: physical contact visible AND player falls OR referee raises arm

3. **Explicit DO NOT list** added to ACTION_PROMPT:
   > Do NOT report: running/dribbling, clearances, replays, anything < 65% confident

4. **NUM_FRAMES raised from 8 → 16** to give the VLM more temporal context per clip.
   Frame timing line added to prompt so VLM knows each frame's offset.

5. **Per-action CONFIDENCE_THRESHOLDS** introduced:
   ```
   Goal: 0.85, Corner: 0.75, Shot: 0.65 (→ 0.60 in Fix 015),
   Foul: 0.65, Free_Kick: 0.60, Substitution: 0.55, Offside: 0.50
   ```

---

## Result

12-clip test (12 min): 2 detections, 2 TP, 0 FP — perfect precision.  
35-clip test (35 min): 9 detections, 4 TP, 5 FP — F1=0.500.  
Further improvements via Fix 013 (gate) and Fix 015 (threshold).
