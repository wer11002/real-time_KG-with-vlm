# Fix 014 — Free_Kick Prompt: Softened Criteria

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`

---

## Problem

35-clip test (35 min) had 1 FN Free_Kick at 22.4'. The previous criteria required
a hard wall of defenders, which is not always visible (camera angle, early free
kicks, indirect free kicks). The overly strict wall requirement caused the VLM to
skip Free_Kicks it correctly identified but could not confirm with a wall.

---

## Fix

Free_Kick criteria in `ACTION_PROMPT` changed from:

> "player kicks a stationary ball with a wall of defenders present,  
>  OR referee is shown pointing to a spot before the kick."

to:

> "referee blows whistle and a player takes a stationary kick,  
>  OR a wall of defenders is visible near a stationary ball,  
>  OR players from both teams are standing around a stationary ball  
>  waiting for a kick to be taken."

The "both teams standing around a stationary ball" condition captures the common
game-reset scenario where a free kick is about to be taken without a visible wall
(e.g. quick free kicks, far-field restarts, indirect free kicks).

---

## What this does not fix

A cross from open play could satisfy "stationary ball + players standing around" if
the VLM misidentifies a throw-in or corner restart. Corner has its own criteria
(corner arc flag) so Corner events are not at risk, but throw-ins could produce
false Free_Kick detections. Monitor.
