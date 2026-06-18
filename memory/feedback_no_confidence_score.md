---
name: feedback-no-confidence-score
description: Never use or suggest keeping hasConfidence/confidence score in the KG — user wants it removed entirely
metadata:
  type: feedback
---

Remove `ekg:hasConfidence` from the system entirely — do not suggest fixing its datatype, do not suggest keeping it.

**Why:** The confidence value is a pure neural network self-report from Qwen3-VL — the VLM just outputs a number with no formula, no pipeline logic, no ground truth backing. There is no meaningful interpretation for storing it. The user has stated this multiple times and is frustrated when the suggestion keeps coming back.

**How to apply:** Any time `hasConfidence` comes up (T-Box, kg_builder.py, migration scripts, evaluation), the action is DELETE, not fix. Do not propose "change xsd:float to xsd:double" or any datatype fix as an alternative. Just remove it.
