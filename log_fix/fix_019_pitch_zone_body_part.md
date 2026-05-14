# Fix 019 — Pitch Zone + Body Part: Commentary Context Fields

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`, `src/3_buffer_matcher/buffer.py`, `src/3_buffer_matcher/align.py`, `src/4_kg_builder/ekg_schema.py`, `src/4_kg_builder/kg_builder.py`, `main.py`

---

## Problem

95% of events had no player name (jersey lookup fails when jersey is not visible
or roster lookup misses). The LLM commentator could only say "a player shoots"
because `IS_PERFORMED_BY` linked to nobody and the description was a single
generic sentence with no location or technique.

---

## Fix

Two new VLM output fields requested in `ACTION_PROMPT`:

```
"pitch_zone": "penalty_box" | "edge_of_area" | "midfield" | "own_half" | "wing" | null
"body_part":  "right_foot"  | "left_foot"    | "header"   | null
```

`body_part` applies to Shot and Goal only; `pitch_zone` applies to all actions.

`description` instruction updated from "one sentence describing the action" to:
> "who (jersey#), what technique (left/right foot, header), where on pitch
>  (penalty box / edge of area / midfield)"

Pipeline wiring:

| File | Change |
|------|--------|
| `action_recognizer.py` | JSON schema extended; `detect_actions()` extracts and returns both fields |
| `buffer.py` | `VideoEvent` gains `pitch_zone`, `body_part`; `make_video_event()` passes them |
| `align.py` | `MatchedEvent` gains both fields; `match_by_jersey()`, `match_by_time()`, `align_buffer()` v_dict all pass them |
| `ekg_schema.py` | `hasPitchZone` (XSD.string) and `hasBodyPart` (XSD.string) declared in T-Box |
| `kg_builder.py` | `_create_event_node()` and `ingest_matched_event()` store both triples |
| `main.py` | `ingest_matched_event()` call passes `pitch_zone=m.pitch_zone, body_part=m.body_part` |

---

## Expected commentary improvement

Before: "A player in red kit shoots."  
After: "A right-footed effort from the edge of the penalty area by #7."

This is the highest-value commentary gain achievable without solving jersey
reading, because location + technique context requires no player identity.
