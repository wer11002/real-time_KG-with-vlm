# Fix 011 — Kit Detail Fields: Shorts, Socks, Kit Pattern

**Status:** Applied  
**Files changed:** `src/1_video_processor/action_recognizer.py`, `src/3_buffer_matcher/align.py`, `src/4_kg_builder/ekg_schema.py`, `src/4_kg_builder/kg_builder.py`, `main.py`

---

## Problem

`team_color` captured only the jersey top color (e.g. "blue"). Two teams with
similar tops (navy vs blue) collide. No way to distinguish them without more
kit information.

---

## Fix

VLM prompt extended to report three additional kit fields per detection:
- `shorts_color` — colour of the shorts (e.g. "white")
- `socks_color`  — colour of the socks (e.g. "blue")
- `kit_pattern`  — one of "solid", "striped", "hooped", or null

These flow through the full pipeline:

| File | Change |
|------|--------|
| `action_recognizer.py` | Three new fields added to JSON output schema in `ACTION_PROMPT`; `detect_actions()` extracts and returns them |
| `align.py` | `MatchedEvent` gains `shorts_color`, `socks_color`, `kit_pattern`; both `match_by_jersey()` and `match_by_time()` pass them through |
| `ekg_schema.py` | `DATATYPE_PROPERTIES` extended with `hasDetectedShortsColor`, `hasDetectedSocksColor`, `hasKitPattern` (all `XSD.string`) |
| `kg_builder.py` | `_create_event_node()` and `ingest_matched_event()` store the three triples |
| `main.py` | `ingest_matched_event()` call passes `shorts_color`, `socks_color`, `kit_pattern` |

---

## What this does not fix

Team disambiguation from kit details is not yet implemented; the fields are stored
as debug data. Color collision (Fix 006 partial) still applies.
