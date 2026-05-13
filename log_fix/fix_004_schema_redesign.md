# FIX 004 — Schema Redesign via SKILL.md CQ + OOPS! Pipeline

**Date:** 2026-05-10
**Status:** Applied
**Methodology:** SKILL.md (`.claude/skills/ontology-pipeline/SKILL.md`)
**Files changed:**
- `src/4_kg_builder/ekg_schema.py`
- `src/4_kg_builder/kg_builder.py`
**New files:**
- `src/4_kg_builder/schema_eval.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE METHODOLOGY (from SKILL.md)
════════════════════════════════════════════════════════════════

The SKILL.md ontology pipeline scores a schema using two tools:

1. **SPARQL CQ check** — for each Competency Question, a SPARQL query
   is run against the T-Box only. If the query returns results (i.e.,
   the needed class or property exists), the CQ passes. This is a cheap
   structural check that catches missing vocabulary without needing A-Box data.

2. **OOPS! pitfall check** — the T-Box Turtle is sent to the OOPS! REST
   API (`https://oops.linkeddata.es/rest`). OOPS! returns pitfall codes
   categorised by severity. Critical codes (P05, P06, P19, P29) indicate
   wrong inverse declarations, cycle errors, or multi-domain properties.

This fix applies step 1 fully and step 2 partially (OOPS! server returned
HTTP 500 on the day of this fix — external service issue, not a code problem).
A manual OOPS! analysis was done in its place.

════════════════════════════════════════════════════════════════
## PART 2 — COMPETENCY QUESTIONS DEFINED
════════════════════════════════════════════════════════════════

13 CQs were defined covering the two downstream use cases (commentary and prediction).

### Commentary CQs
- CQ01: What events happened in match M, sorted by chronological order?
- CQ02: Who performed event E?
- CQ03: Which team was involved in event E?
- CQ04: What is the precise game time (as a sortable number) of event E?
- CQ05: What event directly preceded event E in the match timeline?
- CQ06: Who assisted goal G?
- CQ07: How many fouls has player P committed in match M?
- CQ08: Did player P receive a card in match M, and what type?

### Prediction CQs
- CQ09: What sequence of event types immediately preceded the last goal?
- CQ10: How many shots has team T taken in the first half?
- CQ11: Which team was player P playing for in match M (temporal)?
- CQ12: How many consecutive Shot events occurred before goal G?

### Pass CQ (new — Pass added to pipeline in Fix 002)
- CQ13: Were there any Pass events in the 5 minutes before a goal?

════════════════════════════════════════════════════════════════
## PART 3 — BEFORE SCORE (original schema)
════════════════════════════════════════════════════════════════

CQ score: **8/13 (61%)**

```
CQ01  ✗  hasTime is xsd:string ("1st 09:34") — not sortable numerically
CQ02  ✓
CQ03  ✓
CQ04  ✗  no numeric time property — can't filter ?t > 9.0
CQ05  ✓
CQ06  ✓
CQ07  ✗  no numeric time for match window filtering
CQ08  ✓
CQ09  ✓
CQ10  ✗  no hasPeriod — can't filter "first half shots"
CQ11  ✓
CQ12  ✓
CQ13  ✗  no PassEvent class
```

T-Box size: 124 triples / 4,196 chars Turtle

Manual OOPS! analysis (server unavailable):
- P08 (Missing annotations): no rdfs:comment on any class or property → minor
- P04 (Unconnected elements): hasFullText, hasDate have no rdfs:domain → minor
- P05/P06/P19/P29 (critical): NOT present — inverses correctly declared,
  no multi-domain properties, no cycles

════════════════════════════════════════════════════════════════
## PART 4 — THE FIX
════════════════════════════════════════════════════════════════

Four additions to `ekg_schema.py` and one change to `kg_builder.py`.

────────────────────────────────────────────────────────────────
### Addition 1 — PassEvent class

Added to CLASSES:
```python
"PassEvent": EKG.PassEvent,
```

Added to CLASS_HIERARCHY:
```python
(EKG.PassEvent, EKG.ActionEvent),
```

Added to EVENT_TYPE_CLASS:
```python
"Pass": EKG.PassEvent,
```

PassEvent is a subclass of ActionEvent (alongside ShotEvent, FoulEvent, etc.).
This gives Pass detections an OWL type — enabling class-based reasoning and
SPARQL queries like `?e a ekg:PassEvent`.

────────────────────────────────────────────────────────────────
### Addition 2 — hasMinute: xsd:decimal

```python
"hasMinute": XSD.decimal,
# e.g. 9.567 — numeric match minute for SPARQL ORDER BY / FILTER
```

Derived from gametime string "1st 09:34" → 9.567.
Stored alongside the existing `hasTime` (which is kept for display).
Unlocks CQ01, CQ04, CQ07 which all require numeric time comparison.

────────────────────────────────────────────────────────────────
### Addition 3 — hasPeriod: xsd:integer

```python
"hasPeriod": XSD.integer,
# 1 = first half, 2 = second half
```

Derived from gametime string prefix "1st" / "2nd".
Unlocks CQ10 (shots in first half = filter hasPeriod=1).

────────────────────────────────────────────────────────────────
### Addition 4 — isMatched: xsd:boolean (formally declared)

```python
"isMatched": XSD.boolean,
# True if VLM event was matched against ESPN/roster data
```

This property was already being written to the A-Box by kg_builder.py
as a provenance annotation, but was NOT declared in the T-Box. It is now
formally declared as a DatatypeProperty. This makes it queryable via SPARQL
and visible to OOPS! validation.

────────────────────────────────────────────────────────────────
### kg_builder.py change — write hasMinute + hasPeriod on events

In `_create_event_node()`, after writing `hasTime`:

```python
try:
    half, t  = time_raw.strip().split(" ", 1)
    mm, ss   = t.strip().split(":")
    period   = 1 if half == "1st" else 2
    minute   = int(mm) + int(ss) / 60.0
    ekg.g.add((event_uri, EKG.hasMinute, Literal(round(minute, 3), datatype=XSD.decimal)))
    ekg.g.add((event_uri, EKG.hasPeriod, Literal(period,           datatype=XSD.integer)))
except Exception:
    pass
```

Also changed `isMatched` to use `datatype=XSD.boolean` explicitly
(was written as a plain Literal before).

════════════════════════════════════════════════════════════════
## PART 5 — AFTER SCORE
════════════════════════════════════════════════════════════════

CQ score: **13/13 (100%)**

```
CQ01  ✓  hasMinute xsd:decimal enables ORDER BY
CQ02  ✓
CQ03  ✓
CQ04  ✓  hasMinute enables numeric FILTER comparison
CQ05  ✓
CQ06  ✓
CQ07  ✓  hasMinute + hasPeriod enable match window count
CQ08  ✓
CQ09  ✓
CQ10  ✓  hasPeriod = 1 filter now possible
CQ11  ✓
CQ12  ✓
CQ13  ✓  PassEvent OWL class exists
```

T-Box size: 136 triples / 4,586 chars Turtle (+12 triples, +390 chars)

════════════════════════════════════════════════════════════════
## PART 6 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- P08 (missing rdfs:comment) is not addressed — adding comments to all 15
  classes and 13+ properties is low priority and not needed for CQ coverage.
- P04 (missing rdfs:domain on some datatype properties) is not addressed —
  hasFullText, hasDate, hasConfidence do not have domain restrictions because
  they may appear on multiple types in the future.
- The OOPS! server returned HTTP 500 on the day of this fix. The manual
  analysis shows no critical pitfalls exist, but the formal OOPS! score
  has not been verified by the API. schema_eval.py will call the API
  automatically when it becomes available again.
- hasMinute for second-half events is the minute within the second half
  (0–45+), not the absolute match minute (45–90+). For absolute minute
  comparison across halves, callers must add `HALFTIME_SEC / 60` when
  hasPeriod = 2. This is a known limitation — the gametime string format
  "2nd 07:30" does not carry halftime offset.
