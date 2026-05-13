# FIX 005 — Unmatched-Player Events Discard VLM-Identified Team

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/4_kg_builder/kg_builder.py`
- `src/3_buffer_matcher/align.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

In `kg_builder.py`, `ingest_matched_event()` has two branches depending on
whether the ESPN alignment succeeded:

```python
if matched.matched and matched.player:
    team_id = get_or_create_team(matched.team, ekg) if matched.team else None
    player_id, is_new_player = get_or_create_player(...)
    ...
else:
    player_id = None
    team_id   = None     # ← BUG
```

When the VLM detects an action (e.g. a Shot) but ESPN alignment fails
(either no ESPN event nearby, or player is UNKNOWN), we fall into the
`else` branch. Here, `team_id = None` is unconditionally set — even when
the VLM *did* identify which team was involved via kit color or team hint.

**Consequence:** Every unmatched event in the KG has no team link.
A SPARQL query for "how many shots did Blackburn take?" will miss all
VLM-detected shots that ESPN didn't record. This defeats the purpose of
keeping unmatched events in the graph at all.

**Scope:** In a typical 90-minute match the VLM detects ~80-120 events.
ESPN covers ~30-40. The remaining 40-80 unmatched events all lose their
team link, even though the VLM usually identifies team color correctly.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

### Change A — kg_builder.py: use matched.team in else branch

Changed the `else` branch from:

```python
else:
    player_id = None
    team_id   = None
```

To:

```python
else:
    player_id = None
    team_id   = get_or_create_team(matched.team, ekg) if matched.team else None
```

This is strictly additive: events that were already matched (ESPN hit)
are unaffected. Only unmatched events that carry a VLM-resolved team now
get that team written into the KG.

────────────────────────────────────────────────────────────────
### Change B — align.py: propagate VLM team to unmatched MatchedEvent

**This change is required for Change A to have any effect.**

`matched.team` is populated from the `MatchedEvent.team` field. But in
`match_by_time()`, the unmatched path was:

```python
return MatchedEvent(
    ...
    matched      = False,
    match_method = "unmatched",
    # team field NOT set → defaults to None
)
```

The VLM-resolved team (`video_event.get("team")`) was already computed by
`detect_actions()` → `resolve_team(team_color)` and stored in the video
event dict, but was never propagated to the MatchedEvent.

Fixed by adding `team=video_event.get("team")` to the unmatched return:

```python
return MatchedEvent(
    ...
    matched      = False,
    team         = video_event.get("team"),   # preserve VLM-resolved team
    match_method = "unmatched",
)
```

Without this change, `matched.team` would always be None for unmatched
events regardless of the kg_builder fix, because the team was lost at
the align.py layer before it could reach kg_builder.

`matched.team` is set to a team name only when:
- `video_event["team"]` is non-None (color map resolved `team_color` → team name)
- If color was ambiguous, white, or color_map was empty → `video_event["team"]`
  is None → `matched.team` stays None → `team_id` stays None (correct)

════════════════════════════════════════════════════════════════
## PART 3 — PROOF THIS WORKS
════════════════════════════════════════════════════════════════

Scenario: VLM detects Shot by "blue" kit at 14:22.
  - ESPN has no Shot near 14:22 → matched.matched = False, matched.player = None
  - color_map = {"blue": "Blackburn Rovers"} → align.py sets matched.team = "Blackburn Rovers"

Before fix:
  - else branch → team_id = None
  - Event node created with no INVOLVED_IN edge
  - SPARQL: "shots by Blackburn" → miss this event  ✗

After fix:
  - else branch → team_id = get_or_create_team("Blackburn Rovers", ekg)
  - Event node has INVOLVED_IN edge to Blackburn team node
  - SPARQL: "shots by Blackburn" → correctly counts this event  ✓

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- If `matched.team` is None (color was white, ambiguous, or unresolved),
  the event still has no team link. This is correct — it's better to have
  no team than a wrong team.
- Kit color collision (Problem #5, fixed separately in Fix 006) still
  causes color_map to be empty, so many events will have matched.team = None
  until Fix 006 reduces collisions.
