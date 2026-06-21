# Football Event Collection & LLM-Queryable Storage

## Project Purpose

**Current stage:** Data collection with minimum corruption — detect every football action from match video with accurate timestamps, correct player identity where possible, and no duplicates. The pipeline must produce clean, trustworthy events before any downstream use.

**Storage goal:** The Event Knowledge Graph (EKG) is a structured semantic store designed so that an LLM can easily read and use the collected events — to commentate, predict, or answer questions about a match.

**Downstream use (future):** An LLM queries the EKG to generate live commentary and event predictions. This is the motivating use case for the storage design, but is not being built yet.

---

## What This System Does

The pipeline watches football match video, detects actions using a Vision Language Model (VLM), and produces one primary output:

1. **Event Knowledge Graph (EKG)** — an RDF/OWL semantic graph linking events, players, teams, and matches. Designed as a structured store for LLM-based commentary and prediction queries.

ESPN data is used **only** for static reference: player names (jersey → name), team kit colors, and team names. ESPN is **not** used to validate or gate action detection.

---

## Architecture Overview

```
Video File (720p / 224p)
    │
    ▼
[1] Sliding Window          src/1_video_processor/sliding_window.py
    60s clips, 30s step     Extracts clips, converts time to gametime string
    │
    ▼
[2] Action Recognizer       src/1_video_processor/action_recognizer.py
    Qwen3-VL-30B-A3B        VLM reads 8 frames per clip
    Outputs: action, jersey, team_color, description, confidence, frame_index
    │
    ▼
[3] Event Buffer            src/3_buffer_matcher/buffer.py
    Deduplication           Removes duplicate detections from overlapping clips
    │
    ▼
[4] ESPN Scraper            src/2_web_scraper/espn_scraper.py
    Reference only          Roster (jersey→player), team colors, team names
    │
    ▼
[5] Alignment               src/3_buffer_matcher/align.py
    Enrichment only         Adds player name if jersey matches roster
                            All events pass through regardless of ESPN match
    │
    ├──► [6a] Commentator   src/commentator/commentator.py  (planned)
    │         Sorted by exact timestamp [MM:SS], saved as .txt and .json
    │
    └──► [6b] KG Builder    src/4_kg_builder/kg_builder.py
              RDF graph     Loads T-Box from ekg_tbox.ttl at startup,
                            writes A-Box with single-leaf-class typing
                            → data/kg_output/ekg.ttl
```

---

## Key Design Decisions

**VLM is the only action detector.** ESPN match events are enrichment, not ground truth. If ESPN has no record of a VLM-detected action, the action is still recorded with `player = "unidentified"`.

**Exact timestamps.** Each action timestamp is derived from `frame_index` within the clip:
```
exact_seconds = clip_start + (frame_index / num_frames) * clip_duration
```
Output format: `[MM:SS]` (e.g., `[09:37]`), not just `[09:00]`.

**ESPN reference scope.** ESPN provides:
- Roster: jersey number → player name
- Team primary color (hex) → color name for VLM kit matching
- Kit colors (jersey/pants/socks) for VLM prompt context
- Team names for display

ESPN does **not** gate, filter, or validate VLM detections.

---

## Known Problems (Active Work)

| Problem | Status |
|---------|--------|
| VLM over-detection — 0.5 confidence filter applied, prompt updated to "CLEARLY visible only"; borderline clips still risk false positives | Monitor |
| Overlap duplication — dedup uses AND logic (time_close AND clip_overlap); fixed from previous OR bug | Fixed |
| Kit color collision — similar-shade teams map to same color name; raw color now stored as `hasDetectedColor` for debugging | Partial |
| ESPN coverage gaps — tackles/headers/clearances not in ESPN, stored as `isMatched=false` with VLM description | By design |
| `hasMinute` is minute-within-half (0–45+), not absolute match minute — cross-half SPARQL queries must add `HALFTIME_SEC / 60` when `hasPeriod = 2` | Known limitation |
| No commentator module yet | Fixed (commentator.py + downstream scripts now use new T-Box property names) |
| Ontology redesign for prediction use case | In progress (clean T-Box at `ekg_tbox.ttl`, applied to 15 downstream files) |
| 6 of 7 matches still on old T-Box — need pipeline re-run to evaluate multi-track; only Blackburn has new T-Box data | In progress |
| Commentator Chinese drift persists on edge events even with English-only system prompt — needs prompt reinforcement or switch to Llama 3.1 70B | Pending |
| Llama 3.1 70B access pending HuggingFace approval — stuck on Qwen 2.5-7B for now | Pending approval |
| CIDEr corpus size disadvantage — our 7-match (~500 event) corpus is ~50× smaller than JAIST's MatchText (27k pairs); absolute CIDEr cannot match theirs regardless of commentary quality; report multi-ref CIDEr as best-effort honest signal, corpus-size caveat printed in every report | By design |

---

## Fix History

All applied. Full detail in `log_fix/fix_NNN_*.md`.

| Fix | What it solved |
|-----|---------------|
| 001 | Timestamp was clip start, not frame time — now per-frame (±4s accuracy) |
| 002 | VLM over-detection — 0.5 confidence filter + prompt reworded to "CLEARLY visible" |
| 003 | Dedup OR→AND bug silently dropped events; `find_by_color()` ignored its color arg |
| 004 | Schema redesign: added `hasMinute`, `hasPeriod`, `PassEvent`; formally declared `isMatched` |
| 005 | Unmatched events discarded VLM-identified team — now preserved via `INVOLVED_IN` |
| 006 | `hex_to_color_name()` → nearest-neighbor RGB distance; collision detection returns `{}` |
| 007 | ESPN triple-scan at startup → shared `game_id` cuts worst-case 66 → 23 API calls |
| 008 | Consume key 30s bucket → 6s; two fouls 25s apart no longer share a key |
| 009 | SYSTEM_PROMPT rewrite (default empty), per-action strict criteria, 16 frames — precision 14% → 100% on 12-clip test |
| 010 | Recent context injection — last 2-min buffer events passed to VLM to suppress re-detection of same action |
| 011 | Shorts/socks/kit_pattern fields added to VLM output, `MatchedEvent`, KG schema (`hasDetectedShortsColor` etc.) |
| 012 | `evaluate.py` ESPN CSV dependency removed; `--coverage-min` flag added; `Pass` removed from `KEY_ACTIONS` |
| 013 | ESPN confirmation gate — Shot with nearby non-Shot/Goal ESPN event + conf < 0.75 is gated (not ingested into KG) |
| 014 | Free_Kick prompt softened — "both teams around stationary ball" added to criteria; wall no longer required |
| 015 | Shot threshold lowered 0.65 → 0.60; safe because gate (013) blocks low-confidence contradicted Shots |
| 016 | `find_by_color()` missing hit counter — `_tier1_attempts` and `_tier1_hits` now incremented correctly |
| 017 | `VideoEvent` missing `shorts_color`/`socks_color`/`kit_pattern` — kit fields were silently dropped at buffer stage; now wired through |
| 018 | `NUM_FRAMES` 16 → 32 (~1 frame/2s); hardcoded timestamp line removed from prompt — FN shots were missing all sampled frames |
| 019 | `pitch_zone` + `body_part` added to VLM output, full pipeline, and KG schema — enables "right-foot from edge of area" commentary without player identity |
| 020 | `evaluate.py`: ESPN CSV restored for Foul/Corner GT only; default tolerance 2.0 → 0.5 min; gated events logged in `align_buffer()` |
| 021 | Shot threshold 0.60 → 0.65; ESPN gate fires when best match >1.0 min away; cross-batch dedup in `main.py`; eval tolerance 0.5 → 1.0 min |
| 022 | Goal gate in `align.py` — Goal with conf < 0.85 where ESPN says Shot (not Goal) is gated |
| 023 | Mandatory ESPN Shot confirmation — Shot rejected unless ESPN has Shot/Goal within ±1.5 min |
| 024 | Shot threshold 0.65 → 0.80; dedup window 30s → 60s; Shot→Goal guard in `main.py` (Goal within 10s of Shot = double-detection) |
| 025 | Checkpoint registry (`data/kg_output/processed_matches.json`) — KG grows match-by-match; already-processed matches skipped on re-run; missing TTL triggers full reprocess with warning |
| 026 | T-Box rewrite — `ekg_tbox.ttl` with 106 clean classes; `kg_builder.py` asserts single leaf class per instance; foreign-vocab (foaf, schema, prov) types dropped |
| 027 | `hasPeriod` (Match→Period) split from `hasPeriodNumber` (Event→int); `hasJersey` → `hasJerseyNumber` |
| 028 | 15 downstream files renamed: `IS_PERFORMED_BY` → `isPerformedBy` etc. (see commit for full list) |
| 030 | `evaluate_commentary.py`: added METEOR/ROUGE-L/CIDEr metrics; multi-track GT support (ESPN raw, Qwen v1, Qwen v2); `--match` for single-match mode for partial T-Box re-runs |
| 031 | `ai_commentary.json` field `ai_text` renamed to `human_text` for uniform commentary key; evaluator falls back to both for safety |
| 032 | `commentator.py`: ESPN-style 7-action few-shot prompt + 50-70 word length target + `frequency_penalty 0.3` + post-process regenerate if <35 words. Targets +50% BLEU/METEOR/ROUGE/CIDEr without changing the model |
| 033 | `evaluate_commentary.py`: added BLEU-1 and ROUGE-1 columns; proper multi-reference CIDEr computed from all 3 GT tracks pooled per event (`metric_cider_multireference`); BLEU-4 weights fixed (was BLEU-2); hardcoded JAIST MatchAware SN-Long+retrieval reference numbers for honest direct comparison in every single-match and aggregate report |
| 034 | Cleanup: moved `evaluate.py`, `validate_confidence.py`, `check_KG_for_pee_dodo.py`, `ekg_explorer.py` to `deprecated/` — replaced by `src/commentator/evaluate_commentary.py` and `src/5_web_viz/server.py`; `deprecated/README.md` added with restore instructions |
| 035 | Removed RDF reification on `playsFor` — direct triples now used in `kg_builder.py` (both `get_or_create_player` and `prepopulate_roster`) and `serializer.py`. `ekg_schema.py` `player_team_at()` rewritten to direct SPARQL; `plays_for_uri()` marked deprecated. Eliminates `rdf:Statement` class appearing in Protégé and the resulting HermiT inconsistency. Temporal `validFrom` metadata dropped (unused on single-day dataset). Restore from git if multi-season player tracking is needed later |
| 036 | `strip_reification.py` — one-off CPU script to clean existing `ekg.ttl` in-place: removes `rdf:Statement` reification on `playsFor`, replaces with direct triples. Lets users apply fix 035 data-side without re-running the VLM pipeline. Idempotent. Auto-backup to `ekg_pre_fix_035.ttl.bak` before write |
| 037 | `fix_datatypes.py` — one-off migration: removes `hasConfidence` triples, retyped `hasDate` as `xsd:date`, renames `hasKitPattern`→`hasDetectedKitPattern` on Event nodes, removes phantom event nodes left by the `precedes` URI bug |
| 038 | `ekg_schema.py`: added `PROP_TYPES`, `typed_literal()`, `add_typed_triple()` — single source of truth for predicate→xsd:type mapping. Corrupted VLM outputs (e.g. "9a" jersey) are silently dropped and logged to `data/logs/typing_warnings.log` instead of polluting the KG |
| 039 | `kg_builder.py`: all numeric/boolean/date writes now go through `add_typed_triple()`; logging wired to `typing_warnings.log`; `detectedJersey` and `hasJerseyNumber` write `xsd:integer`; `hasPeriodNumber` stays `xsd:integer`; `hasDate` always `xsd:date` |
| 040 | `repair_literal_types.py` — one-off migration: retypes every existing literal in `ekg.ttl` to the correct xsd: type per T-Box; drops triples whose values cannot be coerced (logged). Auto-backup to `ekg_pre_repair_literals.ttl.bak` |
| 041 | T-Box (`ekg_tbox.ttl`): all `xsd:int` ranges changed to `xsd:integer`; `hasJerseyNumber` and `detectedJersey` changed to `xsd:integer`; `hasDetectedKitPattern` added (domain `ekg:Event`); `hasConfidence` removed entirely |
| 042 | T-Box: removed `owl:AllDisjointClasses` on 6 top-level classes (was triggering 130+ false HermiT contradictions after typed literals enabled range checking in fix 038); removed `rdfs:domain ekg:Match` from all DatatypeProperty and `hasPeriod` — kept on `hasHomeTeam`/`hasAwayTeam`/`officiatedBy`/`hasEvent` |
| 043 | `check_consistency.py` — CPU-only HermiT runner via owlready2; prints CONSISTENT or lists classes inferred as `owl:Nothing`; use instead of opening Protégé for quick iteration |
| 044 | `check_consistency.py`: convert TTL → temp RDF/XML before loading (owlready2 cannot parse Turtle); enumerate all inconsistent classes + sample individuals after failure; suppress owlready2 cyclic-subclass noise |
| 045 | `xsd:date` → `xsd:dateTime` across T-Box, `PROP_TYPES`, and `typed_literal()`. HermiT only supports the OWL 2 datatype map (which excludes `xsd:date`). Plain dates like `"2019-10-01"` are coerced to `"2019-10-01T00:00:00Z"`. `repair_literal_types.py` now has an explicit pass to convert any remaining `xsd:date` literals in existing data |
| 046 | Density-biased frame sampling (p=2.5): ~60% of frames in middle 33% of clip, sparse at edges. Replaces end-biased sampling (`t**0.7`). Pairs with event-anchored evaluation where the event is at clip center. `NUM_FRAMES` lowered 32 → 30 |
| 047 | `commentator.py` event-anchored mode: JAIST SN-Long style 5-example SYSTEM_PROMPT (2-3 sentence target, no past-event references); `agent_commentate()` user prompt feeds only current event KG facts (player, team, action, body_part, pitch_zone, outcome, VLM description, match name); past-event tool schemas renamed `TOOLS_V1` and excluded from LLM call; standalone mode SPARQL query extended to fetch `body_part`, `pitchZone`, `outcome`; `SimpleNamespace` and events dict updated accordingly |
| 048 | `event_anchored_eval.py` — GT-anchored clip evaluation (moved to `deprecated/` in fix 051; superseded by sliding-window evaluation) |
| 050 | `convert_jaist_gt.py` rewritten: pure local-filesystem reads from `~/work/s2616011/Augmented_Soccer/Dataset/short/`. No GitHub API. Skips matches with no video folder. Output: `data/sn_long/<season> - <match>/human_commentary.json` |
| 051 | Cleanup: 9 files moved to `deprecated/` — T-Box migration scripts (fix_datatypes, repair_literal_types, strip_reification, migrate_to_new_tbox) already applied; T-DEED dropped; event_anchored_eval superseded by sliding-window eval; EFL-era GT scrapers (3) replaced by JAIST converter |
| 052 | `.gitignore` expanded: ekg.ttl, processed_matches.json, videos, auto-generated GT, evaluation reports, logs, `__pycache__`. Re-runnable artifacts removed from git history |

---

## Running the Pipeline

```bash
python main.py                     # all matches in data/sn_long/
python main.py --test              # 5 clips, 224p, first match only
python main.py --clips 20          # first 20 clips per match
python main.py --match "Burnley"   # specific match (partial name ok)
python main.py --espn-every 3      # ESPN tick every 3 clips
```

### Running JAIST Benchmark Evaluation

```bash
# Step 1: Convert JAIST SN-Short GT (one-time, re-run after new video downloads)
python src/commentator/convert_jaist_gt.py

# Step 2: Reset checkpoint and run pipeline on sample matches
echo '[]' > data/kg_output/processed_matches.json
for m in "Burnley" "Dortmund" "Newcastle"; do
    python main.py --match "$m"
done

# Step 3: Evaluate AI commentary against JAIST GT per match
for m in "Burnley" "Dortmund" "Newcastle"; do
    python src/commentator/evaluate_commentary.py \
        --match "$m" \
        --gt-file human_commentary.json
done
```

---

## Data & Output Files

| Path | Description |
|------|-------------|
| `data/sn_long/<match>/720p.mp4` | Full resolution match video (gitignored) |
| `data/sn_long/<match>/224p.mp4` | Low resolution (used in --test mode, gitignored) |
| `data/sn_long/<match>/human_commentary.json` | JAIST SN-Short GT (written by convert_jaist_gt.py, gitignored) |
| `data/sn_long/<match>/ai_commentary.json` | AI commentary from sliding-window pipeline |
| `data/blackburn_forest_2019-10-01.csv` | ESPN fallback CSV for test match |
| `ekg_tbox.ttl` | T-Box schema (loaded by `ekg_schema.py` at startup) |
| `data/kg_output/ekg.ttl` | RDF/OWL knowledge graph — gitignored, rebuilt each run |
| `data/kg_output/nodes.csv` | KG nodes: players, teams, events, matches |
| `data/kg_output/edges.csv` | KG edges: PERFORMED, PLAYS_FOR, PRECEDED_BY, etc. |
| `data/commentator_output/evaluation_<match>_multitrack.txt` | Per-match multi-GT evaluation |
| `data/commentator_output/evaluation_multitrack_aggregate.txt` | Aggregate across all matches |

---

## Action Types

The VLM detects 7 action types:

| Action | Description |
|--------|-------------|
| `Shot` | Any shot attempt (blocked, saved, wide, on target) |
| `Goal` | Goal scored |
| `Foul` | Foul, handball, trip, contact |
| `Corner` | Corner kick taken |
| `Free_Kick` | Free kick taken |
| `Substitution` | Player replacement |
| `Offside` | Offside called |

Cards (`YellowCard`, `RedCard`) are derived from ESPN full_text and linked to Foul events.

---

## Model

**VLM:** `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Input: 8 evenly-sampled frames from a 60-second clip
- Output: JSON list of detected actions with jersey, team_color, description, confidence

---

## Available Skills

Two project skills live in `.claude/skills/ontology-pipeline/`.

### 1. Ontology Pipeline (`SKILL.md`)
**Trigger:** `/ontology-pipeline`

Hybrid LLM + traditional pipeline for generating OWL ontologies from competency questions.

```
CQs + ODPs → LLM (Ontogenia) → OOPS! loop → SPARQL CQ check → flag for KE review
```

| Step | What it does |
|------|-------------|
| ODP library | Loads vetted Turtle snippets from `odps/` as reusable templates |
| Ontogenia prompting | Feeds each CQ + accumulated ontology + ODPs to Claude; outputs Turtle |
| Merge | `rdflib` merges and deduplicates triples incrementally |
| OOPS! pitfall check | Calls OOPS! REST API; auto-fixes critical pitfalls (P05, P06, P19, P29) with LLM |
| SPARQL CQ check | Generates and runs SPARQL per CQ against T-Box; flags failures for KE review |

Stack: `rdflib`, `anthropic`, `requests`

---

### 2. Production-Grade T-Box & LLM Commentary Readiness (`skill-commentator.md`)
**Trigger:** use directly as a recipe

Three-phase validation to ensure the EKG T-Box is structurally sound and narrative-ready for an LLM football commentator.

```
ekg.ttl → SPARQL context → text serialization → LLM prompt → commentary
          ↑ validated in three escalating phases: Structural → Narrative → Metacognitive
```

| Phase | Tool | What it checks |
|-------|------|---------------|
| **1a** Structural integrity | `pitfall_scanner.py` | OOPS! critical pitfalls (P05, P06, P19, P29) that break LLM reasoning |
| **1b** Conciseness | `conciseness_check.py` | Superfluous element rate — T-Box elements not referenced by any CCQ. **Target < 15%**; above 15% the LLM follows irrelevant edges and invents facts |
| **2** Narrative CCQs | `commentator_cqs.py` | CCQ01–CCQ10 SPARQL existence checks against T-Box |
| **3** Metacognitive | `metacognitive_validator.py` | Claude in "Ontologist Persona" evaluates each CQ in isolation — generates a minimal A-Box example, rates READY / PARTIAL / BLOCKED |
| Serialization debug | `serializer.py` | Every ActionEvent → `event_to_context()` → flags thin-context events |
| LLM commentary | `commentator.py` | Calls `claude-haiku-4-5` per event; factual check (minute mismatch) |

**Four scoring dimensions:**

| Dimension | Formula | What it penalises |
|-----------|---------|------------------|
| Accuracy | metacognitive READY rate × 10 | CQs the LLM can't instantiate as A-Box examples |
| Completeness | CQ SPARQL pass rate × 10 | Missing properties/classes |
| Conciseness | 10 − max(0, superfluous% − 15) / 5 | Superfluous elements above 15% threshold |
| Consistency | 10 − critical_pitfalls × 1.5 | Critical OOPS! pitfalls |
| Commentary | 10 − thin_events% / 10 | A-Box events with no player / no text / no PRECEDED_BY |

Score interpretation:

| All four ≥ 8 | Commentary ≥ 8 | Meaning |
|---|---|---|
| No | — | Fix whichever dimension is lowest first |
| Yes | < 7 | T-Box solid — enrich A-Box (add `hasDescription`, fix player links) |
| Yes | ≥ 8 | Ready for live LLM commentary |

Run:
```bash
# Full 3-phase with A-Box
python debug_commentator.py --ttl ../../ekg.ttl --sample 5

# Phase 1+2 only, no LLM calls (offline)
python debug_commentator.py --no-llm --no-meta

# Fully offline (skip OOPS!, LLM commentary, and metacognitive)
python debug_commentator.py --ttl ../../ekg.ttl --no-oops --no-llm --no-meta
```

Files needed in `src/4_kg_builder/`:

| File | Status | Role |
|------|--------|------|
| `commentator_cqs.py` | ✓ exists | Phase 2 CCQs |
| `serializer.py` | ✓ exists | event serialization |
| `debug_commentator.py` | ✓ exists (old version) | runner — needs updating to 3-phase |
| `pitfall_scanner.py` | needs creating | Phase 1a |
| `conciseness_check.py` | needs creating | Phase 1b |
| `metacognitive_validator.py` | needs creating | Phase 3 |
| `commentator.py` | needs creating | LLM generation |

Stack: `rdflib`, `anthropic`, `requests`, `thefuzz`, `owlready2`

---

## Prediction Sub-Goal (Future)

The EKG is structured to support SPARQL queries for pattern-based prediction:
- Event sequences: `PRECEDED_BY` chain allows temporal pattern matching
- Player-level history: `IS_PERFORMED_BY` links events to players across matches
- Team-level aggregation: `INVOLVED_IN` links teams to event types

Example prediction targets:
- Probability of a goal given 3 consecutive shots in 5 minutes
- Likelihood of a substitution in the 60–75 minute window
- Card probability for a player with 2+ fouls
