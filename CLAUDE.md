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
              RDF graph     data/ekg.ttl
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
| `evaluate.py` still uses 2-minute match tolerance despite timestamp accuracy improving to ±4s — may inflate precision | In plan |
| No commentator module yet | In plan |
| Ontology redesign for prediction use case | User researching |

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

---

## Running the Pipeline

```bash
python main.py                    # all matches in data/
python main.py --test             # 5 clips, 224p, first match only
python main.py --clips 20         # first 20 clips per match
python main.py --match "Blackburn" # specific match (partial name ok)
python main.py --espn-every 3     # ESPN tick every 3 clips
```

---

## Data & Output Files

| Path | Description |
|------|-------------|
| `data/<match>/720p.mp4` | Full resolution match video |
| `data/<match>/224p.mp4` | Low resolution (used in --test mode) |
| `data/<match>/Labels-ball.json` | Ball tracking ground truth (frame-level annotations) |
| `data/blackburn_forest_2019-10-01.csv` | ESPN fallback CSV for test match |
| `data/ekg.ttl` | RDF/OWL knowledge graph (Turtle format) |
| `data/kg_output/nodes.csv` | KG nodes: players, teams, events, matches |
| `data/kg_output/edges.csv` | KG edges: PERFORMED, PLAYS_FOR, PRECEDED_BY, etc. |
| `data/commentator_output/` | Commentator logs (planned) |
| `data/logs/` | Pipeline run logs |

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
