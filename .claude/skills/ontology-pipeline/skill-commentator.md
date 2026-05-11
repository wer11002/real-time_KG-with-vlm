# Skill: Production-Grade T-Box & LLM Commentary Readiness Framework

Purpose: A high-rigor, three-phase validation process to ensure an EKG T-Box is
structurally sound and narrative-ready for an LLM football commentator.

```
ekg.ttl → SPARQL context → text serialization → LLM prompt → commentary
          ↑ this skill validates every step of the chain
          ↑ with three escalating phases: Structural → Narrative → Metacognitive
```

---

## Stack

```
pip install rdflib anthropic requests thefuzz owlready2
```

---

## Why T-Box requirements differ for LLM vs SPARQL

| Need | SPARQL use | LLM commentator use |
|---|---|---|
| OWL class hierarchy | Reasoning over subtypes | Less critical — LLM reads string labels |
| Inverse properties | Graph traversal | Less critical — LLM doesn't traverse |
| **Text properties** | Low value | **Critical — hasDescription, hasFullText, rdfs:label** |
| **Temporal chain** | ORDER BY | **Critical — narrative needs sequence** |
| **Causal links** | Optional | **Critical — "foul led to card" is commentary gold** |
| **hasMinute** | ORDER BY filter | **Critical — "in the 67th minute"** |
| **hasPeriod** | Half filter | Important — "in the second half" |
| **detectedJersey** | Low | Medium — backup when player name missing |
| Reification standard | Reasoning | Not critical — LLM ignores reification |

---

## Evaluation Dimensions

Commentary readiness is assessed across four dimensions, derived from ontology
evaluation research:

| Dimension | What it measures |
|---|---|
| **Accuracy** | T-Box correctly models domain concepts without logical contradictions |
| **Completeness** | All CQs required for commentary can be answered |
| **Conciseness** | No superfluous classes/properties that distract the LLM |
| **Logical Consistency** | No pitfalls (e.g. multiple domains/ranges) that break LLM reasoning |

---

## Phase 1 — Structural & Logical Integrity (Automated)

Traditional SPARQL CQs are necessary but insufficient for LLM-driven systems.
This phase targets pitfalls that specifically degrade LLM reasoning quality.

### 1.1 Critical Pitfall Scanning (via OOPS!)

Automated check for the "Critical" errors most likely to break LLM commentary:

| Pitfall | Code | LLM impact |
|---|---|---|
| Multiple domains/ranges | P19 | LLM interprets union as intersection — wrong class membership |
| Inverse property errors | P05 | Narrative traversal fails — "foul committed by" chain breaks |
| Missing disjointness | P06 | LLM conflates event types — confuses GoalEvent with ShotEvent |
| No domain/range defined | P29 | LLM can't infer property targets — blind property use |

```python
# pitfall_scanner.py

import requests
import xml.etree.ElementTree as ET

OOPS_URL          = "http://localhost:8080/OOPS/rest"
CRITICAL_PITFALLS = {"P05", "P06", "P19", "P29"}


def run_oops(turtle_str: str) -> dict:
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
  <OntologyContent><![CDATA[{turtle_str}]]></OntologyContent>
  <Pitfalls></Pitfalls>
  <OutputFormat>RDF/XML</OutputFormat>
</OOPSRequest>"""
    try:
        r = requests.post(
            OOPS_URL,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=60,
        )
        r.raise_for_status()
        root     = ET.fromstring(r.text)
        pitfalls = {}
        for p in root.iter("{http://oops.linkeddata.es/def#}pitfall"):
            code  = p.findtext("{http://oops.linkeddata.es/def#}hasCode") or "?"
            elems = [
                e.text for e in p.findall(
                    ".//{http://oops.linkeddata.es/def#}hasAffectedElement"
                ) if e.text
            ]
            pitfalls[code] = elems
        return pitfalls
    except Exception as e:
        return {"_error": str(e)}


def report_pitfalls(oops_results: dict):
    critical = [c for c in oops_results if c in CRITICAL_PITFALLS]
    print(f"\n── OOPS! Pitfall Report ──")
    print(f"  Critical pitfalls found: {len(critical)}")
    for code in critical:
        elems = oops_results[code]
        print(f"  {code}  ({len(elems)} affected elements)")
        for e in elems[:5]:
            print(f"       → {e}")
    if not critical:
        print("  ✓ No critical pitfalls detected")
```

### 1.2 Conciseness Metric: Superfluous Element Rate

Research shows LLMs generate redundant classes (e.g. `employedSince` and
`employmentStartDate`) that distract the model during generation. A "superfluous"
element is any named class or property not referenced by any Commentary CQ.

**Target: Superfluous Element Rate < 15%**

Above 15%, "model distraction" degrades commentary — the LLM follows irrelevant
edges and invents unsupported facts.

```python
# conciseness_check.py

from rdflib import Graph, OWL, RDF
from rdflib.namespace import Namespace

EKG = Namespace("http://soccerekg.org/ontology#")

# All URIs exercised by the 10 Commentary CQs
CQ_USED_ELEMENTS = {
    EKG.PRECEDED_BY, EKG.hasMinute, EKG.PERFORMED, EKG.PLAYS_FOR,
    EKG.TRIGGERED, EKG.hasPeriod, EKG.hasDescription, EKG.hasFullText,
    EKG.INVOLVED_IN, EKG.ShotEvent, EKG.ASSISTED_BY, EKG.YellowCardEvent,
    EKG.Player, EKG.ActionEvent, EKG.detectedJersey,
}


def superfluous_element_rate(g: Graph) -> dict:
    """
    Calculate ratio of T-Box classes/properties not exercised by any Commentary CQ.
    """
    all_classes = set(g.subjects(RDF.type, OWL.Class))
    all_props   = (
        set(g.subjects(RDF.type, OWL.ObjectProperty)) |
        set(g.subjects(RDF.type, OWL.DatatypeProperty))
    )
    all_elements  = all_classes | all_props
    superfluous   = all_elements - CQ_USED_ELEMENTS
    rate          = len(superfluous) / len(all_elements) * 100 if all_elements else 0

    result = {
        "total"         : len(all_elements),
        "used"          : len(all_elements) - len(superfluous),
        "superfluous"   : len(superfluous),
        "rate_pct"      : round(rate, 1),
        "exceeds_limit" : rate > 15.0,
        "elements"      : [str(e).split("#")[-1] for e in superfluous],
    }

    print(f"\n── Conciseness Check ──")
    print(f"  Total T-Box elements : {result['total']}")
    print(f"  Used by CQs          : {result['used']}")
    print(f"  Superfluous          : {result['superfluous']}  ({result['rate_pct']}%)")
    if result["exceeds_limit"]:
        print(f"  ✗ Exceeds 15% target — consider pruning:")
        for e in result["elements"][:10]:
            print(f"    → {e}")
    else:
        print(f"  ✓ Within 15% target")

    return result
```

---

## Phase 2 — Narrative "Commentary Gold" Readiness

### 2.1 Commentary Competency Questions (CCQ01–CCQ10)

These differ from schema_eval.py CQs — they test what an LLM commentator
specifically needs. Each is evaluated against the T-Box via SPARQL existence checks.

```python
# commentator_cqs.py

from rdflib import Graph
from pathlib import Path

COMMENTARY_CQS = [
    # ── Narrative sequence ─────────────────────────────────────────────────
    {
        "id"    : "CCQ01",
        "text"  : "What is the sequence of events leading up to event E?",
        "why"   : "Commentator needs build-up narrative — 'after a long period of pressure...'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?prop WHERE {
                ekg:PRECEDED_BY a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:hasMinute   a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:PRECEDED_BY AS ?prop)
            }
        """,
    },
    # ── Player identity ────────────────────────────────────────────────────
    {
        "id"    : "CCQ02",
        "text"  : "Who performed event E and which team do they play for?",
        "why"   : "Core commentary fact — 'Adam Armstrong of Blackburn scores!'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:PERFORMED a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:PLAYS_FOR a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:PERFORMED AS ?p)
            }
        """,
    },
    # ── Causal chain ───────────────────────────────────────────────────────
    {
        "id"    : "CCQ03",
        "text"  : "What foul triggered the yellow card shown to player P?",
        "why"   : "Causal commentary — 'after a reckless tackle, the referee reached for his pocket'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:TRIGGERED a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:TRIGGERED AS ?p)
            }
        """,
    },
    # ── Time context ───────────────────────────────────────────────────────
    {
        "id"    : "CCQ04",
        "text"  : "What minute and period did event E happen in?",
        "why"   : "Essential for commentary — 'in the 67th minute of the second half'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?m ?p WHERE {
                ?m a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?m <http://www.w3.org/2000/01/rdf-schema#label> "hasMinute" .
                ?p a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?p <http://www.w3.org/2000/01/rdf-schema#label> "hasPeriod" .
            }
        """,
    },
    # ── Natural language description ───────────────────────────────────────
    {
        "id"    : "CCQ05",
        "text"  : "What did the VLM observe about event E?",
        "why"   : "Rich visual context for LLM — 'player in blue kit lunges into tackle'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:hasDescription a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:hasDescription AS ?p)
            }
        """,
    },
    # ── ESPN commentary text ───────────────────────────────────────────────
    {
        "id"    : "CCQ06",
        "text"  : "What is the official commentary text for event E?",
        "why"   : "Ground truth text from ESPN — LLM can rephrase/expand this",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:hasFullText a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:hasFullText AS ?p)
            }
        """,
    },
    # ── Team statistics ────────────────────────────────────────────────────
    {
        "id"    : "CCQ07",
        "text"  : "How many shots has team T taken in the first half?",
        "why"   : "Stats commentary — 'Blackburn dominating with 7 shots in the first half'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?a ?b WHERE {
                ekg:INVOLVED_IN  a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:ShotEvent    a <http://www.w3.org/2002/07/owl#Class> .
                ekg:hasPeriod    a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:INVOLVED_IN AS ?a)
                BIND(ekg:ShotEvent   AS ?b)
            }
        """,
    },
    # ── Assist ─────────────────────────────────────────────────────────────
    {
        "id"    : "CCQ08",
        "text"  : "Who assisted goal G?",
        "why"   : "Goal commentary — 'set up beautifully by Lenihan'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:ASSISTED_BY a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:ASSISTED_BY AS ?p)
            }
        """,
    },
    # ── Player card history ────────────────────────────────────────────────
    {
        "id"    : "CCQ09",
        "text"  : "Has player P already received a card in this match?",
        "why"   : "Danger commentary — 'one more foul and he walks'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?a ?b WHERE {
                ekg:YellowCardEvent a <http://www.w3.org/2002/07/owl#Class> .
                ekg:PERFORMED       a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:YellowCardEvent AS ?a)
                BIND(ekg:PERFORMED       AS ?b)
            }
        """,
    },
    # ── Player label ───────────────────────────────────────────────────────
    {
        "id"    : "CCQ10",
        "text"  : "What is the human-readable name of player P?",
        "why"   : "LLM needs rdfs:label to name players — not just URI fragments",
        "sparql": """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ekg:  <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:Player a <http://www.w3.org/2002/07/owl#Class> .
                ?label a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                FILTER(?label = rdfs:label)
                BIND(rdfs:label AS ?p)
            }
        """,
    },
]


def run_commentary_cqs(g: Graph) -> list:
    results = []
    for cq in COMMENTARY_CQS:
        try:
            rows   = list(g.query(cq["sparql"]))
            passed = len(rows) > 0
        except Exception as e:
            passed = False
            cq["_error"] = str(e)
        results.append({**cq, "passed": passed})
    return results


def print_commentary_cq_report(results: list):
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    print(f"\n{'═'*76}")
    print(f"  COMMENTARY CQ CHECK  {passed}/{total} ({int(passed/total*100)}%)")
    print(f"{'─'*76}")
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {r['id']}  {icon}  {r['text'][:50]:<50}")
        if not r["passed"]:
            print(f"          WHY NEEDED: {r['why']}")
    print(f"{'═'*76}\n")
```

### 2.2 T-Box Axiom Checks for Commentary Patterns

Beyond SPARQL existence, verify that specific linguistic and temporal structures
are properly axiomatized:

| Commentary requirement | T-Box metric / axiom check | Why |
|---|---|---|
| Textual labels | Presence of `rdfs:label` & `hasDescription` | LLMs rely on string labels, not URIs |
| Temporal chains | `hasMinute` & `hasPeriod` as DatatypeProperties | "In the 67th minute of the second half..." |
| Causal linking | `TRIGGERED` connecting FoulEvent → CardEvent | Build-up narrative vs isolated event listing |
| Inverse properties | `IS_PERFORMED_BY` alongside `PERFORMED` | Bidirectional graph traversal for player lookup |
| Discourse patterns | DoCO/DEO-compatible structures (optional) | Reusing standard patterns improves interoperability |

---

## Phase 3 — Metacognitive Validation (The "Ontogenia" Test)

Instead of just running a SPARQL existence check, use an LLM in an "Ontologist
Persona" to self-evaluate the T-Box through Metacognitive Prompting. This catches
gaps that automated queries miss — especially holistic narrative quality.

### 3.1 Memoryless CQ-by-CQ Testing

**Key principle:** evaluate each CQ in isolation to prevent context distraction.
Providing the full T-Box for every CQ risks the model relying on adjacent context
rather than testing the specific requirement.

```python
# metacognitive_validator.py

import anthropic
from rdflib import Graph

client = anthropic.Anthropic()

ONTOLOGIST_SYSTEM = """You are a senior ontology engineer evaluating an OWL T-Box
for use by an LLM football commentator. You will be given:
1. A single Competency Question (CQ)
2. The relevant T-Box module (classes and properties that could answer it)

For each CQ you must:
a) Generate a minimal A-Box example (2-3 individuals) that would satisfy this CQ
b) State whether the T-Box schema is sufficient to instantiate your example
c) Identify any missing axioms that would be needed
d) Rate readiness: READY / PARTIAL / BLOCKED

Be precise. Do not assume axioms not explicitly shown."""


def evaluate_cq_metacognitive(cq: dict, tbox_module: str) -> dict:
    """
    Evaluate a single CQ in isolation — no other CQs in context.
    Returns structured assessment dict.
    """
    user_msg = f"""Competency Question: {cq['id']} — {cq['text']}

Commentary need: {cq['why']}

Relevant T-Box module:
{tbox_module}

Generate an A-Box example, assess readiness, and list any missing axioms."""

    response = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 512,
        system     = ONTOLOGIST_SYSTEM,
        messages   = [{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text.strip()

    # Extract readiness verdict from response
    status = "UNKNOWN"
    for verdict in ("READY", "PARTIAL", "BLOCKED"):
        if verdict in text.upper():
            status = verdict
            break

    return {
        "cq_id"     : cq["id"],
        "status"    : status,
        "assessment": text,
    }


def run_metacognitive_validation(cqs: list, tbox_modules: dict) -> list:
    """
    Run CQ-by-CQ metacognitive validation.
    tbox_modules: dict mapping cq_id → relevant Turtle snippet.
    """
    results = []
    print(f"\n── Phase 3: Metacognitive Validation ──")
    for cq in cqs:
        module = tbox_modules.get(cq["id"], "# (no specific module provided)")
        result = evaluate_cq_metacognitive(cq, module)
        icon   = {"READY": "✓", "PARTIAL": "~", "BLOCKED": "✗"}.get(result["status"], "?")
        print(f"  {cq['id']}  {icon}  {result['status']:<8}  {cq['text'][:45]}")
        results.append(result)
    return results
```

### 3.2 Expert Qualitative Baseline

Automated metrics can miss holistic narrative quality. After Phase 3, perform a
final manual check on generated commentary samples using this rubric:

| Criterion | Pass condition |
|---|---|
| **Accuracy** | No facts invented beyond the KG context |
| **Fluency** | Reads naturally as live broadcast speech |
| **Completeness** | Names player, minute, and event type (when available) |
| **Causality** | Mentions foul → card chain when TRIGGERED edge exists |
| **Build-up** | References at least one preceding event when PRECEDED_BY exists |

---

## Serialization Test — Event Neighborhood → Text

This is the most important LLM-readiness check. If you can't serialize a clean,
informative text block for each event, the LLM commentator won't work regardless
of how good the OWL structure is.

```python
# serializer.py

from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD
from rdflib.namespace import Namespace

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")


def event_to_context(event_uri: URIRef, g: Graph, n_preceding: int = 3) -> dict:
    """
    Extract structured context for one event from the A-Box.
    Returns a dict — good for inspection and for building LLM prompts.
    """
    def lit(uri, prop):
        vals = list(g.objects(uri, prop))
        return str(vals[0]) if vals else None

    ctx = {
        "uri"        : str(event_uri),
        "event_type" : lit(event_uri, EKG.hasEventType),
        "time"       : lit(event_uri, EKG.hasTime),
        "minute"     : lit(event_uri, EKG.hasMinute),
        "period"     : lit(event_uri, EKG.hasPeriod),
        "description": lit(event_uri, EKG.hasDescription),
        "full_text"  : lit(event_uri, EKG.hasFullText),
        "jersey"     : lit(event_uri, EKG.detectedJersey),
        "is_matched" : lit(event_uri, EKG.isMatched),
        "player"     : None,
        "player_team": None,
        "assist"     : None,
        "card"       : None,
        "preceded_by": [],
    }

    # player
    players = list(g.subjects(EKG.PERFORMED, event_uri))
    if players:
        p_uri         = players[0]
        ctx["player"] = lit(p_uri, RDFS.label) or str(p_uri).split("#")[-1]
        for edge in g.subjects(RDF.subject, p_uri):
            if (edge, RDF.type, RDF.Statement) in g:
                team_uri = list(g.objects(edge, RDF.object))
                if team_uri:
                    ctx["player_team"] = lit(team_uri[0], RDFS.label)

    # assist
    assists = list(g.objects(event_uri, EKG.ASSISTED_BY))
    if assists:
        ctx["assist"] = lit(assists[0], RDFS.label) or str(assists[0]).split("#")[-1]

    # triggered card
    cards = list(g.objects(event_uri, EKG.TRIGGERED))
    if cards:
        ctx["card"] = lit(cards[0], EKG.hasEventType)

    # preceding events (walk PRECEDED_BY chain)
    current = event_uri
    for _ in range(n_preceding):
        prev_list = list(g.objects(current, EKG.PRECEDED_BY))
        if not prev_list:
            break
        prev         = prev_list[0]
        prev_players = list(g.subjects(EKG.PERFORMED, prev))
        prev_player  = None
        if prev_players:
            prev_player = lit(prev_players[0], RDFS.label)
        ctx["preceded_by"].append({
            "type"  : lit(prev, EKG.hasEventType),
            "time"  : lit(prev, EKG.hasTime),
            "minute": lit(prev, EKG.hasMinute),
            "player": prev_player,
        })
        current = prev

    return ctx


def context_to_text(ctx: dict) -> str:
    """Convert extracted context dict → natural language block for LLM prompt."""
    lines = []

    period_str = {"1": "first half", "2": "second half"}.get(str(ctx.get("period", "")), "")
    minute_str = f"{float(ctx['minute']):.1f}'" if ctx.get("minute") else ctx.get("time", "?")

    lines.append(
        f"EVENT: {ctx['event_type']} at {minute_str}" +
        (f" ({period_str})" if period_str else "")
    )

    if ctx.get("player"):
        team_str = f" ({ctx['player_team']})" if ctx.get("player_team") else ""
        lines.append(f"PLAYER: {ctx['player']}{team_str}")

    if ctx.get("assist"):
        lines.append(f"ASSIST: {ctx['assist']}")

    if ctx.get("card"):
        lines.append(f"CARD: {ctx['card']} issued following this event")

    # ESPN text is ground truth — placed before VLM to anchor LLM generation
    if ctx.get("full_text"):
        lines.append(f"ESPN: {ctx['full_text']}")

    if ctx.get("description"):
        lines.append(f"VLM: {ctx['description']}")

    if ctx.get("preceded_by"):
        lines.append("RECENT EVENTS (before this):")
        for prev in reversed(ctx["preceded_by"]):
            t = f"{float(prev['minute']):.1f}'" if prev.get("minute") else prev.get("time", "?")
            p = f" by {prev['player']}" if prev.get("player") else ""
            lines.append(f"  {t}  {prev['type']}{p}")

    return "\n".join(lines)


def serialization_debug(g: Graph) -> list:
    """
    Run serialization on all ActionEvent nodes. Flag events with thin context.
    Returns list of (event_uri, time, issues).
    """
    issues = []
    events = list(g.subjects(RDF.type, EKG.ActionEvent))

    print(f"\n── Serialization debug: {len(events)} ActionEvent nodes ──")
    for ev in events:
        ctx       = event_to_context(ev, g)
        ev_issues = []

        if not ctx["player"]:
            ev_issues.append("NO PLAYER — commentator can't name who did it")
        if not ctx["minute"]:
            ev_issues.append("NO hasMinute — can't say 'in the 67th minute'")
        if not ctx["description"] and not ctx["full_text"]:
            ev_issues.append("NO TEXT — neither hasDescription nor hasFullText")
        if not ctx["preceded_by"]:
            ev_issues.append("NO PRECEDED_BY — no build-up narrative possible")

        if ev_issues:
            t = ctx.get("time", "?")
            issues.append((str(ev), t, ev_issues))

    if not issues:
        print("  ✓ All events have sufficient context for commentary")
    else:
        print(f"  {len(issues)} events with thin context:")
        for uri, t, evissues in issues[:10]:
            name = uri.split("#")[-1]
            print(f"  ✗ {t:<8} {name}")
            for i in evissues:
                print(f"          → {i}")

    return issues
```

---

## LLM Commentary Generation

```python
# commentator.py

import anthropic
from rdflib import Graph, URIRef, RDF
from rdflib.namespace import Namespace
from serializer import event_to_context, context_to_text

EKG    = Namespace("http://soccerekg.org/ontology#")
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a live football commentator with the style of a professional
broadcaster. You are given structured facts from a Knowledge Graph about a specific event.

Rules:
- Be excited but precise — ONLY state facts from the context given
- If player name is missing, say "the player" or use the jersey number if available
- ESPN text is ground truth — you may rephrase it but never contradict it
- Keep commentary to 2–3 sentences
- Do NOT invent player names, scores, or any facts not present in the context"""


def generate_commentary(event_uri: URIRef, g: Graph, match_context: str = "") -> str:
    ctx     = event_to_context(event_uri, g, n_preceding=3)
    kg_text = context_to_text(ctx)

    user_msg = f"""Generate live commentary for this football event.

{f'Match context: {match_context}' if match_context else ''}

Knowledge Graph context:
{kg_text}

Speak as if this is happening right now."""

    response = client.messages.create(
        model      = "claude-haiku-4-5-20251001",  # fast + cheap for real-time commentary
        max_tokens = 256,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


def commentary_factual_check(commentary: str, ctx: dict) -> list:
    """
    Rule-based check: does the commentary contradict the KG facts?
    Returns list of issues found.
    """
    import re
    issues = []

    if ctx.get("minute"):
        minute          = round(float(ctx["minute"]))
        numbers_in_text = re.findall(r"\b(\d{1,3})\b", commentary)
        for n in numbers_in_text:
            n = int(n)
            if 1 <= n <= 120 and abs(n - minute) > 3:
                issues.append(f"Minute mismatch: KG={minute}', commentary mentions {n}'")

    return issues
```

---

## Full Debug Runner

```python
# debug_commentator.py
"""
Full T-Box + commentary readiness check — all three phases.

Usage:
    python debug_commentator.py --ttl path/to/ekg.ttl
    python debug_commentator.py --ttl ekg.ttl --sample 3
    python debug_commentator.py --ttl ekg.ttl --no-oops --no-llm --no-meta
"""

import sys
import argparse
from pathlib import Path
from rdflib import Graph, RDF
from rdflib.namespace import Namespace

sys.path.insert(0, str(Path(__file__).parent / "src/4_kg_builder"))
from ekg_schema        import EKG_Graph, EKG
from commentator_cqs   import run_commentary_cqs, print_commentary_cq_report
from serializer        import serialization_debug, event_to_context, context_to_text
from pitfall_scanner   import run_oops, report_pitfalls, CRITICAL_PITFALLS
from conciseness_check import superfluous_element_rate

TBOX_MODULES = {}   # populate with {cq_id: turtle_snippet} for metacognitive phase


def score_for_commentator(
    cq_results, oops_results, serial_issues, superfluous, meta_results, n_events
) -> dict:
    """
    Four-dimensional score card: accuracy, completeness, conciseness, consistency.
    """
    cq_pass  = sum(1 for r in cq_results if r["passed"])
    cq_total = len(cq_results)
    critical = sum(1 for c in oops_results if c in CRITICAL_PITFALLS)
    thin_pct = len(serial_issues) / n_events * 100 if n_events > 0 else 0

    meta_ready = sum(1 for r in meta_results if r["status"] == "READY") if meta_results else cq_pass
    meta_total = len(meta_results) if meta_results else cq_total

    # Accuracy: metacognitive READY rate (falls back to SPARQL CQ pass rate)
    accuracy     = round(meta_ready / meta_total * 10, 1) if meta_total else 0
    # Completeness: CQ SPARQL coverage
    completeness = round(cq_pass / cq_total * 10, 1)
    # Conciseness: penalise if superfluous rate > 15%
    conc_penalty = max(0, superfluous.get("rate_pct", 0) - 15) / 5
    conciseness  = round(max(0, 10 - conc_penalty), 1)
    # Logical consistency: penalise per critical pitfall
    consistency  = round(max(0, 10 - critical * 1.5), 1)
    # Commentary score: thin event coverage
    commentary   = round(10 - thin_pct / 10, 1)

    return {
        "accuracy"          : accuracy,
        "completeness"      : completeness,
        "conciseness"       : conciseness,
        "consistency"       : consistency,
        "commentary_score"  : commentary,
        "cq_coverage"       : f"{cq_pass}/{cq_total}",
        "critical_pitfalls" : critical,
        "superfluous_pct"   : superfluous.get("rate_pct", "N/A"),
        "thin_events_pct"   : round(thin_pct, 1),
    }


def main(args):
    print("\n─── T-Box Debug + LLM Commentator Readiness (3-Phase) ───\n")

    # ── 1. Load T-Box ──────────────────────────────────────────────────────
    ekg_obj = EKG_Graph()
    tbox_g  = ekg_obj.g
    turtle  = tbox_g.serialize(format="turtle")
    print(f"T-Box: {len(tbox_g)} triples")

    # ── Phase 1a: OOPS! pitfall scan ──────────────────────────────────────
    oops_results = {}
    if not args.no_oops:
        print("\nPhase 1a — Structural integrity (OOPS!)...")
        oops_results = run_oops(turtle)
        report_pitfalls(oops_results)
    else:
        print("Phase 1a skipped (--no-oops)")

    # ── Phase 1b: Conciseness check ───────────────────────────────────────
    print("\nPhase 1b — Conciseness metric...")
    superfluous = superfluous_element_rate(tbox_g)

    # ── Phase 2: Commentary CQ check ──────────────────────────────────────
    print("\nPhase 2 — Narrative readiness (CCQ01-CCQ10)...")
    cq_results = run_commentary_cqs(tbox_g)
    print_commentary_cq_report(cq_results)

    # ── Phase 3: Metacognitive validation ─────────────────────────────────
    meta_results = []
    if not args.no_meta:
        from metacognitive_validator import run_metacognitive_validation
        print("\nPhase 3 — Metacognitive validation (Ontogenia test)...")
        meta_results = run_metacognitive_validation(cq_results, TBOX_MODULES)
    else:
        print("Phase 3 skipped (--no-meta)")

    # ── Serialization debug (A-Box) ────────────────────────────────────────
    serial_issues = []
    n_events      = 0
    if args.ttl and Path(args.ttl).exists():
        abox_g        = Graph()
        abox_g.parse(args.ttl, format="turtle")
        n_events      = len(list(abox_g.subjects(RDF.type, EKG.ActionEvent)))
        serial_issues = serialization_debug(abox_g)

        if not args.no_llm and n_events > 0:
            from commentator import generate_commentary, commentary_factual_check
            events = list(abox_g.subjects(RDF.type, EKG.ActionEvent))[:args.sample]
            print(f"\n── Sample commentary ({len(events)} events) ──")
            for ev in events:
                ctx  = event_to_context(ev, abox_g)
                text = context_to_text(ctx)
                print(f"\n  Context:\n{text}\n")
                try:
                    commentary = generate_commentary(ev, abox_g)
                    print(f"  Commentary:\n  \"{commentary}\"")
                    issues = commentary_factual_check(commentary, ctx)
                    for i in issues:
                        print(f"  ⚠ {i}")
                    if not issues:
                        print(f"  ✓ no factual issues detected")
                except Exception as e:
                    print(f"  LLM error: {e}")
    else:
        print("\nNo A-Box loaded — serialization and commentary tests skipped")

    # ── Score card ─────────────────────────────────────────────────────────
    scores = score_for_commentator(
        cq_results, oops_results, serial_issues, superfluous, meta_results, n_events
    )

    print(f"\n{'═'*64}")
    print(f"  COMMENTATOR READINESS SCORE CARD")
    print(f"{'─'*64}")
    print(f"  Accuracy     (Phase 3 metacognitive) : {scores['accuracy']}/10")
    print(f"  Completeness (Phase 2 CQ coverage)   : {scores['completeness']}/10  [{scores['cq_coverage']}]")
    print(f"  Conciseness  (Phase 1b superfluous)  : {scores['conciseness']}/10  [{scores['superfluous_pct']}% superfluous]")
    print(f"  Consistency  (Phase 1a OOPS! critical): {scores['consistency']}/10  [{scores['critical_pitfalls']} critical pitfalls]")
    print(f"{'─'*64}")
    print(f"  Commentary score (A-Box thin events)  : {scores['commentary_score']}/10  [{scores['thin_events_pct']}% thin]")
    print(f"{'─'*64}")

    bottleneck = min(scores, key=lambda k: scores[k] if isinstance(scores[k], (int, float)) else 10)
    print(f"  Bottleneck dimension: {bottleneck.upper()}")

    if scores["conciseness"] < 7:
        print(f"  → Prune superfluous T-Box elements below 15% to reduce LLM distraction")
    if scores["consistency"] < 7:
        print(f"  → Fix multiple domain/range (P19) and inverse property (P05) errors first")
    if scores["completeness"] < 7:
        print(f"  → Add missing properties: check failed CQs above")
    if scores["commentary_score"] < 7:
        print(f"  → Add hasDescription / hasFullText to more A-Box events")
        print(f"  → Ensure IS_PERFORMED_BY is asserted (not just PERFORMED)")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",     default=None)
    parser.add_argument("--sample",  type=int, default=3)
    parser.add_argument("--no-oops", action="store_true")
    parser.add_argument("--no-llm",  action="store_true")
    parser.add_argument("--no-meta", action="store_true")
    args = parser.parse_args()
    main(args)
```

---

## Score Interpretation

| Accuracy | Completeness | Conciseness | Consistency | Meaning |
|---|---|---|---|---|
| < 7 | any | any | any | Metacognitive test blocking — T-Box can't instantiate key CQs |
| any | < 7 | any | any | Too many CQs failing — add missing properties |
| any | any | < 7 | any | Too many superfluous elements — prune before tuning LLM prompts |
| any | any | any | < 7 | Critical OOPS! pitfalls — fix multiple domains/ranges first |
| ≥ 8 | ≥ 8 | ≥ 8 | ≥ 8 | T-Box ready — focus on A-Box richness and commentary score |

---

## Debug Checklist

If commentary comes out generic or wrong, check in this order:

```
1. Commentary says "the player" instead of a name?
   → IS_PERFORMED_BY not asserted in kg_builder (only PERFORMED)
   → OR rdfs:label missing on Player node

2. Commentary gets the minute wrong?
   → hasMinute not written to A-Box
   → Check _create_event_node() writes ekg:hasMinute

3. Commentary invents events?
   → System prompt not strict enough — tighten "ONLY use facts from context"
   → Superfluous T-Box elements distracting the LLM — run conciseness check

4. Commentary ignores the foul-to-card story?
   → TRIGGERED edge missing — check card_type detection in kg_builder
   → OR P05 pitfall: inverse of TRIGGERED not axiomatized

5. Commentary is flat / no build-up?
   → PRECEDED_BY chain not wired — check last_event dict in kg_builder
   → OR n_preceding=0 in event_to_context()

6. Commentary contradicts ESPN text?
   → hasFullText is ground truth — confirm it's placed before VLM in context_to_text()

7. Metacognitive test rates CQ as BLOCKED?
   → LLM cannot generate a valid A-Box example → T-Box axiom is missing or ambiguous
   → Check OOPS! P19 (multiple domains) on the relevant property
```

---

## File Layout

```
project/
├── .claude/skills/ontology-pipeline/
│   └── skill-commentator.md     # this file
├── src/4_kg_builder/
│   ├── ekg_schema.py
│   ├── kg_builder.py
│   ├── schema_eval.py
│   ├── commentator_cqs.py       # Phase 2: CCQ01-CCQ10
│   ├── debug_commentator.py     # full 3-phase runner
│   ├── serializer.py            # event_to_context() + context_to_text()
│   ├── pitfall_scanner.py       # Phase 1a: NEW — needs creating
│   ├── conciseness_check.py     # Phase 1b: NEW — needs creating
│   ├── metacognitive_validator.py # Phase 3: NEW — needs creating
│   └── commentator.py           # LLM generation — needs creating (if missing)
├── ekg.ttl
├── ekg_problems.md
├── main.py
└── evaluate.py
```

```bash
# Phase 1 + 2 only (no A-Box, no LLM calls)
python debug_commentator.py --no-llm --no-meta

# Full 3-phase check with A-Box
python debug_commentator.py --ttl ../../ekg.ttl --sample 5

# Offline mode (skip OOPS!, LLM commentary, and metacognitive)
python debug_commentator.py --ttl ../../ekg.ttl --no-oops --no-llm --no-meta
```