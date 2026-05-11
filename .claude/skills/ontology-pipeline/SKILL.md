---
name: Ontology Pipeline
description: Hybrid LLM + ODP pipeline for generating OWL ontologies from competency questions using OOPS! validation and SPARQL CQ checks.
---
# Skill: Hybrid LLM + Traditional Ontology Generation

A coding recipe for the combined pipeline:
```
CQs + ODPs → LLM (Ontogenia) → OOPS! loop → SPARQL CQ check → flag for KE review
```

---

## Stack

```
pip install rdflib anthropic requests
```

- `rdflib` — parse, build, and query OWL/Turtle ontologies
- `anthropic` (or `openai`) — LLM API calls
- `requests` — call the OOPS! REST API
- No local reasoner needed for basic CQ checks; rdflib handles SPARQL

---

## 1. Data structures

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CompetencyQuestion:
    id: str               # "CQ01"
    text: str             # "Who is the author of a book?"
    sparql: Optional[str] = None   # filled by generate_sparql()
    modelled: bool = False         # filled by cq_check()
    minor_issue: bool = False

@dataclass
class OntologyDesignPattern:
    name: str             # "AgentRole"
    description: str      # one-line purpose
    turtle: str           # Turtle snippet — the vetted template
```

---

## 2. ODP library (old-gen input)

Keep a folder `odps/` of vetted Turtle snippets. Load them at startup:

```python
from pathlib import Path
import rdflib

def load_odps(odp_dir: str = "odps/") -> list[OntologyDesignPattern]:
    patterns = []
    for path in Path(odp_dir).glob("*.ttl"):
        turtle = path.read_text()
        name = path.stem
        # first comment line in the file = description
        desc = next((l.lstrip("# ") for l in turtle.splitlines() if l.startswith("#")), "")
        patterns.append(OntologyDesignPattern(name=name, description=desc, turtle=turtle))
    return patterns

def odps_as_prompt_block(odps: list[OntologyDesignPattern]) -> str:
    lines = ["## Available Ontology Design Patterns\n"]
    for odp in odps:
        lines.append(f"### {odp.name}\n{odp.description}\n```turtle\n{odp.turtle}\n```\n")
    return "\n".join(lines)
```

---

## 3. Ontogenia prompting (new-gen core)

Each CQ is addressed in sequence; the accumulated ontology is injected into context.

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are an expert ontology engineer working in OWL 2 / Turtle.
You will be given:
- A competency question (CQ) that the ontology must answer
- The ontology built so far (may be empty)
- A library of vetted Ontology Design Patterns (ODPs) to reuse

Rules:
1. Output ONLY valid Turtle. No explanation outside the code block.
2. Reuse ODPs wherever they fit — prefer them over inventing new structure.
3. Every object property must have exactly ONE rdfs:domain and ONE rdfs:range.
4. Declare owl:inverseOf for every inverse pair.
5. Do not add classes or properties not needed to answer the CQ.
"""

def generate_for_cq(
    cq: CompetencyQuestion,
    accumulated_ttl: str,
    odps: list[OntologyDesignPattern],
) -> str:
    odp_block = odps_as_prompt_block(odps)
    user_msg = f"""## Competency Question
{cq.text}

## Ontology so far
```turtle
{accumulated_ttl or "(empty)"}
```

{odp_block}

Add the minimum OWL axioms needed to answer the CQ. Output Turtle only."""

    response = client.messages.create(
        model="claude-opus-4-7",  # swap for o1 / gpt-4 if preferred
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return extract_turtle(response.content[0].text)


def extract_turtle(text: str) -> str:
    """Pull content from first ```turtle ... ``` block."""
    import re
    m = re.search(r"```(?:turtle|ttl)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()
```

---

## 4. Merge Turtle incrementally

```python
def merge_turtle(base: str, addition: str, base_prefix: str = "http://example.org/ontology#") -> str:
    g = rdflib.Graph()
    if base:
        g.parse(data=base, format="turtle")
    g.parse(data=addition, format="turtle")
    return g.serialize(format="turtle")
```

rdflib deduplicates triples automatically on merge.

---

## 5. OOPS! pitfall check (old-gen tool, automated)

OOPS! has a REST API. Send the Turtle, get back pitfalls as XML.

```python
import requests
import xml.etree.ElementTree as ET

OOPS_URL = "https://oops.linkeddata.es/rest"

CRITICAL_PITFALLS = {"P05", "P06", "P19", "P29"}  # wrong inverses, cycles, multi-domain, wrong transitive

def run_oops(turtle: str) -> dict[str, list[str]]:
    """Returns {pitfall_code: [affected_elements]}."""
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
  <OntologyContent><![CDATA[{turtle}]]></OntologyContent>
  <Pitfalls></Pitfalls>
  <OutputFormat>RDF/XML</OutputFormat>
</OOPSRequest>"""
    r = requests.post(OOPS_URL, data=payload.encode("utf-8"),
                      headers={"Content-Type": "application/xml"}, timeout=60)
    r.raise_for_status()
    return parse_oops_response(r.text)

def parse_oops_response(xml_text: str) -> dict[str, list[str]]:
    root = ET.fromstring(xml_text)
    pitfalls = {}
    for p in root.iter("Pitfall"):
        code = p.findtext("Code", "")
        elements = [e.text for e in p.findall(".//AffectedElement") if e.text]
        pitfalls[code] = elements
    return pitfalls

def fix_pitfalls_with_llm(turtle: str, pitfalls: dict[str, list[str]]) -> str:
    critical = {k: v for k, v in pitfalls.items() if k in CRITICAL_PITFALLS}
    if not critical:
        return turtle

    pitfall_text = "\n".join(
        f"- {code}: affects {', '.join(elems[:5])}" for code, elems in critical.items()
    )
    user_msg = f"""The ontology has these OWL pitfalls detected by OOPS!:
{pitfall_text}

Fix them in the Turtle below. Output corrected Turtle only.

```turtle
{turtle}
```"""
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system="You are an expert OWL ontology engineer. Fix only the stated pitfalls. Do not add new elements.",
        messages=[{"role": "user", "content": user_msg}],
    )
    return extract_turtle(response.content[0].text)
```

---

## 6. SPARQL CQ verification (automated ground-truth check)

Generate a SPARQL query for each CQ, run it against the ontology, flag failures.

```python
def generate_sparql_for_cq(cq: CompetencyQuestion, turtle: str) -> str:
    user_msg = f"""Given this ontology:
```turtle
{turtle}
```

Write a SPARQL SELECT query that answers this competency question:
"{cq.text}"

Output the SPARQL query only, inside a ```sparql ... ``` block."""
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        system="You are a SPARQL expert. Write minimal valid SPARQL for the given ontology and CQ.",
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text
    m = __import__("re").search(r"```(?:sparql)?\s*(.*?)```", raw, __import__("re").DOTALL)
    return m.group(1).strip() if m else raw.strip()

def cq_check(cq: CompetencyQuestion, turtle: str) -> bool:
    """Returns True if the ontology can answer the CQ (query returns results on T-Box alone)."""
    if not cq.sparql:
        cq.sparql = generate_sparql_for_cq(cq, turtle)
    g = rdflib.Graph()
    g.parse(data=turtle, format="turtle")
    try:
        results = list(g.query(cq.sparql))
        # T-Box only: we check the query is syntactically valid and the variables resolve
        # A result count > 0 means the needed classes/properties exist
        cq.modelled = len(results) > 0
    except Exception:
        cq.modelled = False
    return cq.modelled
```

> Note: running SPARQL against T-Box alone (no A-Box instances) only catches structural errors — missing classes or properties. It won't catch wrong semantics, but it is a cheap first filter.

---

## 7. Main pipeline

```python
def run_pipeline(
    cqs: list[CompetencyQuestion],
    odps: list[OntologyDesignPattern],
    oops_max_retries: int = 2,
) -> tuple[str, list[CompetencyQuestion]]:
    """
    Returns (final_turtle, flagged_cqs_for_ke_review).
    """
    ontology_ttl = ""
    flagged: list[CompetencyQuestion] = []

    for cq in cqs:
        print(f"\n── {cq.id}: {cq.text[:60]}…")

        # Step 1: Ontogenia — generate sub-ontology for this CQ
        addition = generate_for_cq(cq, ontology_ttl, odps)
        ontology_ttl = merge_turtle(ontology_ttl, addition)

        # Step 2: OOPS! loop — fix critical pitfalls automatically
        for attempt in range(oops_max_retries):
            pitfalls = run_oops(ontology_ttl)
            critical = {k: v for k, v in pitfalls.items() if k in CRITICAL_PITFALLS}
            if not critical:
                break
            print(f"  OOPS! attempt {attempt+1}: {list(critical.keys())}")
            ontology_ttl = fix_pitfalls_with_llm(ontology_ttl, pitfalls)

        # Step 3: SPARQL CQ check — flag failures for KE
        if not cq_check(cq, ontology_ttl):
            print(f"  ✗ CQ not modelled — flagged for KE review")
            flagged.append(cq)
        else:
            print(f"  ✓ CQ modelled")

    return ontology_ttl, flagged


def main():
    odps = load_odps("odps/")

    cqs = [
        CompetencyQuestion("CQ01", "Who is the author of a book?"),
        CompetencyQuestion("CQ02", "Which books were published after 2010?"),
        CompetencyQuestion("CQ03", "What role did a person have in an organization at a given date?"),
    ]

    turtle, flagged = run_pipeline(cqs, odps)

    # Save ontology
    Path("output/ontology.ttl").write_text(turtle)
    print(f"\n── Done. {len(flagged)} CQ(s) flagged for KE review:")
    for cq in flagged:
        print(f"  {cq.id}: {cq.text}")
```

---

## 8. KE review interface (minimal)

For flagged CQs, print a diff of what's missing:

```python
def ke_review_prompt(cq: CompetencyQuestion, turtle: str) -> str:
    return f"""
=== KE REVIEW REQUIRED ===
CQ {cq.id}: {cq.text}

SPARQL that failed:
{cq.sparql}

Current ontology excerpt (last 50 lines):
{chr(10).join(turtle.splitlines()[-50:])}

Fix the ontology so the SPARQL above returns results.
"""
```

In a real system this becomes a web UI or a simple `input()` loop where the KE pastes corrected Turtle.

---

## File layout

```
project/
├── skill.md             # this file
├── odps/
│   ├── agent_role.ttl   # vetted ODP: Agent → Role → Organization
│   ├── time_interval.ttl
│   └── ...
├── output/
│   └── ontology.ttl     # generated result
├── pipeline.py          # code from sections 1–7
└── requirements.txt
```

---

## Key design decisions

| Decision | Why |
|---|---|
| Ontogenia (accumulated context) over Memoryless | Better for reification and cross-CQ consistency |
| ODP injection in every prompt | Prevents LLM from inventing wrong structure |
| OOPS! loop capped at 2 retries | Diminishing returns; some pitfalls require KE judgment |
| SPARQL check on T-Box only | Cheap, zero A-Box data needed; catches structural gaps |
| Flag, don't block | Pipeline keeps going; KE reviews async, not inline |

