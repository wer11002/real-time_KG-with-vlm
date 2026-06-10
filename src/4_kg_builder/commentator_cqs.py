# commentator_cqs.py

from rdflib import Graph
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ekg_schema import EKG_Graph

COMMENTARY_CQS = [
    # ── Narrative sequence ─────────────────────────────────────────────────
    {
        "id"    : "CCQ01",
        "text"  : "What is the sequence of events leading up to event E?",
        "why"   : "Commentator needs build-up narrative — 'after a long period of pressure...'",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?prop WHERE {
                ekg:precededBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:hasMinute   a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:precededBy AS ?prop)
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
                ekg:performed a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:playsFor a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:performed AS ?p)
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
                ekg:triggered a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:triggered AS ?p)
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
                ?p <http://www.w3.org/2000/01/rdf-schema#label> "hasPeriodNumber" .
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
                ekg:involvedTeam  a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:Shot    a <http://www.w3.org/2002/07/owl#Class> .
                ekg:hasPeriodNumber    a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:involvedTeam AS ?a)
                BIND(ekg:Shot   AS ?b)
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
                ekg:assistedBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:assistedBy AS ?p)
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
                ekg:YellowCard a <http://www.w3.org/2002/07/owl#Class> .
                ekg:performed       a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:YellowCard AS ?a)
                BIND(ekg:performed       AS ?b)
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


def run_commentary_cqs(g) -> list:
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
