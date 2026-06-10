"""
schema_eval.py — EKG Schema Scorer (SKILL.md methodology)
──────────────────────────────────────────────────────────

Implements the CQ + OOPS! scoring pipeline from .claude/skills/ontology-pipeline/SKILL.md:
    CQs → SPARQL T-Box check → OOPS! pitfall scan → score card

13 Competency Questions covering commentary and prediction use cases.
SPARQL queries run against the T-Box only (no A-Box needed).
OOPS! REST API called for pitfall detection.

Run:
    python schema_eval.py            # score current schema
    python schema_eval.py --no-oops  # skip OOPS! (offline / no network)
"""

import sys
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from io import StringIO

from rdflib import Graph

sys.path.insert(0, str(Path(__file__).parent))
from ekg_schema import EKG_Graph, EKG

OOPS_URL = "https://oops.linkeddata.es/rest"

# OOPS! codes that are considered critical (from SKILL.md)
CRITICAL_PITFALLS = {"P05", "P06", "P19", "P29"}
# minor but worth reporting
MINOR_PITFALLS    = {"P04", "P08", "P11", "P13", "P22", "P24", "P25", "P36"}


# ═══════════════════════════════════════════════════════════════════════════
# COMPETENCY QUESTIONS + SPARQL CHECKS
# ═══════════════════════════════════════════════════════════════════════════

# Each CQ has:
#   text   : natural-language question
#   sparql : SPARQL SELECT run against T-Box only
#            "pass" = query returns ≥1 result (needed class/property exists)

CQS = [
    {
        "id"    : "CQ01",
        "text"  : "What events happened in match M, sorted by chronological order?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?prop WHERE {
                ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?prop <http://www.w3.org/2000/01/rdf-schema#label> "hasMinute" .
                ?prop <http://www.w3.org/2000/01/rdf-schema#range> xsd:decimal .
            }
        """,
        "note"  : "Requires hasMinute (xsd:decimal) for numeric ORDER BY",
    },
    {
        "id"    : "CQ02",
        "text"  : "Who performed event E?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:isPerformedBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:isPerformedBy AS ?p)
            }
        """,
        "note"  : "Requires isPerformedBy object property",
    },
    {
        "id"    : "CQ03",
        "text"  : "Which team was involved in event E?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:involvedTeam a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:involvedTeam AS ?p)
            }
        """,
        "note"  : "Requires involvedTeam (Team → Event)",
    },
    {
        "id"    : "CQ04",
        "text"  : "What is the precise game time (as a sortable number) of event E?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?prop WHERE {
                ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?prop <http://www.w3.org/2000/01/rdf-schema#label> "hasMinute" .
                ?prop <http://www.w3.org/2000/01/rdf-schema#range> xsd:decimal .
            }
        """,
        "note"  : "Requires hasMinute (xsd:decimal) for numeric comparison",
    },
    {
        "id"    : "CQ05",
        "text"  : "What event directly preceded event E in the match timeline?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:precededBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:precededBy AS ?p)
            }
        """,
        "note"  : "Requires precededBy (Event → Event)",
    },
    {
        "id"    : "CQ06",
        "text"  : "Who assisted goal G?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:assistedBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:assistedBy AS ?p)
            }
        """,
        "note"  : "Requires assistedBy (Event → Player)",
    },
    {
        "id"    : "CQ07",
        "text"  : "How many fouls has player P committed in match M?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?prop WHERE {
                ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?prop <http://www.w3.org/2000/01/rdf-schema#label> "hasMinute" .
            }
        """,
        "note"  : "Requires hasMinute to filter events to a match time window",
    },
    {
        "id"    : "CQ08",
        "text"  : "Did player P receive a card in match M, and what type?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?p WHERE {
                ekg:triggered a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                BIND(ekg:triggered AS ?p)
            }
        """,
        "note"  : "Requires triggered (PlayerAction → Card)",
    },
    {
        "id"    : "CQ09",
        "text"  : "What sequence of event types immediately preceded the last goal in match M?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?prec ?type WHERE {
                ekg:precededBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:hasEventType a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                BIND(ekg:precededBy AS ?prec)
                BIND(ekg:hasEventType AS ?type)
            }
        """,
        "note"  : "Requires precededBy chain + hasEventType string label",
    },
    {
        "id"    : "CQ10",
        "text"  : "How many shots has team T taken in the first half of match M?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?prop WHERE {
                ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?prop <http://www.w3.org/2000/01/rdf-schema#label> "hasPeriodNumber" .
                ?prop <http://www.w3.org/2000/01/rdf-schema#range> xsd:integer .
            }
        """,
        "note"  : "Requires hasPeriodNumber (xsd:integer) for first/second half filter",
    },
    {
        "id"    : "CQ11",
        "text"  : "Which team was player P playing for in match M (temporal)?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?prop WHERE {
                ?prop a <http://www.w3.org/2002/07/owl#DatatypeProperty> .
                ?prop <http://www.w3.org/2000/01/rdf-schema#label> "validFrom" .
                ?prop <http://www.w3.org/2000/01/rdf-schema#range> xsd:date .
            }
        """,
        "note"  : "Requires validFrom (xsd:date) on playsFor TKG edge",
    },
    {
        "id"    : "CQ12",
        "text"  : "How many consecutive Shot events occurred before goal G?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?prec ?type WHERE {
                ekg:precededBy a <http://www.w3.org/2002/07/owl#ObjectProperty> .
                ekg:Shot   a <http://www.w3.org/2002/07/owl#Class> .
                BIND(ekg:precededBy AS ?prec)
                BIND(ekg:Shot   AS ?type)
            }
        """,
        "note"  : "Requires precededBy + Shot OWL class",
    },
    {
        "id"    : "CQ13",
        "text"  : "Were there any Pass events in the 5 minutes before a goal?",
        "sparql": """
            PREFIX ekg: <http://soccerekg.org/ontology#>
            SELECT ?c WHERE {
                ekg:PassEvent a <http://www.w3.org/2002/07/owl#Class> .
                BIND(ekg:PassEvent AS ?c)
            }
        """,
        "note"  : "Requires PassEvent OWL class",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# CQ CHECK
# ═══════════════════════════════════════════════════════════════════════════

def run_cq_checks(g: Graph) -> list:
    results = []
    for cq in CQS:
        try:
            rows = list(g.query(cq["sparql"]))
            passed = len(rows) > 0
        except Exception as e:
            passed = False
            cq["_error"] = str(e)
        results.append({**cq, "passed": passed})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# OOPS! CHECK
# ═══════════════════════════════════════════════════════════════════════════

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
        return parse_oops_response(r.text)
    except requests.exceptions.RequestException as e:
        return {"_error": str(e)}


def parse_oops_response(xml_text: str) -> dict:
    pitfalls = {}
    try:
        root = ET.fromstring(xml_text)
        for p in root.iter("{http://oops.linkeddata.es/def#}pitfall"):
            code  = (p.findtext("{http://oops.linkeddata.es/def#}hasCode") or
                     p.findtext("Code") or "?")
            name  = (p.findtext("{http://oops.linkeddata.es/def#}hasName") or
                     p.findtext("Name") or "")
            elems = [e.text for e in p.findall(
                ".//{http://oops.linkeddata.es/def#}hasAffectedElement") if e.text]
            if not elems:
                elems = [e.text for e in p.findall(".//AffectedElement") if e.text]
            pitfalls[code] = {"name": name, "elements": elems}
    except ET.ParseError:
        pitfalls["_parse_error"] = {"name": "Could not parse OOPS! XML response", "elements": []}
    return pitfalls


# ═══════════════════════════════════════════════════════════════════════════
# SCORE CARD PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_score_card(cq_results: list, oops_results: dict, label: str = ""):
    w = 76
    passed  = sum(1 for r in cq_results if r["passed"])
    total   = len(cq_results)
    pct     = int(passed / total * 100)

    print(f"\n{'═'*w}")
    if label:
        print(f"  {label}")
    print(f"{'═'*w}")
    print(f"  CQ Coverage: {passed}/{total} ({pct}%)")
    print(f"{'─'*w}")
    print(f"  {'ID':<6} {'Pass':<6} {'Note'}")
    print(f"  {'─'*6} {'─'*6} {'─'*60}")
    for r in cq_results:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {r['id']:<6} {icon:<6} {r['note']}")

    print(f"\n{'─'*w}")
    if "_error" in oops_results:
        print(f"  OOPS! check: FAILED — {oops_results['_error']}")
    else:
        critical_found = [c for c in oops_results if c in CRITICAL_PITFALLS]
        minor_found    = [c for c in oops_results if c in MINOR_PITFALLS]
        other_found    = [c for c in oops_results
                          if c not in CRITICAL_PITFALLS and c not in MINOR_PITFALLS
                          and not c.startswith("_")]

        print(f"  OOPS! pitfalls found:")
        if not oops_results:
            print(f"    (none)")
        else:
            for code, info in oops_results.items():
                if code.startswith("_"):
                    continue
                severity = "CRITICAL" if code in CRITICAL_PITFALLS else (
                           "minor"    if code in MINOR_PITFALLS    else "info")
                name     = info.get("name", "")[:50]
                count    = len(info.get("elements", []))
                print(f"    [{severity:8}] {code}  {name}  ({count} elements)")

        print(f"\n  Critical pitfalls : {len(critical_found)}  "
              f"(P05 P06 P19 P29 — wrong inverses, cycles, multi-domain)")
        print(f"  Minor pitfalls    : {len(minor_found)}")
        print(f"  Other             : {len(other_found)}")

    print(f"{'═'*w}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(run_oops_check: bool = True):
    print("\n─── EKG Schema Eval (SKILL.md CQ + OOPS! pipeline) ───")

    # load T-Box
    ekg_graph = EKG_Graph()
    g         = ekg_graph.g
    print(f"  T-Box loaded: {len(g)} triples")

    # serialize T-Box to Turtle for OOPS!
    turtle_str = g.serialize(format="turtle")
    print(f"  Turtle size : {len(turtle_str)} chars\n")

    # CQ check
    print("  Running CQ SPARQL checks against T-Box...")
    cq_results = run_cq_checks(g)

    # OOPS! check
    oops_results = {}
    if run_oops_check:
        print("  Calling OOPS! REST API...")
        oops_results = run_oops(turtle_str)
        if "_error" in oops_results:
            print(f"  OOPS! failed: {oops_results['_error']}")
        else:
            print(f"  OOPS! returned {len(oops_results)} pitfall code(s)")
    else:
        print("  OOPS! check skipped (--no-oops)")
        oops_results = {"_error": "skipped by user (--no-oops flag)"}

    print_score_card(cq_results, oops_results, label="EKG Schema Score Card")

    passed = sum(1 for r in cq_results if r["passed"])
    return passed, len(cq_results), oops_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-oops", action="store_true",
                        help="Skip the OOPS! REST API call (offline mode)")
    args = parser.parse_args()
    main(run_oops_check=not args.no_oops)
