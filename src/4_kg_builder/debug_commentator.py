"""
debug_commentator.py — T-Box Debug + LLM Commentator Readiness Check

Run:
    python debug_commentator.py --ttl path/to/ekg.ttl
    python debug_commentator.py --ttl ekg.ttl --sample 3
    python debug_commentator.py --ttl ekg.ttl --no-oops --no-llm
"""

import sys
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from rdflib import Graph, RDF
from rdflib.namespace import Namespace

sys.path.insert(0, str(Path(__file__).parent))
from ekg_schema import EKG_Graph, EKG
from commentator_cqs import run_commentary_cqs, print_commentary_cq_report
from serializer     import serialization_debug, event_to_context, context_to_text

OOPS_URL          = "https://oops.linkeddata.es/rest"
CRITICAL_PITFALLS = {"P05", "P06", "P19", "P29"}


def run_oops(turtle_str: str) -> dict:
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
  <OntologyContent><![CDATA[{turtle_str}]]></OntologyContent>
  <Pitfalls></Pitfalls>
  <OutputFormat>RDF/XML</OutputFormat>
</OOPSRequest>"""
    try:
        r = requests.post(OOPS_URL, data=payload.encode("utf-8"),
                          headers={"Content-Type": "application/xml"}, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        pitfalls = {}
        for p in root.iter("{http://oops.linkeddata.es/def#}pitfall"):
            code = (p.findtext("{http://oops.linkeddata.es/def#}hasCode") or "?")
            elems = [e.text for e in p.findall(
                ".//{http://oops.linkeddata.es/def#}hasAffectedElement") if e.text]
            pitfalls[code] = elems
        return pitfalls
    except Exception as e:
        return {"_error": str(e)}


def score_for_commentator(cq_results, oops_results, serial_issues, n_events) -> dict:
    cq_pass  = sum(1 for r in cq_results if r["passed"])
    cq_total = len(cq_results)
    critical = sum(1 for c in oops_results if c in CRITICAL_PITFALLS)
    thin_pct = len(serial_issues) / n_events * 100 if n_events > 0 else 0

    structural = round((cq_pass / cq_total) * 10 - critical * 0.5, 1)
    commentary = round(10 - thin_pct / 10, 1)
    structural = max(0, min(10, structural))
    commentary = max(0, min(10, commentary))

    return {
        "structural_score"  : structural,
        "commentary_score"  : commentary,
        "cq_coverage"       : f"{cq_pass}/{cq_total}",
        "critical_pitfalls" : critical,
        "thin_events_pct"   : round(thin_pct, 1),
    }


def main(args):
    print("\n─── T-Box Debug + LLM Commentator Readiness ───\n")

    # ── 1. Load T-Box ──────────────────────────────────────────────────────
    ekg_obj = EKG_Graph()
    tbox_g  = ekg_obj.g
    turtle  = tbox_g.serialize(format="turtle")
    print(f"T-Box: {len(tbox_g)} triples")

    # ── 2. Commentary CQ check ─────────────────────────────────────────────
    print("Running Commentary CQs against T-Box...")
    cq_results = run_commentary_cqs(tbox_g)
    print_commentary_cq_report(cq_results)

    # ── 3. OOPS! ──────────────────────────────────────────────────────────
    oops_results = {}
    if not args.no_oops:
        print("Calling OOPS! REST API...")
        oops_results = run_oops(turtle)
        critical = [c for c in oops_results if c in CRITICAL_PITFALLS]
        print(f"  Critical pitfalls: {len(critical)}  {critical or '✓ none'}")
    else:
        print("OOPS! skipped (--no-oops)")

    # ── 4. Serialization debug ─────────────────────────────────────────────
    serial_issues = []
    n_events      = 0
    if args.ttl and Path(args.ttl).exists():
        abox_g = Graph()
        abox_g.parse(args.ttl, format="turtle")
        n_events      = len(list(abox_g.subjects(RDF.type, EKG.PlayerAction)))
        serial_issues = serialization_debug(abox_g)

        # ── 5. Sample commentary ───────────────────────────────────────────
        if not args.no_llm and n_events > 0:
            from commentator import generate_commentary, commentary_factual_check
            events = list(abox_g.subjects(RDF.type, EKG.PlayerAction))[:args.sample]
            print(f"\n── Sample commentary ({len(events)} events) ──")
            for ev in events:
                ctx  = event_to_context(ev, abox_g)
                text = context_to_text(ctx)
                print(f"\n  Context:\n{text}\n")
                try:
                    commentary = generate_commentary(ev, abox_g)
                    print(f"  Commentary:\n  \"{commentary}\"")
                    issues = commentary_factual_check(commentary, ctx)
                    if issues:
                        for i in issues:
                            print(f"  ⚠ {i}")
                    else:
                        print(f"  ✓ no factual issues detected")
                except Exception as e:
                    print(f"  LLM error: {e}")
    else:
        print("\nNo A-Box loaded (--ttl not provided or file not found)")
        print("Structural check only — serialization and commentary tests skipped")

    # ── 6. Score card ──────────────────────────────────────────────────────
    scores = score_for_commentator(cq_results, oops_results, serial_issues, n_events)

    print(f"\n{'═'*60}")
    print(f"  COMMENTATOR READINESS SCORE CARD")
    print(f"{'─'*60}")
    print(f"  Structural score  : {scores['structural_score']}/10")
    print(f"    CQ coverage     : {scores['cq_coverage']}")
    print(f"    Critical OOPS!  : {scores['critical_pitfalls']}")
    print(f"  Commentary score  : {scores['commentary_score']}/10")
    print(f"    Thin events     : {scores['thin_events_pct']}%")
    print(f"      (events with no player / no text / no precededBy)")
    print(f"{'─'*60}")
    print(f"  Bottleneck: structural={scores['structural_score']} "
          f"commentary={scores['commentary_score']}")
    if n_events == 0:
        print(f"  → No A-Box loaded — commentary score reflects T-Box only")
    elif scores['commentary_score'] < scores['structural_score']:
        print(f"  → Commentary readiness is the limiting factor.")
        print(f"    Add hasDescription / hasFullText to more events.")
        print(f"    Ensure isPerformedBy is asserted (not just performed).")
    else:
        print(f"  → T-Box structure is the limiting factor.")
        print(f"    Fix CQ coverage or reduce critical OOPS! pitfalls.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",     default=None,
                        help="Path to ekg.ttl for A-Box serialization test")
    parser.add_argument("--sample",  type=int, default=3,
                        help="Number of events to generate commentary for")
    parser.add_argument("--no-oops", action="store_true")
    parser.add_argument("--no-llm",  action="store_true")
    args = parser.parse_args()
    main(args)
