"""
fix_datatypes.py — one-off migration: fix datatype violations in ekg.ttl

Fixes five problems that caused HermiT inconsistency after fix 035:

  1. Remove all ekg:hasConfidence triples (confidence is a raw VLM number with
     no formula — not meaningful to store)
  2. Rewrite ekg:hasDate literals as xsd:date  (was bare string xsd:string)
  3. Rewrite ekg:hasKitPattern on Event nodes → ekg:hasDetectedKitPattern
     (hasKitPattern domain is ekg:Team; writing on Events caused domain violation)
  4. Remove phantom event nodes created by the precedes URI bug (bare data:event_NNNN
     nodes with no rdf:type, created because match_id was missing from event_uri call)

Note: hasPeriodNumber (xsd:integer written as xsd:int) and hasJerseyNumber
(xsd:string written without datatype) are fixed in the T-Box, not the data —
  - hasPeriodNumber: T-Box range changed xsd:int → xsd:integer (code already writes xsd:integer)
  - hasJerseyNumber: T-Box range changed xsd:int → xsd:string (code already writes xsd:string)
So no data-side fix needed for those two.

Usage:
    # default: fix ekg.ttl in-place, auto-backup first
    python src/4_kg_builder/fix_datatypes.py

    # custom paths / skip backup
    python src/4_kg_builder/fix_datatypes.py --in data/kg_output/ekg.ttl --out clean.ttl
    python src/4_kg_builder/fix_datatypes.py --no-backup
"""

import argparse
import shutil
import sys
from pathlib import Path

from rdflib import Graph, Namespace, RDF, Literal, URIRef, XSD

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTL = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
DEFAULT_BAK = BASE_DIR / "data" / "kg_output" / "ekg_pre_fix_datatypes.ttl.bak"


# ── helpers ────────────────────────────────────────────────────────────────

def _count_triples(g: Graph, subj=None, pred=None, obj=None) -> int:
    return sum(1 for _ in g.triples((subj, pred, obj)))


def _is_event_node(g: Graph, uri: URIRef) -> bool:
    """True if the node has at least one rdf:type that is NOT rdf:Statement."""
    for t in g.objects(uri, RDF.type):
        if t != RDF.Statement:
            return True
    return False


# ── migration steps ────────────────────────────────────────────────────────

def remove_confidence(g: Graph) -> int:
    triples = list(g.triples((None, EKG.hasConfidence, None)))
    for s, p, o in triples:
        g.remove((s, p, o))
    return len(triples)


def fix_has_date(g: Graph) -> int:
    triples = list(g.triples((None, EKG.hasDate, None)))
    fixed = 0
    for s, p, o in triples:
        if not isinstance(o, Literal):
            continue
        if o.datatype == XSD.date:
            continue  # already correct
        date_str = str(o).strip()
        g.remove((s, p, o))
        g.add((s, EKG.hasDate, Literal(date_str, datatype=XSD.date)))
        fixed += 1
    return fixed


def rename_kit_pattern_on_events(g: Graph) -> int:
    """
    For each (event, hasKitPattern, value) where event is NOT a Team,
    replace with (event, hasDetectedKitPattern, value).
    """
    triples = list(g.triples((None, EKG.hasKitPattern, None)))
    renamed = 0
    for s, p, o in triples:
        if (s, RDF.type, EKG.Team) in g:
            continue  # belongs on a Team — leave it
        g.remove((s, p, o))
        g.add((s, EKG.hasDetectedKitPattern, o))
        renamed += 1
    return renamed


def remove_phantom_event_nodes(g: Graph) -> int:
    """
    Remove bare data:event_NNNN nodes (no rdf:type) that were created by the
    precedes URI bug (match_id missing from event_uri call).  These nodes appear
    only as objects of ekg:precededBy or subjects of ekg:precedes.
    """
    candidates: set[URIRef] = set()
    for s, p, o in g.triples((None, EKG.precededBy, None)):
        if isinstance(o, URIRef) and not _is_event_node(g, o):
            candidates.add(o)
    for s, p, o in g.triples((None, EKG.precedes, None)):
        if isinstance(o, URIRef) and not _is_event_node(g, o):
            candidates.add(o)

    removed_triples = 0
    for phantom in candidates:
        edges = list(g.predicate_objects(phantom)) + [(pred, phantom) for _, pred in g.subject_predicates(phantom)]
        for pred, obj in list(g.predicate_objects(phantom)):
            g.remove((phantom, pred, obj))
            removed_triples += 1
        for subj, pred in list(g.subject_predicates(phantom)):
            g.remove((subj, pred, phantom))
            removed_triples += 1

    return len(candidates)


# ── main ───────────────────────────────────────────────────────────────────

def fix(in_path: Path, out_path: Path, backup: bool) -> None:
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}")
        sys.exit(1)

    print(f"Loading {in_path} ...")
    g = Graph()
    g.parse(str(in_path), format="turtle")
    n_before = len(g)
    print(f"  Total triples before : {n_before:,}")

    if backup:
        print(f"Backing up → {DEFAULT_BAK} ...")
        shutil.copy2(str(in_path), str(DEFAULT_BAK))
        print("  Backup written.")

    print("\nApplying fixes ...")

    n1 = remove_confidence(g)
    print(f"  [1] hasConfidence removed    : {n1} triple(s)")

    n2 = fix_has_date(g)
    print(f"  [2] hasDate retyped xsd:date : {n2} literal(s)")

    n3 = rename_kit_pattern_on_events(g)
    print(f"  [3] hasKitPattern→hasDetectedKitPattern on events: {n3}")

    n4 = remove_phantom_event_nodes(g)
    print(f"  [4] Phantom event nodes removed : {n4} node(s)")

    n_after = len(g)
    print(f"\n  Total triples after  : {n_after:,}")
    print(f"  Net change           : {n_after - n_before:+,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving → {out_path} ...")
    g.serialize(destination=str(out_path), format="turtle")
    print("Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Fix datatype violations in ekg.ttl (post fix-035 cleanup).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--in",        dest="in_path",  type=Path, default=DEFAULT_TTL)
    ap.add_argument("--out",       dest="out_path", type=Path, default=None)
    ap.add_argument("--no-backup", dest="backup",   action="store_false", default=True)
    args = ap.parse_args()

    out_path = args.out_path or args.in_path
    fix(args.in_path, out_path, args.backup)


if __name__ == "__main__":
    main()
