"""
strip_reification.py — one-off migration: fix 035

Removes RDF reification of playsFor from an existing ekg.ttl and replaces
each reified edge with a single direct triple.

Before (reified, broken in Protégé / HermiT):
    edge_uri  rdf:type      rdf:Statement
    edge_uri  rdf:subject   <player>
    edge_uri  rdf:predicate ekg:playsFor
    edge_uri  rdf:object    <team>
    edge_uri  ekg:validFrom "2019-10-01"^^xsd:date

After (direct):
    <player>  ekg:playsFor  <team>

Usage:
    # default: clean ekg.ttl in-place, auto-backup first
    python src/4_kg_builder/strip_reification.py

    # custom paths
    python src/4_kg_builder/strip_reification.py \\
        --in  data/kg_output/ekg.ttl \\
        --out data/kg_output/ekg_clean.ttl

    # skip backup (e.g. already made one)
    python src/4_kg_builder/strip_reification.py --no-backup
"""

import argparse
import shutil
import sys
from pathlib import Path

from rdflib import Graph, Namespace, RDF, URIRef

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTL = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
DEFAULT_BAK = BASE_DIR / "data" / "kg_output" / "ekg_pre_fix_035.ttl.bak"

FIND_REIFIED_Q = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ekg: <http://soccerekg.org/ontology#>
SELECT ?edge ?player ?team WHERE {
    ?edge rdf:type      rdf:Statement ;
          rdf:subject   ?player ;
          rdf:predicate ekg:playsFor ;
          rdf:object    ?team .
}
"""

COUNT_STATEMENTS_Q = """
SELECT (COUNT(?e) AS ?n) WHERE {
    ?e a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
}
"""

COUNT_PLAYS_FOR_Q = """
PREFIX ekg: <http://soccerekg.org/ontology#>
SELECT (COUNT(*) AS ?n) WHERE {
    ?p ekg:playsFor ?t .
}
"""


def _count(g: Graph, q: str) -> int:
    for row in g.query(q):
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0
    return 0


def strip(in_path: Path, out_path: Path, backup: bool) -> int:
    """
    Load in_path, strip reification, save to out_path.
    Returns the number of reified edges removed.
    """
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}")
        sys.exit(1)

    print(f"Loading {in_path} ...")
    g = Graph()
    g.parse(str(in_path), format="turtle")

    n_before       = len(g)
    n_reified      = _count(g, COUNT_STATEMENTS_Q)
    n_direct_before = _count(g, COUNT_PLAYS_FOR_Q)

    print(f"  Total triples          : {n_before:,}")
    print(f"  rdf:Statement nodes    : {n_reified}")
    print(f"  Direct playsFor triples: {n_direct_before}")

    if n_reified == 0:
        print("\nAlready clean — nothing to do.")
        return 0

    # ── collect all reified playsFor edges ────────────────────────────────
    reified_edges = list(g.query(FIND_REIFIED_Q))
    print(f"\nFound {len(reified_edges)} reified playsFor edge(s) to migrate.")

    # ── backup before any write ────────────────────────────────────────────
    if backup:
        print(f"Backing up → {DEFAULT_BAK} ...")
        shutil.copy2(str(in_path), str(DEFAULT_BAK))
        print("  Backup written.")

    # ── migrate ────────────────────────────────────────────────────────────
    added = skipped = removed_nodes = 0

    for row in reified_edges:
        edge: URIRef   = row.edge
        player: URIRef = row.player
        team: URIRef   = row.team

        # Add direct triple (skip if already present)
        direct = (player, EKG.playsFor, team)
        if direct not in g:
            g.add(direct)
            added += 1
        else:
            skipped += 1

        # Remove every triple about this edge node (catches validFrom and
        # any other metadata attached to it, not just the 4 reification triples)
        edge_triples = list(g.predicate_objects(edge))
        for pred, obj in edge_triples:
            g.remove((edge, pred, obj))
        # Also remove triples where edge appears as object (shouldn't exist
        # in practice, but be thorough)
        for subj, pred in list(g.subject_predicates(edge)):
            g.remove((subj, pred, edge))

        removed_nodes += 1

    n_after        = len(g)
    n_direct_after = _count(g, COUNT_PLAYS_FOR_Q)

    print(f"\n  Reified edges removed  : {removed_nodes}")
    print(f"  Direct triples added   : {added}  (skipped duplicates: {skipped})")
    print(f"  Total triples now      : {n_after:,}")
    print(f"  Net change             : {n_after - n_before:+,} triples")
    print(f"  Direct playsFor now    : {n_direct_after}")

    # ── sanity check ──────────────────────────────────────────────────────
    remaining = _count(g, COUNT_STATEMENTS_Q)
    if remaining != 0:
        print(f"\nERROR: {remaining} rdf:Statement triple(s) still present — "
              f"strip incomplete. NOT saving.")
        sys.exit(1)

    # ── save ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving → {out_path} ...")
    g.serialize(destination=str(out_path), format="turtle")
    print("Done.")

    return removed_nodes


def main():
    ap = argparse.ArgumentParser(
        description="Strip RDF reification of playsFor from ekg.ttl (fix 035).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--in",       dest="in_path",  type=Path,
                    default=DEFAULT_TTL,
                    help=f"Input TTL (default: {DEFAULT_TTL})")
    ap.add_argument("--out",      dest="out_path", type=Path,
                    default=None,
                    help="Output TTL (default: same as --in, overwrite in-place)")
    ap.add_argument("--no-backup", dest="backup",  action="store_false",
                    default=True,
                    help="Skip the automatic .bak file")
    args = ap.parse_args()

    out_path = args.out_path or args.in_path
    strip(args.in_path, out_path, args.backup)


if __name__ == "__main__":
    main()
