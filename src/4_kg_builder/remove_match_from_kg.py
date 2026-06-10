"""
remove_match_from_kg.py
───────────────────────
Delete every triple in data/kg_output/ekg.ttl that belongs to a single
match: the match node itself + all of its events. Shared nodes
(players, teams) are left alone so other matches keep working.

Usage:
    python src/4_kg_builder/remove_match_from_kg.py "Leeds"
    python src/4_kg_builder/remove_match_from_kg.py "Leeds" --dry-run
    python src/4_kg_builder/remove_match_from_kg.py "Leeds" --ttl path/to/ekg.ttl
"""

import argparse
import shutil
import sys
from pathlib import Path

from rdflib import Graph, URIRef

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TTL_PATH = BASE_DIR / "data" / "kg_output" / "ekg.ttl"


def find_matches(g: Graph, name_filter: str) -> list[URIRef]:
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?match ?label WHERE {
        ?match a ekg:Match .
        OPTIONAL { ?match rdfs:label ?label }
    }
    """
    hits = []
    nf   = name_filter.lower()
    for r in g.query(q):
        label = str(r.label) if r.label else ""
        uri   = str(r.match)
        if nf in label.lower() or nf in uri.lower():
            hits.append((URIRef(uri), label))
    return hits


def events_in_match(g: Graph, match_uri: URIRef) -> list[URIRef]:
    q = """
    PREFIX ekg: <http://soccerekg.org/ontology#>
    SELECT DISTINCT ?e WHERE { ?e ekg:inMatch <%s> }
    """ % str(match_uri)
    return [URIRef(str(r.e)) for r in g.query(q)]


def purge_subject(g: Graph, node: URIRef) -> int:
    """Remove every triple where ?node is the subject OR the object.
       Returns count removed."""
    n = 0
    for s, p, o in list(g.triples((node, None, None))):
        g.remove((s, p, o)); n += 1
    for s, p, o in list(g.triples((None, None, node))):
        g.remove((s, p, o)); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Partial match name (e.g. 'Leeds')")
    ap.add_argument("--ttl", default=str(TTL_PATH), help="Path to ekg.ttl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be removed without writing")
    args = ap.parse_args()

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        print(f"TTL not found: {ttl_path}")
        sys.exit(1)

    print(f"Loading {ttl_path} …")
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    print(f"  {len(g):,} triples loaded")

    matches = find_matches(g, args.name)
    if not matches:
        print(f"No match found for '{args.name}'.")
        sys.exit(1)

    if len(matches) > 1:
        print(f"Multiple matches found for '{args.name}':")
        for uri, label in matches:
            print(f"  - {label or uri}")
        print("Refusing to delete more than one match. Be more specific.")
        sys.exit(1)

    match_uri, match_label = matches[0]
    name_shown = match_label or str(match_uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    print(f"\nTarget match: {name_shown}")
    print(f"  URI: {match_uri}")

    events = events_in_match(g, match_uri)
    print(f"  Events to remove: {len(events)}")

    before = len(g)
    removed_total = 0
    for ev in events:
        removed_total += purge_subject(g, ev)
    removed_total += purge_subject(g, match_uri)
    after = len(g)

    print(f"  Triples removed: {before - after:,} "
          f"({removed_total:,} touched, deltas may overlap)")
    print(f"  Triples remaining: {after:,}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    backup = ttl_path.with_suffix(ttl_path.suffix + ".bak")
    shutil.copy2(ttl_path, backup)
    print(f"\nBackup written: {backup.name}")

    g.serialize(destination=str(ttl_path), format="turtle")
    print(f"Saved updated TTL: {ttl_path}")


if __name__ == "__main__":
    main()
