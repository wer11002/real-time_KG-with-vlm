"""
check_consistency.py — CPU-only HermiT consistency check via owlready2

owlready2 cannot parse Turtle directly, so the script converts the TTL to a
temp RDF/XML file first, runs HermiT, then deletes the temp file.

Usage:
    python src/4_kg_builder/check_consistency.py
    python src/4_kg_builder/check_consistency.py --ttl data/kg_output/ekg.ttl
    python src/4_kg_builder/check_consistency.py --tbox-only
"""

import argparse
import sys
import warnings
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTL  = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
DEFAULT_TBOX = BASE_DIR / "ekg_tbox.ttl"

warnings.filterwarnings("ignore", message=".*ignoring cyclic subclass.*")


def _convert_ttl_to_xml(ttl_path: Path) -> Path:
    from rdflib import Graph
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    xml_path = ttl_path.with_suffix(".tmp.owl")
    g.serialize(destination=str(xml_path), format="pretty-xml")
    print(f"  Converted {ttl_path.name} → {xml_path.name} ({len(g):,} triples)")
    return xml_path


def run(ttl_path: Path, tbox_only: bool) -> bool:
    try:
        import owlready2
        from owlready2 import get_ontology, sync_reasoner_hermit, Nothing
    except ImportError:
        print("owlready2 not installed. Run: pip install owlready2")
        sys.exit(1)

    target = DEFAULT_TBOX if tbox_only else ttl_path
    if not target.exists():
        print(f"ERROR: file not found: {target}")
        sys.exit(1)

    print(f"Loading {target} ...")
    xml_path = _convert_ttl_to_xml(target)

    consistent = True
    try:
        onto = get_ontology(f"file://{xml_path.resolve()}").load()
        print("Running HermiT ...")
        try:
            with onto:
                sync_reasoner_hermit(infer_property_values=False)
            print("\n✓  Ontology is CONSISTENT")
        except owlready2.base.OwlReadyInconsistentOntologyError as e:
            consistent = False
            print(f"\n✗  Inconsistency detected:")
            print(f"   {e}")

            bad = [cls for cls in onto.classes() if Nothing in cls.equivalent_to]
            print(f"\n   Inconsistent classes (equivalent to Nothing):")
            if bad:
                for cls in bad:
                    print(f"     ✗ {cls.iri}")
            else:
                print("     (owlready2 could not enumerate — check Protégé for details)")

            print(f"\n   Sample inconsistent individuals:")
            shown = 0
            for cls in bad[:5]:
                for ind in list(cls.instances())[:3]:
                    print(f"     - {ind.iri}")
                    shown += 1
            if shown == 0:
                print("     (none retrievable from owlready2)")
    finally:
        if xml_path.exists():
            xml_path.unlink()
            print("\n   Cleaned up temp file.")

    return consistent


def main():
    ap = argparse.ArgumentParser(
        description="Run HermiT consistency check on ekg.ttl (converts TTL→RDF/XML first).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ttl",       type=Path, default=DEFAULT_TTL,
                    help=f"TTL to check (default: {DEFAULT_TTL})")
    ap.add_argument("--tbox-only", action="store_true",
                    help="Check T-Box only (skip A-Box instances)")
    args = ap.parse_args()

    ok = run(args.ttl, args.tbox_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
