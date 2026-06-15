"""
check_consistency.py — CPU-only HermiT consistency check via owlready2

Loads ekg_tbox.ttl + ekg.ttl and runs HermiT. Reports CONSISTENT or lists
the classes inferred as owl:Nothing so you can iterate without opening Protégé.

Usage:
    python src/4_kg_builder/check_consistency.py
    python src/4_kg_builder/check_consistency.py --ttl data/kg_output/ekg.ttl
    python src/4_kg_builder/check_consistency.py --tbox-only   # skip A-Box
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTL  = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
DEFAULT_TBOX = BASE_DIR / "ekg_tbox.ttl"


def run(ttl_path: Path, tbox_only: bool) -> bool:
    try:
        import owlready2
        from owlready2 import get_ontology, sync_reasoner_hermit, owl
    except ImportError:
        print("owlready2 not installed. Run: pip install owlready2")
        sys.exit(1)

    target = tbox_only and DEFAULT_TBOX or ttl_path
    if not target.exists():
        print(f"ERROR: file not found: {target}")
        sys.exit(1)

    print(f"Loading {target} ...")
    onto = get_ontology(f"file://{target.resolve()}").load()
    print(f"  Loaded. Running HermiT ...")

    consistent = True
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=False)
        print("\n✓  Ontology is CONSISTENT")
    except owlready2.base.OwlReadyInconsistentOntologyError as e:
        consistent = False
        print(f"\n✗  Ontology is INCONSISTENT")
        print(f"   Reasoner: {e}")

        bad = [cls for cls in onto.classes()
               if owl.Nothing in cls.equivalent_to]
        if bad:
            print(f"\n   Classes inferred as owl:Nothing ({len(bad)}):")
            for cls in bad:
                print(f"     - {cls.name}")
        else:
            print("\n   (owlready2 could not enumerate inconsistent classes)")

    return consistent


def main():
    ap = argparse.ArgumentParser(
        description="Run HermiT consistency check on ekg.ttl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ttl",       type=Path, default=DEFAULT_TTL,
                    help=f"A-Box TTL to check (default: {DEFAULT_TTL})")
    ap.add_argument("--tbox-only", action="store_true",
                    help="Check T-Box only (no A-Box instances)")
    args = ap.parse_args()

    ok = run(args.ttl, args.tbox_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
