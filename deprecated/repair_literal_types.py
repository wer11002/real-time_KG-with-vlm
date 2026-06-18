"""
repair_literal_types.py — one-off migration: fix all literal datatypes in ekg.ttl

For every triple whose predicate is listed in PROP_TYPES:
  - Already correct datatype  → leave unchanged
  - Wrong datatype / no type  → retype to the correct xsd: type
  - Conversion fails          → DROP the triple entirely (log warning)

This brings existing data in line with ekg_tbox.ttl without re-running the VLM.
Auto-backup written before any write.

Usage:
    python src/4_kg_builder/repair_literal_types.py
    python src/4_kg_builder/repair_literal_types.py --in data/kg_output/ekg.ttl --no-backup
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, URIRef, XSD

_SCHEMA_DIR = Path(__file__).resolve().parent
if str(_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_DIR))

from ekg_schema import PROP_TYPES, typed_literal

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTL = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
DEFAULT_BAK = BASE_DIR / "data" / "kg_output" / "ekg_pre_repair_literals.ttl.bak"
LOG_DIR     = BASE_DIR / "data" / "logs"

logging.basicConfig(
    level=logging.WARNING,
    handlers=[
        logging.FileHandler(str(LOG_DIR / "typing_warnings.log")),
        logging.StreamHandler(),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
)


def repair(in_path: Path, out_path: Path, backup: bool) -> None:
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
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

    retyped   = defaultdict(int)
    dropped   = defaultdict(int)
    unchanged = defaultdict(int)

    # Collect all triples touching PROP_TYPES predicates first (mutating while
    # iterating is unsafe)
    candidates = []
    for s, p, o in g:
        if not isinstance(p, URIRef):
            continue
        local = str(p).split("#")[-1]
        if local in PROP_TYPES:
            candidates.append((s, p, o, local))

    # Explicit pass: any remaining xsd:date literals anywhere in the graph
    # (e.g. from old pipeline runs before fix 045).
    # HermiT doesn't support xsd:date — coerce to xsd:dateTime midnight UTC.
    date_converted = 0
    for s, p, o in list(g.triples((None, None, None))):
        if isinstance(o, Literal) and o.datatype == XSD.date:
            new_lit = Literal(f"{str(o)}T00:00:00Z", datatype=XSD.dateTime)
            g.remove((s, p, o))
            g.add((s, p, new_lit))
            date_converted += 1

    if date_converted:
        print(f"\n=== RETYPED xsd:date → xsd:dateTime (HermiT workaround) ===")
        print(f"  {date_converted} literal(s) converted")
        print(f"  e.g. '2019-10-01'^^xsd:date → '2019-10-01T00:00:00Z'^^xsd:dateTime")

    print(f"\n  Candidates to inspect : {len(candidates):,}")

    for s, p, o, local in candidates:
        target = PROP_TYPES[local]

        if isinstance(o, Literal) and o.datatype == target:
            unchanged[local] += 1
            continue

        new_lit = typed_literal(local, o, context=f"repair {s}")

        if new_lit is None:
            g.remove((s, p, o))
            dropped[local] += 1
        elif new_lit != o:
            g.remove((s, p, o))
            g.add((s, p, new_lit))
            retyped[local] += 1
        else:
            unchanged[local] += 1

    n_after = len(g)

    # ── report ─────────────────────────────────────────────────────────────
    print("\n=== RETYPED ===")
    if retyped:
        for prop, n in sorted(retyped.items()):
            tgt = str(PROP_TYPES[prop]).split("#")[-1]
            print(f"  {prop:<28}  {n:>5}  → xsd:{tgt}")
    else:
        print("  (none)")

    print("\n=== DROPPED (coercion failed) ===")
    if dropped:
        for prop, n in sorted(dropped.items()):
            print(f"  {prop:<28}  {n:>5}")
    else:
        print("  (none — all values were convertible)")

    print("\n=== UNCHANGED (already typed correctly) ===")
    if unchanged:
        for prop, n in sorted(unchanged.items()):
            print(f"  {prop:<28}  {n:>5}")
    else:
        print("  (none)")

    total_retyped = sum(retyped.values())
    total_dropped = sum(dropped.values())
    print(f"\n=== SUMMARY ===")
    print(f"  Total triples before : {n_before:,}")
    print(f"  Total triples after  : {n_after:,}")
    print(f"  Retyped              : {total_retyped:,}")
    print(f"  Dropped (corrupted)  : {total_dropped:,}")
    print(f"  Net change           : {n_after - n_before:+,}")

    if total_dropped:
        print(f"\n  Dropped values logged → data/logs/typing_warnings.log")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving → {out_path} ...")
    g.serialize(destination=str(out_path), format="turtle")
    print("Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Repair xsd: datatypes in ekg.ttl in-place.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--in",        dest="in_path",  type=Path, default=DEFAULT_TTL)
    ap.add_argument("--out",       dest="out_path", type=Path, default=None)
    ap.add_argument("--no-backup", dest="backup",   action="store_false", default=True)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out_path or args.in_path
    repair(args.in_path, out_path, args.backup)


if __name__ == "__main__":
    main()
