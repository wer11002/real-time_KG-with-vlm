"""
migrate_to_new_tbox.py
─────────────────────────────────────────────────────────────────────
One-off transformation of data/kg_output/ekg.ttl from the old messy
class structure into the clean T-Box defined in ekg_tbox.ttl at the
project root.

What this script does, in order:

    1. Backs up the existing A-Box           → ekg_old.ttl.bak
    2. Loads the A-Box and the new T-Box.
    3. Classifies every named subject as event / player / team / match
       by inspecting its current rdf:type triples (both ekg:* and the
       foreign prov / schema / foaf vocabularies).
    4. Drops every rdf:type triple from those subjects, then re-asserts
       exactly ONE clean class (from CLASS_MAP) + owl:NamedIndividual.
       The T-Box's subClassOf chain handles the parent classes — we
       never re-assert them here.
    5. Renames every old object/data property to its new camelCase form
       using PROPERTY_MAP, including when the same URI appears as the
       OBJECT of a triple (e.g. owl:inverseOf, rdfs:subPropertyOf).
    6. Drops any A-Box subject that was typed as owl:Class / ObjectProperty
       / DatatypeProperty / Restriction — those are T-Box concerns and
       the new T-Box re-defines them authoritatively.
    7. Merges the cleaned A-Box with the new T-Box and saves back to
       data/kg_output/ekg.ttl.

Usage:
    python src/4_kg_builder/migrate_to_new_tbox.py
"""

import shutil
import sys
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
ABOX_PATH   = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
BACKUP_PATH = BASE_DIR / "data" / "kg_output" / "ekg_old.ttl.bak"
TBOX_PATH   = BASE_DIR / "ekg_tbox.ttl"

EKG    = Namespace("http://soccerekg.org/ontology#")
DATA   = Namespace("http://soccerekg.org/data#")
PROV   = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("http://schema.org/")
FOAF   = Namespace("http://xmlns.com/foaf/0.1/")


# ════════════════════════════════════════════════════════════════════
# MAPPINGS
# ════════════════════════════════════════════════════════════════════

# Old class URI → new class URI. The order is also a priority list when
# a subject carries multiple old types — the first match wins, so more
# specific classes (Goal, RedCard) appear before more general ones
# (Card, ActionEvent).
CLASS_MAP: dict[URIRef, URIRef] = {
    EKG.GoalEvent         : EKG.Goal,
    EKG.RedCardEvent      : EKG.RedCard,
    EKG.YellowCardEvent   : EKG.YellowCard,
    EKG.CardEvent         : EKG.Card,
    EKG.ShotEvent         : EKG.Shot,
    EKG.FoulEvent         : EKG.Foul,
    EKG.CornerEvent       : EKG.Corner,
    EKG.FreeKickEvent     : EKG.FreeKick,
    EKG.SubstitutionEvent : EKG.Substitution,
    EKG.OffsideEvent      : EKG.OffsideCalled,
    EKG.ActionEvent       : EKG.PlayerAction,
    EKG.MatchEvent        : EKG.LeagueMatch,
    EKG.Match             : EKG.LeagueMatch,
}

# Old property URI → new property URI.
PROPERTY_MAP: dict[URIRef, URIRef] = {
    EKG.IS_PERFORMED_BY: EKG.isPerformedBy,
    EKG.IN_MATCH       : EKG.inMatch,
    EKG.INVOLVED_IN    : EKG.involvedTeam,
    EKG.PRECEDED_BY    : EKG.precededBy,
    EKG.PRECEDES       : EKG.precedes,
    EKG.PERFORMED      : EKG.performed,
    EKG.PLAYS_FOR      : EKG.playsFor,
    EKG.PARTICIPATED_IN: EKG.participatedIn,
    EKG.HAS_PLAYER     : EKG.hasPlayer,
    EKG.TRIGGERED      : EKG.triggered,
    EKG.ASSISTED_BY    : EKG.assistedBy,
    # ekg:hasHomeTeam / ekg:hasAwayTeam already use the new naming.
}

# Foreign vocabulary classes that should be removed entirely.
FOREIGN_TYPES: set[URIRef] = {
    PROV.Activity, PROV.Agent,
    SCHEMA.Action, SCHEMA.SportsTeam, SCHEMA.SportsEvent, SCHEMA.Person,
    FOAF.Person,  FOAF.Agent, FOAF.Group,
}

# Subjects we coalesce into ekg:Player / ekg:Team / ekg:LeagueMatch.
PLAYER_TYPES: set[URIRef] = {EKG.Player, FOAF.Person, SCHEMA.Person}
TEAM_TYPES  : set[URIRef] = {EKG.Team,   SCHEMA.SportsTeam, FOAF.Group}
MATCH_TYPES : set[URIRef] = {EKG.Match,  EKG.MatchEvent, SCHEMA.SportsEvent}

# Any subject typed as one of these is a T-Box concern. Its entire set
# of triples is dropped so the new T-Box can re-define everything.
TBOX_SUBJECT_TYPES: set[URIRef] = {
    OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
    OWL.Restriction, OWL.AnnotationProperty, OWL.Ontology,
    RDFS.Class,
}

# Event-type priority for classification (most specific first).
EVENT_PRIORITY: list[URIRef] = [
    EKG.GoalEvent, EKG.RedCardEvent, EKG.YellowCardEvent, EKG.CardEvent,
    EKG.ShotEvent, EKG.FoulEvent,    EKG.CornerEvent,     EKG.FreeKickEvent,
    EKG.SubstitutionEvent, EKG.OffsideEvent, EKG.ActionEvent,
]


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _short(uri: URIRef) -> str:
    """Pretty-print a URI as 'prefix:local' for log output."""
    s = str(uri)
    if s.startswith(str(EKG)):
        return f"ekg:{s[len(str(EKG)):]}"
    if s.startswith(str(DATA)):
        return f"data:{s[len(str(DATA)):]}"
    if s.startswith(str(OWL)):
        return f"owl:{s[len(str(OWL)):]}"
    if s.startswith(str(FOAF)):
        return f"foaf:{s[len(str(FOAF)):]}"
    if s.startswith(str(SCHEMA)):
        return f"schema:{s[len(str(SCHEMA)):]}"
    if s.startswith(str(PROV)):
        return f"prov:{s[len(str(PROV)):]}"
    return s.split("#")[-1].split("/")[-1]


def _print_mapping_summary():
    print("\n  Class mapping (old → new):")
    for old, new in CLASS_MAP.items():
        print(f"    {_short(old):<28} → {_short(new)}")
    print("\n  Property mapping (old → new):")
    for old, new in PROPERTY_MAP.items():
        print(f"    {_short(old):<28} → {_short(new)}")
    print("\n  Foreign-vocab types being dropped:")
    print(f"    {', '.join(sorted(_short(t) for t in FOREIGN_TYPES))}")


def _classify_subject(types: set[URIRef]) -> URIRef | None:
    """
    Decide what one clean rdf:type a subject should end up with, based
    on the bag of types it currently has. Returns None for subjects we
    don't recognise (they'll be left untouched in the A-Box).
    """
    for old in EVENT_PRIORITY:
        if old in types:
            return CLASS_MAP[old]
    if types & MATCH_TYPES:
        return EKG.LeagueMatch
    if types & PLAYER_TYPES:
        return EKG.Player
    if types & TEAM_TYPES:
        return EKG.Team
    return None


def _bind_namespaces(g: Graph):
    g.bind("ekg",  EKG)
    g.bind("data", DATA)
    g.bind("rdfs", RDFS)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    if not ABOX_PATH.exists():
        print(f"A-Box not found: {ABOX_PATH}"); sys.exit(1)
    if not TBOX_PATH.exists():
        print(f"T-Box not found: {TBOX_PATH}"); sys.exit(1)

    _print_mapping_summary()

    print(f"\nBacking up   {ABOX_PATH} → {BACKUP_PATH.name}")
    shutil.copy2(ABOX_PATH, BACKUP_PATH)

    print(f"Loading A-Box {ABOX_PATH}")
    abox = Graph(); abox.parse(str(ABOX_PATH), format="turtle")
    before = len(abox)
    print(f"  {before:,} triples")

    print(f"Loading T-Box {TBOX_PATH}")
    tbox = Graph(); tbox.parse(str(TBOX_PATH), format="turtle")
    print(f"  {len(tbox):,} triples")

    # ── PASS 1: collect types per subject ──────────────────────────
    types_per_subject: dict[URIRef, set[URIRef]] = {}
    for s, _, o in abox.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            types_per_subject.setdefault(s, set()).add(o)

    # Subjects whose triples should be dropped wholesale (T-Box junk).
    tbox_subjects = {
        s for s, ts in types_per_subject.items()
        if ts & TBOX_SUBJECT_TYPES
    }

    # Classify the rest.
    subject_new_type: dict[URIRef, URIRef] = {}
    counts = {"event": 0, "player": 0, "team": 0,
              "match": 0, "other": 0, "dropped_tbox": len(tbox_subjects)}
    for s, ts in types_per_subject.items():
        if s in tbox_subjects:
            continue
        new = _classify_subject(ts)
        if new is None:
            counts["other"] += 1
            continue
        subject_new_type[s] = new
        if new == EKG.LeagueMatch: counts["match"]  += 1
        elif new == EKG.Player   : counts["player"] += 1
        elif new == EKG.Team     : counts["team"]   += 1
        else                     : counts["event"]  += 1

    # ── PASS 2: build cleaned A-Box ────────────────────────────────
    cleaned = Graph(); _bind_namespaces(cleaned)
    for s, p, o in abox:
        # Skip whole subjects that were T-Box leftovers.
        if s in tbox_subjects:
            continue
        # Drop rdf:type triples for subjects we've reclassified, AND any
        # rdf:type whose object is a foreign-vocab class.
        if p == RDF.type:
            if s in subject_new_type:
                continue
            if o in FOREIGN_TYPES:
                continue
        # Rename predicate, and rename object too in case the property
        # or class URI appears on the right of a triple (e.g.
        # owl:inverseOf, rdfs:subClassOf, rdfs:domain, rdfs:range).
        new_p = PROPERTY_MAP.get(p, p)
        new_o = CLASS_MAP.get(o, PROPERTY_MAP.get(o, o))
        cleaned.add((s, new_p, new_o))

    # Re-assert exactly one clean rdf:type per known subject + NI.
    for s, new_t in subject_new_type.items():
        cleaned.add((s, RDF.type, new_t))
        cleaned.add((s, RDF.type, OWL.NamedIndividual))

    # ── merge with T-Box ───────────────────────────────────────────
    result = Graph(); _bind_namespaces(result)
    for t in cleaned: result.add(t)
    for t in tbox   : result.add(t)

    print(f"\nSaving merged graph → {ABOX_PATH}")
    result.serialize(destination=str(ABOX_PATH), format="turtle")

    # ── stats ──────────────────────────────────────────────────────
    after  = len(result)
    inst   = counts["event"] + counts["player"] + counts["team"] + counts["match"]
    print(f"\n{'═'*60}")
    print("  MIGRATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  Before  : {before:>7,} triples")
    print(f"  After   : {after:>7,} triples  (delta {after - before:+,})")
    print(f"  Instances cleaned: {inst:,}")
    print(f"    Events       : {counts['event']:>5}")
    print(f"    Players      : {counts['player']:>5}")
    print(f"    Teams        : {counts['team']:>5}")
    print(f"    Matches      : {counts['match']:>5}")
    print(f"    Other (kept) : {counts['other']:>5}")
    print(f"    T-Box subjects dropped: {counts['dropped_tbox']:>5}")

    print(f"\n  Class assertions per cleaned instance "
          f"(excluding owl:NamedIndividual):")
    sample = list(subject_new_type.items())[:5]
    for s, _ in sample:
        types_now = [o for _, _, o in result.triples((s, RDF.type, None))
                     if o != OWL.NamedIndividual]
        local = str(s).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        print(f"    {local[:50]:<50} {len(types_now)} type(s)")

    print(f"\n  Backup saved at: {BACKUP_PATH}")
    print(f"  Re-running this script is a no-op — second pass finds "
          f"nothing to migrate.")


if __name__ == "__main__":
    main()
