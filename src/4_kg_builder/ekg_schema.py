"""
ekg_schema.py — RDF/OWL Schema for Soccer Event Knowledge Graph
────────────────────────────────────────────────────────────────

Defines the ontology (T-Box) and provides the EKG container (A-Box holder).

T-Box: Classes and properties of the soccer domain
   Classes    : Player, Team, Match, Event, ActionEvent, CardEvent,
                GoalEvent, ShotEvent, FoulEvent, CornerEvent, OffsideEvent,
                FreeKickEvent, SubstitutionEvent, PenaltyEvent,
                YellowCardEvent, RedCardEvent
   Properties : PERFORMED (+ inverse IS_PERFORMED_BY),
                PLAYS_FOR, PARTICIPATED_IN, IN_MATCH,
                PRECEDED_BY (+ inverse PRECEDES), TRIGGERED,
                ASSISTED_BY, INVOLVED_IN,
                hasHomeTeam, hasAwayTeam

A-Box: Instance data populated incrementally during the pipeline
   (added by kg_builder.py as events stream in)

TKG layer: validFrom / validUntil on PLAYS_FOR edges (RDF standard reification)
VLM layer: hasDescription / detectedJersey on Event nodes

Serialization: Turtle (.ttl) by default.

Quick test:
    python ekg_schema.py
"""

from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


# ═══════════════════════════════════════════════════════════════════════════
# NAMESPACES
# ═══════════════════════════════════════════════════════════════════════════

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")


# ═══════════════════════════════════════════════════════════════════════════
# CLASSES (T-Box)
# ═══════════════════════════════════════════════════════════════════════════

CLASSES = {
    "Player"             : EKG.Player,
    "Team"               : EKG.Team,
    "Match"              : EKG.Match,
    "Event"              : EKG.Event,
    # mid-level
    "ActionEvent"        : EKG.ActionEvent,
    "CardEvent"          : EKG.CardEvent,
    # specific action subtypes (enable OWL reasoning over event types)
    "GoalEvent"          : EKG.GoalEvent,
    "ShotEvent"          : EKG.ShotEvent,
    "FoulEvent"          : EKG.FoulEvent,
    "CornerEvent"        : EKG.CornerEvent,
    "OffsideEvent"       : EKG.OffsideEvent,
    "FreeKickEvent"      : EKG.FreeKickEvent,
    "SubstitutionEvent"  : EKG.SubstitutionEvent,
    "PenaltyEvent"       : EKG.PenaltyEvent,
    # card subtypes
    "YellowCardEvent"    : EKG.YellowCardEvent,
    "RedCardEvent"       : EKG.RedCardEvent,
}

# class hierarchy — (child, parent)
CLASS_HIERARCHY = [
    (EKG.ActionEvent,       EKG.Event),
    (EKG.CardEvent,         EKG.Event),
    (EKG.GoalEvent,         EKG.ActionEvent),
    (EKG.ShotEvent,         EKG.ActionEvent),
    (EKG.FoulEvent,         EKG.ActionEvent),
    (EKG.CornerEvent,       EKG.ActionEvent),
    (EKG.OffsideEvent,      EKG.ActionEvent),
    (EKG.FreeKickEvent,     EKG.ActionEvent),
    (EKG.SubstitutionEvent, EKG.ActionEvent),
    (EKG.PenaltyEvent,      EKG.ActionEvent),
    (EKG.YellowCardEvent,   EKG.CardEvent),
    (EKG.RedCardEvent,      EKG.CardEvent),
]

# map event-type string → OWL class URI (used by kg_builder)
EVENT_TYPE_CLASS = {
    "Goal"        : EKG.GoalEvent,
    "Shot"        : EKG.ShotEvent,
    "Foul"        : EKG.FoulEvent,
    "Corner"      : EKG.CornerEvent,
    "Offside"     : EKG.OffsideEvent,
    "FreeKick"    : EKG.FreeKickEvent,
    "Free_Kick"   : EKG.FreeKickEvent,
    "Substitution": EKG.SubstitutionEvent,
    "Penalty"     : EKG.PenaltyEvent,
    "YellowCard"  : EKG.YellowCardEvent,
    "RedCard"     : EKG.RedCardEvent,
}


# ═══════════════════════════════════════════════════════════════════════════
# OBJECT PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

OBJECT_PROPERTIES = {
    # entity → event
    "PERFORMED"       : (EKG.PERFORMED,       EKG.Player, EKG.Event),
    "INVOLVED_IN"     : (EKG.INVOLVED_IN,     EKG.Team,   EKG.Event),
    "ASSISTED_BY"     : (EKG.ASSISTED_BY,     EKG.Event,  EKG.Player),
    # inverse of PERFORMED
    "IS_PERFORMED_BY" : (EKG.IS_PERFORMED_BY, EKG.Event,  EKG.Player),
    # entity → entity
    "PLAYS_FOR"       : (EKG.PLAYS_FOR,       EKG.Player, EKG.Team),
    "PARTICIPATED_IN" : (EKG.PARTICIPATED_IN, EKG.Player, EKG.Match),
    "IN_MATCH"        : (EKG.IN_MATCH,        EKG.Event,  EKG.Match),
    # match → team
    "hasHomeTeam"     : (EKG.hasHomeTeam,     EKG.Match,  EKG.Team),
    "hasAwayTeam"     : (EKG.hasAwayTeam,     EKG.Match,  EKG.Team),
    # event → event
    "PRECEDED_BY"     : (EKG.PRECEDED_BY,     EKG.Event,  EKG.Event),
    "PRECEDES"        : (EKG.PRECEDES,        EKG.Event,  EKG.Event),
    "TRIGGERED"       : (EKG.TRIGGERED,       EKG.ActionEvent, EKG.CardEvent),
}

# property inverses declared in the T-Box
INVERSE_PAIRS = [
    (EKG.IS_PERFORMED_BY, EKG.PERFORMED),
    (EKG.PRECEDES,        EKG.PRECEDED_BY),
]


# ═══════════════════════════════════════════════════════════════════════════
# DATATYPE PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

DATATYPE_PROPERTIES = {
    # ── event ──────────────────────────────────────────────────────────────
    "hasTime"        : XSD.string,   # "9'" or "45+2'"
    "hasEventType"   : XSD.string,   # "Shot" — human-readable label (use rdf:type for reasoning)
    "hasConfidence"  : XSD.float,
    "hasFullText"    : XSD.string,   # commentary text from ESPN
    "hasDate"        : XSD.string,   # "2019-10-01"

    # ── TKG layer (temporal validity on PLAYS_FOR edges) ──────────────────
    "validFrom"      : XSD.date,
    "validUntil"     : XSD.date,

    # ── VLM layer (from Qwen2-VL output) ─────────────────────────────────
    "hasDescription" : XSD.string,
    "detectedJersey" : XSD.string,  # jersey number read by VLM on an event

    # ── Player roster data ────────────────────────────────────────────────
    "hasJerseyNumber": XSD.string,  # permanent squad jersey number on Player node
}

# NOTE: isMatched (data quality flag) and hasTimeMin (redundant float copy of
# hasTime) are intentionally excluded from the T-Box — they are not part of
# the domain ontology. isMatched is written to the A-Box as a provenance
# annotation by kg_builder but is not a declared ontology property.


# ═══════════════════════════════════════════════════════════════════════════
# T-BOX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_tbox(g: Graph):
    """Populate the graph with T-Box definitions. Called once at startup."""

    g.bind("ekg",  EKG)
    g.bind("data", INST)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # classes
    for name, uri in CLASSES.items():
        g.add((uri, RDF.type,   OWL.Class))
        g.add((uri, RDFS.label, Literal(name)))

    # class hierarchy
    for child, parent in CLASS_HIERARCHY:
        g.add((child, RDFS.subClassOf, parent))

    # object properties
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        g.add((uri, RDF.type,    OWL.ObjectProperty))
        g.add((uri, RDFS.label,  Literal(name)))
        g.add((uri, RDFS.domain, domain))
        g.add((uri, RDFS.range,  range_))

    # inverse property declarations
    for prop, inverse_of in INVERSE_PAIRS:
        g.add((prop, OWL.inverseOf, inverse_of))

    # datatype properties
    for name, range_ in DATATYPE_PROPERTIES.items():
        uri = EKG[name]
        g.add((uri, RDF.type,   OWL.DatatypeProperty))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, RDFS.range, range_))

    return g


# ═══════════════════════════════════════════════════════════════════════════
# EKG CONTAINER
# ═══════════════════════════════════════════════════════════════════════════

class EKG_Graph:
    """Wraps an rdflib Graph with T-Box pre-loaded. Grows A-Box in real-time."""

    def __init__(self):
        self.g = Graph()
        build_tbox(self.g)
        self._seen_players : set = set()
        self._seen_teams   : set = set()
        self._seen_matches : set = set()
        self._event_count  : int = 0

    # ── URI helpers ────────────────────────────────────────────────────────

    @staticmethod
    def player_uri(player_id: str) -> URIRef:
        return INST[f"player_{player_id}"]

    @staticmethod
    def team_uri(team_id: str) -> URIRef:
        return INST[f"team_{team_id}"]

    @staticmethod
    def match_uri(match_id: str) -> URIRef:
        return INST[f"match_{match_id}"]

    @staticmethod
    def event_uri(event_id: str) -> URIRef:
        return INST[f"event_{event_id}"]

    @staticmethod
    def plays_for_uri(player_id: str, team_id: str, date: str) -> URIRef:
        """URI for a time-bounded PLAYS_FOR edge (RDF standard reification)."""
        return INST[f"plays_for_{player_id}_{team_id}_{date}"]

    # ── stats ──────────────────────────────────────────────────────────────

    def stats(self) -> str:
        return (
            f"{len(self._seen_players)} players | "
            f"{self._event_count} events | "
            f"{len(self._seen_teams)} teams | "
            f"{len(self.g)} triples"
        )

    def triple_count(self) -> int:
        return len(self.g)

    # ── SPARQL query helpers ───────────────────────────────────────────────

    def events_by_type(self, event_type: str) -> list:
        q = """
        SELECT ?e WHERE {
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?etype)
        }
        """
        return [row[0] for row in self.g.query(
            q, initBindings={"etype": Literal(event_type)})]

    def count_cards(self, player_id: str, color: str = "YellowCard") -> int:
        q = """
        SELECT (COUNT(?e) AS ?c) WHERE {
            ?p ekg:PERFORMED ?e .
            ?e a ekg:CardEvent .
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?color)
        }
        """
        result = self.g.query(q, initBindings={
            "p"     : self.player_uri(player_id),
            "color" : Literal(color),
        })
        for row in result:
            return int(row[0])
        return 0

    def events_for_player(self, player_id: str) -> list:
        q = "SELECT ?e WHERE { ?p ekg:PERFORMED ?e . }"
        return [row[0] for row in self.g.query(
            q, initBindings={"p": self.player_uri(player_id)})]

    def player_team_at(self, player_id: str, date: str) -> list:
        """
        TKG query: which team was a player on at a given date?
        Uses standard RDF reification (rdf:Statement) with validFrom/validUntil.
        """
        q = """
        SELECT ?team WHERE {
            ?edge rdf:type      rdf:Statement .
            ?edge rdf:subject   ?p .
            ?edge rdf:predicate ekg:PLAYS_FOR .
            ?edge rdf:object    ?team .
            ?edge ekg:validFrom ?from .
            OPTIONAL { ?edge ekg:validUntil ?until }
            FILTER (?from <= ?date)
            FILTER (!BOUND(?until) || ?until >= ?date)
        }
        """
        return [row[0] for row in self.g.query(q, initBindings={
            "p"    : self.player_uri(player_id),
            "date" : Literal(date, datatype=XSD.date),
        })]

    # ── save ───────────────────────────────────────────────────────────────

    def save(self, out_path: Path, format: str = "turtle"):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=str(out_path), format=format)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── ekg_schema.py self-test ───\n")

    ekg = EKG_Graph()
    print(f"T-Box loaded: {len(ekg.g)} triples\n")

    print("── T-Box classes ──")
    for name, uri in CLASSES.items():
        print(f"  {name:<20} {uri}")

    print("\n── T-Box object properties ──")
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        print(f"  {name:<18} {domain.split('#')[-1]} → {range_.split('#')[-1]}")

    print("\n── Inverse pairs ──")
    for prop, inv in INVERSE_PAIRS:
        print(f"  {prop.split('#')[-1]} owl:inverseOf {inv.split('#')[-1]}")

    print("\n── T-Box datatype properties ──")
    for name, range_ in DATATYPE_PROPERTIES.items():
        tag = ""
        if name in ("validFrom", "validUntil"):
            tag = "  ← TKG"
        elif name in ("hasDescription", "detectedJersey"):
            tag = "  ← VLM"
        elif name == "hasJerseyNumber":
            tag = "  ← Player roster"
        print(f"  {name:<18} → {range_.split('#')[-1]}{tag}")

    print("\n── Test A-Box (with TKG + VLM triples) ──")

    lolley  = ekg.player_uri("joe_lolley")
    team    = ekg.team_uri("nottingham_forest")
    event   = ekg.event_uri("0001")
    edge    = ekg.plays_for_uri("joe_lolley", "nottingham_forest", "2019-10-01")

    # player
    ekg.g.add((lolley, RDF.type,            EKG.Player))
    ekg.g.add((lolley, RDFS.label,          Literal("Joe Lolley")))
    ekg.g.add((lolley, EKG.hasJerseyNumber, Literal("23")))

    # team
    ekg.g.add((team, RDF.type,   EKG.Team))
    ekg.g.add((team, RDFS.label, Literal("Nottingham Forest")))

    # TKG edge: PLAYS_FOR using RDF standard reification (rdf:Statement)
    ekg.g.add((edge, RDF.type,        RDF.Statement))
    ekg.g.add((edge, RDF.subject,     lolley))
    ekg.g.add((edge, RDF.predicate,   EKG.PLAYS_FOR))
    ekg.g.add((edge, RDF.object,      team))
    ekg.g.add((edge, EKG.validFrom,   Literal("2017-07-01", datatype=XSD.date)))
    ekg.g.add((edge, EKG.validUntil,  Literal("2021-06-30", datatype=XSD.date)))

    # event with OWL type + VLM description + detectedJersey
    ekg.g.add((event, RDF.type,            EKG.ShotEvent))
    ekg.g.add((event, RDF.type,            EKG.ActionEvent))
    ekg.g.add((event, EKG.hasEventType,    Literal("Shot")))
    ekg.g.add((event, EKG.hasTime,         Literal("1'")))
    ekg.g.add((event, EKG.detectedJersey,  Literal("23")))
    ekg.g.add((event, EKG.hasDescription,  Literal(
        "Player #23 in red kit takes a left-footed shot from the centre of the box")))

    ekg.g.add((lolley, EKG.PERFORMED,      event))
    ekg.g.add((event,  EKG.IS_PERFORMED_BY, lolley))

    print(f"  {ekg.stats()}")

    out = Path("data/kg_output/test_ekg.ttl")
    ekg.save(out)
    print(f"  Saved to: {out}")

    # verify TKG query
    teams = ekg.player_team_at("joe_lolley", "2019-10-01")
    print(f"  player_team_at joe_lolley on 2019-10-01: {[str(t) for t in teams]}")

    print("\n── Sample Turtle output ──")
    with open(out) as f:
        for i, line in enumerate(f):
            if i > 35: print("  ..."); break
            print(f"  {line.rstrip()}")

    print("\n✓ all good!")
