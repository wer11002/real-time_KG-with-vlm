"""
ekg_schema.py — RDF/OWL Schema container for Soccer Event Knowledge Graph
─────────────────────────────────────────────────────────────────────────
The authoritative T-Box now lives in ekg_tbox.ttl at the project root.
This module parses that file into the graph at startup and provides:

  - URI helpers (player_uri, team_uri, match_uri, event_uri, …)
  - ACTION_TO_CLASS dict — leaf-class lookup used by kg_builder.py
  - SPARQL helpers used by kg_builder.py self-tests

Every A-Box instance written by kg_builder.py is single-typed with the
leaf class (e.g. ekg:Shot) only — no foreign-vocab multi-typing.

Quick test:
    python ekg_schema.py
"""

import logging
import re
from pathlib import Path
from rdflib  import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


# ═══════════════════════════════════════════════════════════════════════════
# NAMESPACES
# ═══════════════════════════════════════════════════════════════════════════

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
TBOX_PATH = BASE_DIR / "ekg_tbox.ttl"


# ═══════════════════════════════════════════════════════════════════════════
# ACTION → LEAF CLASS  (single source of truth shared with kg_builder)
# ═══════════════════════════════════════════════════════════════════════════

ACTION_TO_CLASS = {
    "Shot"         : EKG.Shot,
    "Goal"         : EKG.Goal,
    "Foul"         : EKG.Foul,
    "FreeKick"     : EKG.FreeKick,
    "Free_Kick"    : EKG.FreeKick,
    "Corner"       : EKG.Corner,
    "Substitution" : EKG.Substitution,
    "Offside"      : EKG.OffsideCalled,
    "YellowCard"   : EKG.YellowCard,
    "RedCard"      : EKG.RedCard,
}


# ═══════════════════════════════════════════════════════════════════════════
# EKG_Graph — A-Box container with T-Box pre-loaded
# ═══════════════════════════════════════════════════════════════════════════

class EKG_Graph:
    """rdflib Graph + T-Box pre-loaded; grows A-Box in real-time."""

    def __init__(self):
        if not TBOX_PATH.exists():
            raise FileNotFoundError(
                f"T-Box not found: {TBOX_PATH}\n"
                f"Expected ekg_tbox.ttl at the project root."
            )
        self.g = Graph()
        self.g.bind("ekg",  EKG)
        self.g.bind("data", INST)
        self.g.bind("owl",  OWL)
        self.g.bind("rdf",  RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("xsd",  XSD)
        self.g.parse(str(TBOX_PATH), format="turtle")

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
    def event_uri(event_id: str, match_id: str = "") -> URIRef:
        prefix = f"{match_id}_" if match_id else ""
        return INST[f"event_{prefix}{event_id}"]

    @staticmethod
    def plays_for_uri(player_id: str, team_id: str, date: str) -> URIRef:
        # DEPRECATED (fix 035): playsFor is now a direct triple.
        # Helper kept for backward compat — do not use in new code.
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

    # ── SPARQL helpers (use new T-Box property names) ──────────────────────

    def events_by_type(self, event_type: str) -> list:
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT ?e WHERE {
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?etype)
        }
        """
        return [row[0] for row in self.g.query(
            q, initBindings={"etype": Literal(event_type)})]

    def count_cards(self, player_id: str, color: str = "YellowCard") -> int:
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT (COUNT(?e) AS ?c) WHERE {
            ?p ekg:performed ?e .
            ?e a ekg:Card ;
               ekg:hasEventType ?t .
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

    def query_player_history(self, player_id: str) -> list[dict]:
        """All past events for a player, ordered by half then minute.
        Uses ekg:hasPeriodNumber (data property) — the new T-Box reserves
        ekg:hasPeriod as the object property Match → Period."""
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT ?eventType ?minute ?period ?pitchZone ?bodyPart ?outcome WHERE {
            data:player_%s ekg:performed ?e .
            ?e ekg:hasEventType     ?eventType ;
               ekg:hasMinute        ?minute ;
               ekg:hasPeriodNumber  ?period .
            OPTIONAL { ?e ekg:hasPitchZone ?pitchZone }
            OPTIONAL { ?e ekg:hasBodyPart  ?bodyPart  }
            OPTIONAL { ?e ekg:hasOutcome   ?outcome   }
        } ORDER BY ?period ?minute
        """ % player_id
        rows = []
        for r in self.g.query(q):
            rows.append({
                "type"      : str(r.eventType),
                "minute"    : float(r.minute),
                "period"    : int(r.period),
                "pitch_zone": str(r.pitchZone) if r.pitchZone else None,
                "body_part" : str(r.bodyPart)  if r.bodyPart  else None,
                "outcome"   : str(r.outcome)   if r.outcome   else None,
            })
        return rows

    def query_match_summary(self, match_id: str) -> dict:
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT ?team ?eventType (COUNT(?e) AS ?n) WHERE {
            ?e ekg:inMatch        data:match_%s ;
               ekg:hasEventType   ?eventType ;
               ekg:involvedTeam   ?team .
        } GROUP BY ?team ?eventType
        """ % match_id
        summary = {}
        for r in self.g.query(q):
            team  = str(r.team).split("team_")[-1]
            etype = str(r.eventType)
            summary.setdefault(team, {})[etype] = int(r.n)
        return summary

    def events_for_player(self, player_id: str) -> list:
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT ?e WHERE { ?p ekg:performed ?e . }
        """
        return [row[0] for row in self.g.query(
            q, initBindings={"p": self.player_uri(player_id)})]

    def player_team_at(self, player_id: str, date: str) -> list:
        """Which team does a player play for? Direct playsFor triple (fix 035)."""
        q = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX data: <http://soccerekg.org/data#>
        SELECT ?team WHERE {
            ?p ekg:playsFor ?team .
        }
        """
        return [row[0] for row in self.g.query(q, initBindings={
            "p": self.player_uri(player_id),
        })]

    # ── save / load ────────────────────────────────────────────────────────

    def save(self, out_path: Path, format: str = "turtle"):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # RDFS forward-chaining used to be applied here; with the new
        # leaf-only typing we want the rdf:type set to stay at size 1,
        # so reasoning is left to the consumer.
        self.g.serialize(destination=str(out_path), format=format)

    def load(self, path: str):
        """Merge an existing TTL into the graph (checkpoint resume)."""
        self.g.parse(path, format="turtle")


# ═══════════════════════════════════════════════════════════════════════════
# TYPED LITERAL HELPERS  (shared by kg_builder.py and repair_literal_types.py)
# ═══════════════════════════════════════════════════════════════════════════

_log      = logging.getLogger("ekg.typing")
_DATE_RE  = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# Predicate local-name → expected XSD datatype, mirrored from ekg_tbox.ttl.
# Any predicate NOT listed here stays as a plain xsd:string Literal.
PROP_TYPES: dict = {
    # xsd:integer
    "hasJerseyNumber"     : XSD.integer,
    "detectedJersey"      : XSD.integer,
    "hasAge"              : XSD.integer,
    "hasAttendance"       : XSD.integer,
    "hasFinalScoreHome"   : XSD.integer,
    "hasFinalScoreAway"   : XSD.integer,
    "hasHalfTimeScoreHome": XSD.integer,
    "hasHalfTimeScoreAway": XSD.integer,
    "hasFoundedYear"      : XSD.integer,
    "hasPeriodNumber"     : XSD.integer,
    "hasSecond"           : XSD.integer,
    "hasStartTime"        : XSD.integer,
    "hasEndTime"          : XSD.integer,
    "hasAddedTime"        : XSD.integer,
    # xsd:decimal
    "hasMinute"           : XSD.decimal,
    "hasHeight"           : XSD.decimal,
    "hasWeight"           : XSD.decimal,
    "hasMarketValue"      : XSD.decimal,
    "hasPossessionHome"   : XSD.decimal,
    "hasPossessionAway"   : XSD.decimal,
    "hasShotPower"        : XSD.decimal,
    "hasShotAngle"        : XSD.decimal,
    "hasExpectedGoals"    : XSD.decimal,
    "hasPassDistance"     : XSD.decimal,
    "hasXCoord"           : XSD.decimal,
    "hasYCoord"           : XSD.decimal,
    # xsd:boolean
    "isMatched"           : XSD.boolean,
    "hasBallVisible"      : XSD.boolean,
    "hasOnTarget"         : XSD.boolean,
    "hasPassSuccess"      : XSD.boolean,
    # xsd:dateTime (HermiT excludes xsd:date from its OWL 2 datatype map)
    "hasDate"             : XSD.dateTime,
    "validFrom"           : XSD.dateTime,
    "validUntil"          : XSD.dateTime,
}


def typed_literal(prop_local_name: str, value, context: str = ""):
    """
    Return a correctly xsd-typed Literal for the given predicate.

    - Looks up expected datatype from PROP_TYPES.
    - Attempts coercion. On success: returns typed Literal.
    - On failure: logs a WARNING and returns None — caller MUST skip the triple.
    - If prop is NOT in PROP_TYPES: returns plain Literal(str(value)).

    context is included in warning messages for debugging.
    """
    target = PROP_TYPES.get(prop_local_name)

    if target is None:
        return Literal(str(value))

    # Pass through already-correct Literals
    if isinstance(value, Literal) and value.datatype == target:
        return value

    # Fast path for native Python booleans (str(True) == "True" which parses fine,
    # but this avoids the string round-trip)
    if isinstance(value, bool) and target == XSD.boolean:
        return Literal(value, datatype=XSD.boolean)

    s = str(value).strip()

    try:
        if target == XSD.integer:
            return Literal(int(float(s)), datatype=XSD.integer)
        if target == XSD.decimal:
            return Literal(float(s), datatype=XSD.decimal)
        if target == XSD.boolean:
            lower = s.lower()
            if lower in ("true", "1", "yes", "t"):
                return Literal(True,  datatype=XSD.boolean)
            if lower in ("false", "0", "no", "f"):
                return Literal(False, datatype=XSD.boolean)
            raise ValueError(f"unrecognised boolean {s!r}")
        if target == XSD.dateTime:
            # Accept "2019-10-01" or full ISO dateTime. Coerce plain dates to
            # midnight UTC so HermiT can reason over them (xsd:date unsupported).
            if "T" not in s:
                if not _DATE_RE.match(s):
                    raise ValueError(f"expected YYYY-MM-DD or ISO dateTime, got {s!r}")
                s = f"{s}T00:00:00Z"
            return Literal(s, datatype=XSD.dateTime)
    except (ValueError, TypeError) as exc:
        _log.warning(
            "skip triple — could not coerce %s=%r to %s: %s [%s]",
            prop_local_name, value, target, exc, context,
        )
        return None

    return None  # safety net


def add_typed_triple(g, subject, predicate, raw_value, context: str = "") -> bool:
    """
    Convenience wrapper: look up type, coerce, add to graph.
    Returns True if triple was written, False if coercion failed (triple skipped).
    """
    prop_local = str(predicate).split("#")[-1]
    lit = typed_literal(prop_local, raw_value, context)
    if lit is None:
        return False
    g.add((subject, predicate, lit))
    return True


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("─── ekg_schema.py self-test ───")
    ekg = EKG_Graph()
    print(f"  T-Box loaded: {len(ekg.g):,} triples from {TBOX_PATH.name}")

    event  = ekg.event_uri("0001", "test_match")
    player = ekg.player_uri("test_player")
    team   = ekg.team_uri("test_team")

    ekg.g.add((event, RDF.type,          ACTION_TO_CLASS["Shot"]))
    ekg.g.add((event, EKG.hasEventType,  Literal("Shot")))
    ekg.g.add((event, EKG.hasTime,       Literal("1st 09:34")))
    ekg.g.add((player, RDF.type,         EKG.Player))
    ekg.g.add((team,   RDF.type,         EKG.Team))

    types = list(ekg.g.objects(event, RDF.type))
    print(f"  Sample event type-count: {len(types)} (should be 1)")
    print(f"  Sample event types: {[str(t).split('#')[-1] for t in types]}")
    print(f"  Final triple count: {len(ekg.g):,}")
    print("✓ self-test passed")
