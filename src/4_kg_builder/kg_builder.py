"""
kg_builder.py — Soccer EKG builder (RDF/OWL + TKG)
─────────────────────────────────────────────────────

Two entry points:

1. CSV MODE (for testing)
       python kg_builder.py --fast

2. REAL-TIME MODE (for the pipeline)
       from kg_builder import ingest_matched_event
       ingest_matched_event(matched_event, match_name, match_date, ekg, last_event)

Changes from previous version:
  - PLAYS_FOR uses standard RDF reification (rdf:Statement + rdf:subject /
    rdf:predicate / rdf:object) instead of custom ekg:subject / ekg:object
  - hasJersey split: hasJerseyNumber on Player, detectedJersey on Event (VLM)
  - hasTimeMin removed (redundant — hasTime string is the canonical value)
  - isMatched still written as a provenance annotation but not in the T-Box
  - Event nodes get a specific OWL subclass (GoalEvent, ShotEvent, …)
  - get_or_create_match uses regex for robust date extraction and adds
    hasHomeTeam / hasAwayTeam on the Match node
"""

import csv
import json
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from rdflib import Literal, RDF, RDFS, XSD

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Use resolve() so the path is absolute even when __file__ is relative
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_SCHEMA_FILE = _THIS_DIR / "ekg_schema.py"
if not _SCHEMA_FILE.exists():
    raise FileNotFoundError(
        f"ekg_schema.py missing from {_THIS_DIR}\n"
        f"Fix: git checkout HEAD -- src/4_kg_builder/ekg_schema.py"
    )

from ekg_schema import EKG_Graph, EKG, INST, EVENT_TYPE_CLASS, FOAF, SKOS, EVENT, WGS84, GEO


PITCH_ZONE_COORDS = {
    "penalty_area"   : (95.0, 34.0),
    "six_yard_box"   : (103.0, 34.0),
    "edge_of_box"    : (88.0, 34.0),
    "box"            : (95.0, 34.0),
    "left_wing"      : (75.0, 5.0),
    "right_wing"     : (75.0, 63.0),
    "left_flank"     : (70.0, 8.0),
    "right_flank"    : (70.0, 60.0),
    "centre_circle"  : (52.5, 34.0),
    "midfield"       : (52.5, 34.0),
    "own_half"       : (26.0, 34.0),
    "defensive_half" : (26.0, 34.0),
    "corner"         : (105.0, 5.0),
    "centre"         : (52.5, 34.0),
}


# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "blackburn_forest_2019-10-01.csv"
OUT_DIR  = DATA_DIR / "kg_output"
TTL_PATH    = OUT_DIR  / "ekg.ttl"
STREAM_PATH = OUT_DIR  / "events_stream.jsonl"

DEFAULT_MATCH_DATE = "2019-10-01"


def _append_stream(event_data: dict):
    """Append one event JSON line to the live visualization stream."""
    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data, default=str) + "\n")


def clear_stream():
    """Truncate the event stream. Call at the start of each pipeline run."""
    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    STREAM_PATH.write_text("")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")


def parse_time_min(time_str: str) -> float:
    t = re.sub(r"'", "", str(time_str).strip())
    if "+" in t:
        base, extra = t.split("+", 1)
        try: return float(base.strip()) + float(extra.strip())
        except: return 0.0
    try: return float(t.strip())
    except: return 0.0


def extract_assist(full_text: str) -> Optional[str]:
    m = re.search(r"Assisted by ([A-Z][a-zA-Z\s\-'\.]+?)(?:\.|with\b)", full_text or "")
    return m.group(1).strip() if m else None


def _extract_date(match_name: str) -> str:
    """Extract YYYY-MM-DD from a match name robustly using regex."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", match_name)
    return m.group(0) if m else match_name.split(" - ")[0].strip()


def _extract_teams(match_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (home_team, away_team) from 'DATE - Home - Away' format."""
    parts = match_name.split(" - ", 2)
    if len(parts) == 3:
        return parts[1].strip(), parts[2].strip()
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# A-BOX INSTANCE CREATORS
# ═══════════════════════════════════════════════════════════════════════════

def get_or_create_match(match_name: str, ekg: EKG_Graph) -> Tuple[str, str]:
    """
    Create a Match instance if not already present.
    Returns (match_id, match_date).
    Adds hasHomeTeam / hasAwayTeam when team names are parseable.
    """
    mid = normalize_id(match_name)
    date = _extract_date(match_name)

    if mid in ekg._seen_matches:
        return mid, date

    match_uri = ekg.match_uri(mid)
    ekg.g.add((match_uri, RDF.type,    EKG.Match))
    ekg.g.add((match_uri, RDFS.label,  Literal(match_name)))
    ekg.g.add((match_uri, EKG.hasDate, Literal(date)))
    venue_id  = get_or_create_venue("Ewood Park", 53.7286, -2.4895, ekg)
    venue_uri = INST[f"venue_{venue_id}"]
    ekg.g.add((match_uri, EVENT.place, venue_uri))

    home, away = _extract_teams(match_name)
    if home:
        home_id  = normalize_id(home)
        home_uri = ekg.team_uri(home_id)
        ekg.g.add((match_uri, EKG.hasHomeTeam, home_uri))
    if away:
        away_id  = normalize_id(away)
        away_uri = ekg.team_uri(away_id)
        ekg.g.add((match_uri, EKG.hasAwayTeam, away_uri))

    ekg._seen_matches.add(mid)
    return mid, date


def get_or_create_team(team_name: str, ekg: EKG_Graph) -> str:
    tid = normalize_id(team_name)
    if tid in ekg._seen_teams:
        return tid

    team_uri = ekg.team_uri(tid)
    ekg.g.add((team_uri, RDF.type,   EKG.Team))
    ekg.g.add((team_uri, RDFS.label, Literal(team_name)))
    ekg.g.add((team_uri, FOAF.name,  Literal(team_name)))

    ekg._seen_teams.add(tid)
    return tid


def get_or_create_venue(venue_name: str, lat: float, long_: float,
                         ekg: EKG_Graph) -> str:
    vid = normalize_id(venue_name)
    if vid not in getattr(ekg, "_seen_venues", set()):
        venue_uri = INST[f"venue_{vid}"]
        ekg.g.add((venue_uri, RDF.type,       EKG.Venue))
        ekg.g.add((venue_uri, RDFS.label,     Literal(venue_name)))
        ekg.g.add((venue_uri, WGS84.lat,      Literal(lat,   datatype=XSD.decimal)))
        ekg.g.add((venue_uri, WGS84["long"],  Literal(long_, datatype=XSD.decimal)))
        if not hasattr(ekg, "_seen_venues"):
            ekg._seen_venues = set()
        ekg._seen_venues.add(vid)
    return vid


def get_or_create_player(
    player_name : str,
    team_id     : Optional[str],
    ekg         : EKG_Graph,
    match_date  : str = DEFAULT_MATCH_DATE,
) -> Tuple[str, bool]:
    """
    Create a Player instance if not already present.
    PLAYS_FOR uses standard RDF reification (rdf:Statement).
    Returns (player_id, is_new).
    """
    pid    = normalize_id(player_name)
    is_new = pid not in ekg._seen_players

    if is_new:
        player_uri = ekg.player_uri(pid)
        ekg.g.add((player_uri, RDF.type,   EKG.Player))
        ekg.g.add((player_uri, RDFS.label, Literal(player_name)))
        ekg.g.add((player_uri, FOAF.name,  Literal(player_name)))

        if team_id:
            team_uri = ekg.team_uri(team_id)
            edge_uri = ekg.plays_for_uri(pid, team_id, match_date)
            # standard RDF reification: rdf:Statement with rdf:subject / rdf:predicate / rdf:object
            ekg.g.add((edge_uri, RDF.type,      RDF.Statement))
            ekg.g.add((edge_uri, RDF.subject,   player_uri))
            ekg.g.add((edge_uri, RDF.predicate, EKG.PLAYS_FOR))
            ekg.g.add((edge_uri, RDF.object,    team_uri))
            ekg.g.add((edge_uri, EKG.validFrom, Literal(match_date, datatype=XSD.date)))

        ekg._seen_players.add(pid)

    return pid, is_new


def add_participates_in(player_id: str, match_id: str, ekg: EKG_Graph):
    player_uri = ekg.player_uri(player_id)
    match_uri  = ekg.match_uri(match_id)
    if (player_uri, EKG.PARTICIPATED_IN, match_uri) not in ekg.g:
        ekg.g.add((player_uri, EKG.PARTICIPATED_IN, match_uri))


def prepopulate_roster(
    roster_lookup,
    match_name  : str,
    match_date  : str,
    ekg         : EKG_Graph,
) -> int:
    """
    Add ALL players from the roster to the KG at startup.
    Creates Player nodes + TKG PLAYS_FOR edges before any clips run.

    Returns total players added.
    """
    match_id, _ = get_or_create_match(match_name, ekg)
    total = 0

    for team_name in roster_lookup.get_teams():
        team_id = get_or_create_team(team_name, ekg)
        players = roster_lookup.get_all_players(team_name)

        for p in players:
            player_name = p["name"]
            jersey      = p["jersey"]
            pid         = normalize_id(player_name)

            if pid not in ekg._seen_players:
                player_uri = ekg.player_uri(pid)
                ekg.g.add((player_uri, RDF.type,            EKG.Player))
                ekg.g.add((player_uri, RDFS.label,          Literal(player_name)))
                ekg.g.add((player_uri, FOAF.name,           Literal(player_name)))
                # hasJerseyNumber — permanent squad number on the Player node
                ekg.g.add((player_uri, EKG.hasJerseyNumber, Literal(str(jersey))))

                # PLAYS_FOR: standard RDF reification
                team_uri = ekg.team_uri(team_id)
                edge_uri = ekg.plays_for_uri(pid, team_id, match_date)
                ekg.g.add((edge_uri, RDF.type,      RDF.Statement))
                ekg.g.add((edge_uri, RDF.subject,   player_uri))
                ekg.g.add((edge_uri, RDF.predicate, EKG.PLAYS_FOR))
                ekg.g.add((edge_uri, RDF.object,    team_uri))
                ekg.g.add((edge_uri, EKG.validFrom, Literal(match_date, datatype=XSD.date)))
                ekg.g.add((team_uri, FOAF.member,   player_uri))

                ekg._seen_players.add(pid)
                total += 1

        print(f"  [kg] pre-populated {len(players)} players for {team_name}")

    print(f"  [kg] roster pre-population done: {total} players, {ekg.triple_count()} triples")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# CORE: CREATE ONE EVENT NODE
# ═══════════════════════════════════════════════════════════════════════════

def _create_event_node(
    ekg         : EKG_Graph,
    match_id    : str,
    time_raw    : str,
    event_type  : str,
    player_id   : Optional[str],
    team_id     : Optional[str],
    full_text   : str = "",
    confidence  : Optional[float] = None,
    matched     : bool = True,
    last_event  : dict = None,
    description  : Optional[str] = None,
    jersey       : Optional[str] = None,
    team_color   : Optional[str] = None,
    shorts_color : Optional[str] = None,
    socks_color  : Optional[str] = None,
    kit_pattern  : Optional[str] = None,
    pitch_zone   : Optional[str] = None,
    body_part    : Optional[str] = None,
    outcome      : Optional[str] = None,
    foul_type    : Optional[str] = None,
    team_side    : Optional[str] = None,
    ball_visible : Optional[bool] = None,
) -> Tuple[str, list]:
    """
    Create one Event node + wire standard edges.
    - Uses specific OWL subclass (GoalEvent, ShotEvent, …) for rdf:type.
    - detectedJersey carries VLM-detected jersey number (separate from hasJerseyNumber on Player).
    - isMatched written as provenance annotation (not a T-Box property).
    Returns (event_id, edge_description_list).
    """
    ekg._event_count += 1
    event_id  = f"{ekg._event_count:04d}"
    event_uri = ekg.event_uri(event_id, match_id)

    # specific OWL class — enables reasoning over event subtypes
    specific_class = EVENT_TYPE_CLASS.get(event_type)
    if specific_class:
        ekg.g.add((event_uri, RDF.type, specific_class))
    # always also assert the mid-level class so existing queries on
    # ekg:ActionEvent / ekg:CardEvent continue to work without OWL reasoning
    mid_class = EKG.CardEvent if event_type in ("YellowCard", "RedCard") else EKG.ActionEvent
    ekg.g.add((event_uri, RDF.type, mid_class))

    ekg.g.add((event_uri, EKG.hasEventType, Literal(event_type)))
    ekg.g.add((event_uri, EKG.hasTime,      Literal(time_raw)))
    ekg.g.add((event_uri, EKG.isMatched,    Literal(matched, datatype=XSD.boolean)))

    # hasMinute (decimal) + hasPeriod (integer) — derived from gametime string
    # "1st 09:34" → period=1, minute=9.567   "2nd 07:30" → period=2, minute=7.5
    # NOTE: hasMinute is minute-within-half, not absolute match minute.
    #       For cross-half queries add (halftime_sec/60) when hasPeriod=2.
    try:
        half, t  = time_raw.strip().split(" ", 1)
        mm, ss   = t.strip().split(":")
        period   = 1 if half == "1st" else 2
        minute   = int(mm) + int(ss) / 60.0
        ekg.g.add((event_uri, EKG.hasMinute, Literal(round(minute, 3), datatype=XSD.decimal)))
        ekg.g.add((event_uri, EKG.hasPeriod, Literal(period,           datatype=XSD.integer)))
    except Exception:
        print(f"  [kg] WARNING: cannot parse hasMinute/hasPeriod from '{time_raw}'")

    if full_text:
        ekg.g.add((event_uri, EKG.hasFullText,    Literal(full_text)))
    if confidence is not None:
        ekg.g.add((event_uri, EKG.hasConfidence,  Literal(float(confidence))))
    if description:
        ekg.g.add((event_uri, EKG.hasDescription, Literal(description)))
    if jersey:
        # VLM-detected jersey on the event — distinct from hasJerseyNumber on Player
        ekg.g.add((event_uri, EKG.detectedJersey,   Literal(str(jersey))))
    if team_color:
        ekg.g.add((event_uri, EKG.hasDetectedColor,       Literal(str(team_color))))
    if shorts_color:
        ekg.g.add((event_uri, EKG.hasDetectedShortsColor, Literal(str(shorts_color))))
    if socks_color:
        ekg.g.add((event_uri, EKG.hasDetectedSocksColor,  Literal(str(socks_color))))
    if kit_pattern:
        ekg.g.add((event_uri, EKG.hasKitPattern,  Literal(str(kit_pattern))))
    if pitch_zone:
        zone_str = str(pitch_zone).lower().strip()
        ekg.g.add((event_uri, EKG.hasPitchZone, Literal(zone_str)))
        coords = PITCH_ZONE_COORDS.get(zone_str)
        if coords:
            geom_uri = INST[f"geom_{event_id}"]
            ekg.g.add((event_uri, GEO.hasGeometry, geom_uri))
            ekg.g.add((geom_uri,  RDF.type,         GEO.Point))
            ekg.g.add((geom_uri,  GEO.asWKT,
                Literal(f"POINT({coords[0]} {coords[1]})",
                        datatype=GEO.wktLiteral)))
    if body_part:
        ekg.g.add((event_uri, EKG.hasBodyPart,    Literal(str(body_part))))
    if outcome:
        ekg.g.add((event_uri, EKG.hasOutcome,     Literal(str(outcome))))
    if foul_type:
        ekg.g.add((event_uri, EKG.hasFoulType,    Literal(str(foul_type))))
    if team_side:
        ekg.g.add((event_uri, EKG.hasTeamSide,    Literal(str(team_side))))
    if ball_visible is not None:
        ekg.g.add((event_uri, EKG.hasBallVisible, Literal(ball_visible, datatype=XSD.boolean)))

    new_edges = []

    # IN_MATCH
    match_uri = ekg.match_uri(match_id)
    ekg.g.add((event_uri, EKG.IN_MATCH, match_uri))
    new_edges.append(f"event_{event_id} --[IN_MATCH]--> {match_id}")

    # PRECEDED_BY
    if last_event is not None and match_id in last_event:
        prev_uri = ekg.event_uri(last_event[match_id])
        ekg.g.add((event_uri, EKG.PRECEDED_BY, prev_uri))
        # assert the inverse (PRECEDES) explicitly — rdflib doesn't infer it
        ekg.g.add((prev_uri, EKG.PRECEDES, event_uri))
        new_edges.append(f"event_{event_id} --[PRECEDED_BY]--> event_{last_event[match_id]}")
    if last_event is not None:
        last_event[match_id] = event_id

    # INVOLVED_IN — event is subject, team is object
    if team_id:
        team_uri = ekg.team_uri(team_id)
        ekg.g.add((event_uri, EKG.INVOLVED_IN, team_uri))
        new_edges.append(f"event_{event_id} --[INVOLVED_IN]--> {team_id}")

    # PERFORMED + its inverse IS_PERFORMED_BY
    if player_id:
        player_uri = ekg.player_uri(player_id)
        ekg.g.add((player_uri, EKG.PERFORMED,       event_uri))
        ekg.g.add((event_uri,  EKG.IS_PERFORMED_BY, player_uri))
        new_edges.append(f"{player_id} --[PERFORMED]--> event_{event_id}")

    # Live visualization stream
    _append_stream({
        "event_id"   : event_id,
        "event_type" : event_type,
        "time_raw"   : time_raw,
        "player_id"  : player_id,
        "team_id"    : team_id,
        "match_id"   : match_id,
        "description": description,
        "pitch_zone" : pitch_zone,
        "body_part"  : body_part,
        "outcome"    : outcome,
        "foul_type"  : foul_type,
        "team_side"  : team_side,
        "confidence" : confidence,
        "timestamp"  : datetime.utcnow().isoformat(),
    })

    return event_id, new_edges


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — CSV MODE
# ═══════════════════════════════════════════════════════════════════════════

def ingest_event(row: dict, ekg: EKG_Graph, last_event: dict) -> dict:
    """Process one CSV row into the RDF graph."""
    match_name  = row["Match"]
    team_name   = row["Team"]   if row["Team"]   != "None" else None
    time_raw    = row["Time"]
    player_name = row["Player"] if row["Player"] != "None" else None
    action_type = row["Action_Type"]
    full_text   = row["Full_Text"]
    yellow      = row["Yellow_Card"] == "1"
    red         = row["Red_Card"]    == "1"

    new_edges     = []
    is_new_player = False

    match_id, match_date = get_or_create_match(match_name, ekg)
    team_id  = get_or_create_team(team_name, ekg) if team_name else None

    player_id = None
    if player_name:
        player_id, is_new_player = get_or_create_player(
            player_name, team_id, ekg, match_date)
        if is_new_player and team_id:
            new_edges.append(f"{player_name} --[PLAYS_FOR (TKG:{match_date})]--> {team_name}")
        add_participates_in(player_id, match_id, ekg)

    main_event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type = action_type,
        player_id  = player_id,
        team_id    = team_id,
        full_text  = full_text,
        matched    = True,
        last_event = last_event,
    )
    new_edges.extend(edges)

    assist_name = extract_assist(full_text)
    if assist_name and assist_name != player_name:
        assist_pid, assist_new = get_or_create_player(
            assist_name, team_id, ekg, match_date)
        ekg.g.add((ekg.event_uri(main_event_id), EKG.ASSISTED_BY, ekg.player_uri(assist_pid)))
        new_edges.append(f"event_{main_event_id} --[ASSISTED_BY]--> {assist_name}")
        if assist_new:
            is_new_player = True

    card_type = "RedCard" if red else ("YellowCard" if yellow else None)
    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            full_text  = f"{card_type} following {action_type}",
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)
        ekg.g.add((ekg.event_uri(main_event_id), EKG.TRIGGERED, ekg.event_uri(card_event_id)))
        new_edges.append(f"event_{main_event_id} --[TRIGGERED]--> event_{card_event_id} ({card_type})")

    return {
        "time": time_raw, "player": player_name,
        "action_type": action_type, "team": team_name,
        "is_new_player": is_new_player,
        "yellow": yellow, "red": red, "card": card_type,
        "new_edges": new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — REAL-TIME MODE
# ═══════════════════════════════════════════════════════════════════════════

def ingest_matched_event(
    matched,
    match_name   : str,
    match_date   : str,
    ekg          : EKG_Graph,
    last_event   : dict,
    description  : Optional[str] = None,
    jersey       : Optional[str] = None,
    team_color   : Optional[str] = None,
    shorts_color : Optional[str] = None,
    socks_color  : Optional[str] = None,
    kit_pattern  : Optional[str] = None,
    pitch_zone   : Optional[str] = None,
    body_part    : Optional[str] = None,
    outcome      : Optional[str] = None,
    foul_type    : Optional[str] = None,
    team_side    : Optional[str] = None,
    ball_visible : Optional[bool] = None,
) -> dict:
    """
    Ingest one MatchedEvent from align.py into the RDF graph.
    Optionally accepts description + jersey from Qwen2-VL.
    """
    action_type = matched.action
    time_raw    = matched.gametime or f"{matched.video_time/60:.1f}'"
    full_text   = matched.espn_text or ""
    confidence  = matched.confidence

    new_edges     = []
    is_new_player = False

    match_id, _ = get_or_create_match(match_name, ekg)

    if matched.matched and matched.player:
        team_id = get_or_create_team(matched.team, ekg) if matched.team else None
        player_id, is_new_player = get_or_create_player(
            matched.player, team_id, ekg, match_date)
        if is_new_player and team_id:
            new_edges.append(f"{matched.player} --[PLAYS_FOR (TKG:{match_date})]--> {matched.team}")
        add_participates_in(player_id, match_id, ekg)
    else:
        player_id = None
        team_id   = get_or_create_team(matched.team, ekg) if matched.team else None

    if team_id is None:
        print(f"  [kg] WARNING: team_id=None for {matched.action} at {matched.gametime} "
              f"(matched.team={matched.team!r}) — INVOLVED_IN will be skipped")

    event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type   = action_type,
        player_id    = player_id,
        team_id      = team_id,
        full_text    = full_text,
        confidence   = confidence,
        matched      = matched.matched,
        last_event   = last_event,
        description  = description,
        jersey       = jersey,
        team_color   = team_color,
        shorts_color = shorts_color,
        socks_color  = socks_color,
        kit_pattern  = kit_pattern,
        pitch_zone   = pitch_zone,
        body_part    = body_part,
        outcome      = outcome,
        foul_type    = foul_type,
        team_side    = team_side,
        ball_visible = ball_visible,
    )
    new_edges.extend(edges)

    card_type = None
    if matched.matched and full_text:
        if re.search(r"\bred card\b", full_text, re.IGNORECASE):
            card_type = "RedCard"
        elif re.search(r"\byellow card\b|booked", full_text, re.IGNORECASE):
            card_type = "YellowCard"

    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)
        ekg.g.add((ekg.event_uri(event_id), EKG.TRIGGERED, ekg.event_uri(card_event_id)))
        new_edges.append(f"event_{event_id} --[TRIGGERED]--> event_{card_event_id} ({card_type})")

    return {
        "time": time_raw,
        "player": matched.player if matched.matched else "UNKNOWN",
        "action_type": action_type, "team": matched.team,
        "is_new_player": is_new_player,
        "matched": matched.matched, "card": card_type,
        "new_edges": new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

ACTION_ICONS = {
    "Goal": "GOAL", "Shot": "shot", "Foul": "foul",
    "Corner": "corner", "Offside": "offside",
    "FreeKick": "free kick", "Substitution": "sub",
    "YellowCard": "🟨", "RedCard": "🟥", "Penalty": "penalty",
}

def print_event(res: dict, ekg: EKG_Graph):
    icon   = ACTION_ICONS.get(res["action_type"], res["action_type"])
    player = res.get("player") or "(no player)"
    team   = res.get("team")   or "—"
    tag    = " << NEW PLAYER >>" if res.get("is_new_player") else ""
    card   = f" [{res.get('card')}]" if res.get("card") else ""
    print(f"\n[{res['time']:>6}]  {icon:<12}  {player}{card}{tag}")
    print(f"          Team: {team}")
    for edge in res["new_edges"]:
        print(f"          + {edge}")
    print(f"          KG -> {ekg.stats()}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV MODE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def stream_simulation(csv_path: Path, scale: float = 1.0, fast: bool = False):
    with open(csv_path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f),
                      key=lambda r: parse_time_min(r.get("Time", "0")))

    ekg        = EKG_Graph()
    last_event = {}
    prev_min   = None

    match_name = rows[0]["Match"] if rows else "Unknown"
    mode_label = "FAST" if fast else f"scale={scale}s/min"

    print(f"\n{'='*62}")
    print(f"  MATCH : {match_name}")
    print(f"  EVENTS: {len(rows)}  |  MODE: {mode_label}")
    print(f"  T-Box : {len(ekg.g)} triples at start")
    print(f"{'='*62}")

    for i, row in enumerate(rows):
        curr_min = parse_time_min(row.get("Time", "0"))
        if not fast and prev_min is not None:
            wait = max(0.1, min(5.0, (curr_min - prev_min) * scale))
            time.sleep(wait)
        prev_min = curr_min
        result   = ingest_event(row, ekg, last_event)
        print_event(result, ekg)
        if (i + 1) % 20 == 0:
            ekg.save(TTL_PATH)

    ekg.save(TTL_PATH)
    yellow = ekg.events_by_type("YellowCard")
    red    = ekg.events_by_type("RedCard")

    print(f"\n{'='*62}")
    print(f"  MATCH ENDED")
    print(f"  Final EKG    : {ekg.stats()}")
    print(f"  Yellow cards : {len(yellow)}")
    print(f"  Red cards    : {len(red)}")
    print(f"  Saved to     : {TTL_PATH}")
    print(f"{'='*62}\n")
    return ekg


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()
    stream_simulation(CSV_PATH, scale=args.scale, fast=args.fast)
