"""
check_KG_for_pee_dodo.py
═════════════════════════════════════════════════════════════════════
EKG INSPECTION SCRIPT — for TA review
═════════════════════════════════════════════════════════════════════

WHAT IS AN EVENT KNOWLEDGE GRAPH (EKG)?
----------------------------------------
An Event Knowledge Graph is an RDF graph in which the central nodes
are **events** (a Shot, a Goal, a Foul, a Corner, etc.) and the edges
connect each event to the people, teams, matches, and other events
that give it context. Instead of storing match data in flat CSV rows,
the EKG lets you ask graph-shaped questions such as:

    "What happened just before this goal?"
    "Which fouls triggered a yellow card?"
    "List every event Adam Armstrong has been involved in this season."

These questions become SPARQL queries that traverse typed edges
between nodes.

WHAT THIS FILE INSPECTS
-----------------------
`data/kg_output/ekg.ttl` is a Turtle-serialised RDF graph produced by
the VLM + KG-builder pipeline. It contains every event detected from
the 7 EFL Championship matches played on 2019-10-01.

ONTOLOGY — key predicates you will see in the queries below
-----------------------------------------------------------
    ekg:isPerformedBy   event   →  player who performed it
    ekg:inMatch          event   →  match the event belongs to
    ekg:involvedTeam       event   →  team involved in the event
    ekg:precededBy       event   →  the immediately previous event
    ekg:playsFor         player  →  team they play for
    ekg:hasEventType      event   →  string: 'Shot' | 'Goal' | …
    ekg:hasMinute         event   →  integer minute within its half
    ekg:hasPeriodNumber         event   →  1 = first half, 2 = second
    ekg:hasTime           event   →  string e.g. '1st 18:05'
    ekg:hasDescription    event   →  short VLM-generated description
    ekg:triggered         foul    →  card event the foul caused

WHAT THIS SCRIPT DOES
---------------------
Loads ekg.ttl with `rdflib` and runs ten demonstrative SPARQL queries.
Each query is preceded by a short comment explaining what it asks and
why it is interesting. Results are printed as plain text tables.

HOW TO RUN
----------
    cd ~/work/s2616011/real-time_KG-with-vlm
    python check_KG_for_pee_dodo.py
"""

import sys
from collections import defaultdict
from pathlib    import Path

from rdflib import Graph


# ════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════
# Load the Turtle file once and reuse the Graph object for every query.
# The PREFIX block is defined ONCE here as a constant and prepended to
# every SPARQL string below — a recent bug in ekg_schema.py was caused
# by SPARQL strings being submitted without their PREFIX declarations,
# so this script makes the prefixes impossible to forget.

# Use an absolute path anchored on this file. Relative paths through
# rdflib's URI handling drop the last component of os.getcwd().
BASE_DIR = Path(__file__).resolve().parent
TTL_PATH = BASE_DIR / "data" / "kg_output" / "ekg.ttl"

PREFIXES = """
PREFIX ekg:  <http://soccerekg.org/ontology#>
PREFIX data: <http://soccerekg.org/data#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

print(f"Loading {TTL_PATH} ...")
if not TTL_PATH.exists():
    print(f"\nERROR: KG file not found: {TTL_PATH}")
    print("Build it first with:  python main.py --match \"Blackburn\"\n")
    sys.exit(1)
g = Graph()
g.parse(str(TTL_PATH), format="turtle")
print(f"Loaded {len(g):,} triples from {TTL_PATH}\n")


def banner(title: str) -> None:
    """Print a clearly delimited section header for each query."""
    print()
    print("─" * 72)
    print(f"  {title}")
    print("─" * 72)


# Seven action types the VLM is allowed to detect — used in Q3 and Q9.
ACTION_TYPES = ["Shot", "Goal", "Foul", "Free_Kick",
                "Corner", "Substitution", "Offside"]


# ════════════════════════════════════════════════════════════════════
# QUERY 1 — Overall counts
# ════════════════════════════════════════════════════════════════════
# Gives the TA a one-line picture of the whole KG: how many matches,
# how many distinct players were named, how many teams, how many
# events in total. The four counts are computed as independent
# sub-SELECTs and then joined in the outer SELECT so we get one row.

banner("QUERY 1: Overall counts — matches, players, teams, events")

q1 = PREFIXES + """
SELECT ?nMatches ?nPlayers ?nTeams ?nEvents WHERE {
    { SELECT (COUNT(DISTINCT ?m) AS ?nMatches) WHERE { ?m a ekg:Match  } }
    { SELECT (COUNT(DISTINCT ?p) AS ?nPlayers) WHERE { ?p a ekg:Player } }
    { SELECT (COUNT(DISTINCT ?t) AS ?nTeams)   WHERE { ?t a ekg:Team   } }
    { SELECT (COUNT(DISTINCT ?e) AS ?nEvents)
      WHERE { ?e ekg:hasEventType ?anyType } }
}
"""
for r in g.query(q1):
    print(f"  Matches : {int(r.nMatches):>6}")
    print(f"  Players : {int(r.nPlayers):>6}")
    print(f"  Teams   : {int(r.nTeams):>6}")
    print(f"  Events  : {int(r.nEvents):>6}")


# ════════════════════════════════════════════════════════════════════
# QUERY 2 — Events per match
# ════════════════════════════════════════════════════════════════════
# Helps the TA spot matches with abnormally low coverage (e.g. a clip
# range was truncated). GROUP BY collapses every event-of-match link
# into one count per match. The OPTIONAL on ?label is there because
# `rdfs:label` is added by the KG builder; without OPTIONAL a match
# without a label would be silently dropped from the results.

banner("QUERY 2: Events per match")

q2 = PREFIXES + """
SELECT ?label (COUNT(?e) AS ?n) WHERE {
    ?match a ekg:Match .
    OPTIONAL { ?match rdfs:label ?label }
    ?e ekg:inMatch ?match .
} GROUP BY ?match ?label
ORDER BY ?label
"""
print(f"  {'Match':<55} {'Events':>7}")
print(f"  {'─'*55} {'─'*7}")
for r in g.query(q2):
    name = (str(r.label) if r.label else "(no label)")[:55]
    print(f"  {name:<55} {int(r.n):>7}")


# ════════════════════════════════════════════════════════════════════
# QUERY 3 — Action type distribution
# ════════════════════════════════════════════════════════════════════
# Shows the relative frequency of each action category across the
# entire KG. A football-fan TA already knows the rough proportions
# (fouls + shots dominate); deviations from that intuition are a
# fast smell-test for VLM bias.

banner("QUERY 3: Event type distribution (all matches)")

q3 = PREFIXES + """
SELECT ?type (COUNT(?e) AS ?n) WHERE {
    ?e ekg:hasEventType ?type .
} GROUP BY ?type
ORDER BY DESC(?n)
"""
print(f"  {'Event Type':<16} {'Count':>7}")
print(f"  {'─'*16} {'─'*7}")
for r in g.query(q3):
    print(f"  {str(r.type):<16} {int(r.n):>7}")


# ════════════════════════════════════════════════════════════════════
# QUERY 4 — Top 10 most active players
# ════════════════════════════════════════════════════════════════════
# Sums every event each named player performed. We add a UNION over
# both directions of the "player–performed–event" edge because some
# versions of the schema declare the inverse property `performed`
# (player → event) instead of `isPerformedBy` (event → player), and
# we want the query to work either way without code changes.

banner("QUERY 4: Top 10 most active players")

q4 = PREFIXES + """
SELECT ?label ?jersey (COUNT(?e) AS ?n) WHERE {
    ?p a ekg:Player ;
       rdfs:label ?label .
    OPTIONAL { ?p ekg:hasJerseyNumber ?jersey }
    { ?e ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?e }
} GROUP BY ?p ?label ?jersey
ORDER BY DESC(?n)
LIMIT 10
"""
print(f"  {'Player':<28} {'Jersey':>7} {'Events':>7}")
print(f"  {'─'*28} {'─'*7} {'─'*7}")
for r in g.query(q4):
    jersey = str(r.jersey) if r.jersey is not None else "—"
    print(f"  {str(r.label)[:28]:<28} {jersey:>7} {int(r.n):>7}")


# ════════════════════════════════════════════════════════════════════
# QUERY 5 — All goals scored
# ════════════════════════════════════════════════════════════════════
# Lists every Goal event with its scorer, team, and match. This is
# the most direct way for the TA to spot-check the KG against a real
# match report: goals are rare and verifiable.

banner("QUERY 5: All goals scored — gametime, scorer, team, match")

q5 = PREFIXES + """
SELECT ?matchLabel ?gametime ?playerLabel ?teamLabel WHERE {
    ?e ekg:hasEventType  "Goal" ;
       ekg:hasTime       ?gametime ;
       ekg:inMatch      ?match .
    ?match rdfs:label ?matchLabel .
    OPTIONAL {
        { ?e ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?e }
        ?p rdfs:label ?playerLabel .
    }
    OPTIONAL {
        ?e ekg:involvedTeam ?t .
        ?t rdfs:label      ?teamLabel .
    }
} ORDER BY ?matchLabel ?gametime
"""
print(f"  {'Match':<40} {'Time':<11} {'Scorer':<22} {'Team':<18}")
print(f"  {'─'*40} {'─'*11} {'─'*22} {'─'*18}")
for r in g.query(q5):
    match  = str(r.matchLabel)[:40]
    time   = str(r.gametime)[:11]
    player = (str(r.playerLabel) if r.playerLabel else "unidentified")[:22]
    team   = (str(r.teamLabel)   if r.teamLabel   else "?")[:18]
    print(f"  {match:<40} {time:<11} {player:<22} {team:<18}")


# ════════════════════════════════════════════════════════════════════
# QUERY 6 — Sample event drill-down
# ════════════════════════════════════════════════════════════════════
# Picks the first Goal the SPARQL engine returns and prints EVERY
# triple where that event is the subject. The point is to make the
# data model concrete: instead of describing the predicates abstractly
# the TA sees them attached to a real instance.

banner("QUERY 6: Sample event drill-down — all triples of one goal")

# Step 1: find a single goal URI we can show off.
q6_pick = PREFIXES + """
SELECT ?e WHERE { ?e ekg:hasEventType "Goal" } LIMIT 1
"""
goal_uri = None
for r in g.query(q6_pick):
    goal_uri = str(r.e)
    break

if goal_uri is None:
    print("  (no Goal events found in the KG)")
else:
    print(f"  Chosen event URI: {goal_uri}\n")
    # Step 2: dump every (predicate, object) pair where this event is
    # the subject. Predicate URIs are abbreviated for readability.
    q6_dump = PREFIXES + f"""
    SELECT ?p ?o WHERE {{ <{goal_uri}> ?p ?o }}
    """
    print(f"  {'Predicate':<28} Object")
    print(f"  {'─'*28} {'─'*42}")
    for r in g.query(q6_dump):
        pred = str(r.p).split("#")[-1].split("/")[-1]
        obj  = str(r.o)
        if obj.startswith("http"):
            obj = obj.split("#")[-1].split("/")[-1]
        print(f"  {pred:<28} {obj[:60]}")


# ════════════════════════════════════════════════════════════════════
# QUERY 7 — Player history (cross-event)
# ════════════════════════════════════════════════════════════════════
# Demonstrates one of the EKG's selling points: a single player node
# is reused across matches, so we can list every event Adam Armstrong
# participated in regardless of which match it happened in. This
# would require multi-table joins in a relational schema; here it is
# a single graph pattern.

banner("QUERY 7: Player history — every event by Adam Armstrong")

q7 = PREFIXES + """
SELECT ?matchLabel ?gametime ?type WHERE {
    ?p rdfs:label "Adam Armstrong" .
    { ?e ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?e }
    ?e ekg:hasEventType ?type ;
       ekg:hasTime      ?gametime ;
       ekg:inMatch     ?match .
    ?match rdfs:label ?matchLabel .
} ORDER BY ?matchLabel ?gametime
"""
print(f"  {'Match':<45} {'Time':<11} {'Action':<14}")
print(f"  {'─'*45} {'─'*11} {'─'*14}")
hits = 0
for r in g.query(q7):
    print(f"  {str(r.matchLabel)[:45]:<45} "
          f"{str(r.gametime)[:11]:<11} "
          f"{str(r.type)[:14]:<14}")
    hits += 1
if hits == 0:
    print("  (no events found for Adam Armstrong — try a different name)")


# ════════════════════════════════════════════════════════════════════
# QUERY 8 — Event chain (precededBy)
# ════════════════════════════════════════════════════════════════════
# Picks any one match, finds its LAST event (one with no successor),
# and walks the precededBy edges backwards up to 10 hops. The output
# reads from most-recent at the top to oldest at the bottom, so the
# TA can see how the temporal chain is wired up.

banner("QUERY 8: Event chain — walk precededBy backwards (10 hops max)")

# Pick one match to demonstrate on — first one in the KG by label.
q8_pick = PREFIXES + """
SELECT ?match ?label WHERE {
    ?match a ekg:Match .
    OPTIONAL { ?match rdfs:label ?label }
} ORDER BY ?label LIMIT 1
"""
match_uri, match_label = None, None
for r in g.query(q8_pick):
    match_uri   = str(r.match)
    match_label = str(r.label) if r.label else match_uri
    break

if match_uri is None:
    print("  (no matches in KG)")
else:
    print(f"  Match : {match_label}\n")

    # Find the "last" event in that match: an event that nothing else
    # has as its precededBy predecessor. (i.e. nothing came after it.)
    q8_last = PREFIXES + f"""
    SELECT ?e WHERE {{
        ?e ekg:inMatch <{match_uri}> .
        FILTER NOT EXISTS {{ ?next ekg:precededBy ?e }}
    }} LIMIT 1
    """
    current = None
    for r in g.query(q8_last):
        current = str(r.e)
        break

    if current is None:
        print("  (no terminal event found — chain may be empty)")
    else:
        print(f"  {'Step':<5} {'Time':<11} {'Action':<14} {'URI tail':<28}")
        print(f"  {'─'*5} {'─'*11} {'─'*14} {'─'*28}")
        for step in range(10):
            # For each hop, fetch the current event's own details plus
            # the predecessor it points to. We then move "current" to
            # that predecessor and loop.
            q_step = PREFIXES + f"""
            SELECT ?type ?gametime ?prev WHERE {{
                <{current}> ekg:hasEventType ?type ;
                            ekg:hasTime      ?gametime .
                OPTIONAL {{ <{current}> ekg:precededBy ?prev }}
            }} LIMIT 1
            """
            row = None
            for r in g.query(q_step):
                row = r
                break
            if row is None:
                break
            tail = current.split("#")[-1].split("/")[-1][:28]
            print(f"  {step+1:<5} "
                  f"{str(row.gametime)[:11]:<11} "
                  f"{str(row.type)[:14]:<14} "
                  f"{tail:<28}")
            if row.prev is None:
                print("  (start of chain reached)")
                break
            current = str(row.prev)


# ════════════════════════════════════════════════════════════════════
# QUERY 9 — Team event breakdown (matrix)
# ════════════════════════════════════════════════════════════════════
# For every team, count how many events of each action type were
# linked to them via involvedTeam. The raw SPARQL returns one row per
# (team, type) pair; we pivot the rows into a small matrix in Python
# so the TA can read it as a single table.

banner("QUERY 9: Team event breakdown — rows = teams, columns = types")

q9 = PREFIXES + """
SELECT ?teamLabel ?type (COUNT(?e) AS ?n) WHERE {
    ?e ekg:involvedTeam  ?t ;
       ekg:hasEventType ?type .
    ?t rdfs:label ?teamLabel .
} GROUP BY ?teamLabel ?type
"""

# matrix[team][type] = count
matrix = defaultdict(lambda: defaultdict(int))
for r in g.query(q9):
    matrix[str(r.teamLabel)][str(r.type)] = int(r.n)

# Shorten the column labels so the matrix fits in a terminal.
SHORT = {
    "Shot"         : "Shot",
    "Goal"         : "Goal",
    "Foul"         : "Foul",
    "Free_Kick"    : "FK",
    "Corner"       : "Corn",
    "Substitution" : "Sub",
    "Offside"      : "Off",
}
header = "  " + f"{'Team':<26}" + "".join(f"{SHORT[a]:>6}" for a in ACTION_TYPES)
print(header)
print("  " + "─"*26 + "".join(" " + "─"*5 for _ in ACTION_TYPES))
for team in sorted(matrix.keys()):
    row = "".join(f"{matrix[team][a]:>6}" for a in ACTION_TYPES)
    print(f"  {team[:26]:<26}{row}")


# ════════════════════════════════════════════════════════════════════
# QUERY 10 — Cards triggered by fouls
# ════════════════════════════════════════════════════════════════════
# Cards are derived from ESPN data and linked back to the foul that
# caused them with the `triggered` predicate. The query is wrapped in
# OPTIONAL blocks because not every card has a triggering foul in the
# KG (yellow cards for dissent, for example, do not).

banner("QUERY 10: Cards and the fouls that triggered them")

q10 = PREFIXES + """
SELECT ?cardTime ?cardType ?playerLabel ?foulTime WHERE {
    ?card ekg:hasEventType ?cardType ;
          ekg:hasTime      ?cardTime .
    FILTER (?cardType IN ("YellowCard", "RedCard"))
    OPTIONAL {
        ?foul ekg:triggered ?card ;
              ekg:hasTime   ?foulTime .
    }
    OPTIONAL {
        { ?card ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?card }
        ?p rdfs:label ?playerLabel .
    }
} ORDER BY ?cardTime
"""
print(f"  {'Card time':<11} {'Card':<11} "
      f"{'Player':<24} {'Triggering foul':<15}")
print(f"  {'─'*11} {'─'*11} {'─'*24} {'─'*15}")
rows = list(g.query(q10))
if not rows:
    print("  (no YellowCard or RedCard events in the KG)")
for r in rows:
    ctime  = str(r.cardTime)[:11]
    ctype  = str(r.cardType)[:11]
    player = (str(r.playerLabel) if r.playerLabel else "unidentified")[:24]
    foul   = (str(r.foulTime)    if r.foulTime    else "—")[:15]
    print(f"  {ctime:<11} {ctype:<11} {player:<24} {foul:<15}")


# ════════════════════════════════════════════════════════════════════
# CLOSING
# ════════════════════════════════════════════════════════════════════
print()
print("─" * 72)
print("  ✓ Done — 10 queries executed against ekg.ttl")
print("─" * 72)
