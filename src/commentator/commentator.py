"""
commentator.py — Real-time football commentary engine
──────────────────────────────────────────────────────
Runs in a background daemon thread.  main.py feeds MatchedEvent objects
into event_queue after each event is ingested into the KG.  This module
queries ekg.ttl via SPARQL and calls a local LLM at localhost:8001 to
produce live commentary.

Usage (wired from main.py):
    from commentator import event_queue, start_commentator
    start_commentator("data/kg_output/ekg.ttl")
    ...
    event_queue.put_nowait(matched_event)
"""

import json
import queue
import threading
import traceback
from pathlib import Path

import requests
from rdflib import Graph, Namespace, RDF, RDFS

# ── namespaces ─────────────────────────────────────────────────────────────

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")

# ── shared queue (imported by main.py) ────────────────────────────────────

event_queue: queue.Queue = queue.Queue()

# ── commentary output ──────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
LOG_PATH        = BASE_DIR / "data" / "commentator_output" / "commentary_log.txt"
LLM_URL         = "http://localhost:8001/v1/chat/completions"
LLM_MODEL       = "commentator"
MAX_TOOL_ROUNDS = 4

# ── graph loader (no cache — each call reloads so it sees fresh triples) ──

def _load(ttl_path: str) -> Graph:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


# ═══════════════════════════════════════════════════════════════════════════
# SPARQL TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def get_player_history(ttl_path: str, player_uri: str) -> list[dict]:
    """All prior events by this player, ordered by period then minute."""
    g = _load(ttl_path)
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    SELECT ?e ?type ?hasTime ?minute ?period ?outcome ?teamSide WHERE {
        ?player ekg:PERFORMED ?e .
        ?e      a              ?type ;
                ekg:hasTime    ?hasTime ;
                ekg:hasMinute  ?minute ;
                ekg:hasPeriod  ?period .
        OPTIONAL { ?e ekg:hasOutcome  ?outcome  }
        OPTIONAL { ?e ekg:hasTeamSide ?teamSide }
        FILTER (?player = <%s>)
        FILTER (?type != <http://www.w3.org/2002/07/owl#NamedIndividual>)
    }
    ORDER BY ?period ?minute
    """ % player_uri
    rows = []
    for r in g.query(q):
        type_name = str(r.type).split("#")[-1]
        if type_name in ("ActionEvent", "CardEvent"):
            continue
        rows.append({
            "event"    : str(r.e),
            "type"     : type_name,
            "time"     : str(r.hasTime),
            "minute"   : float(r.minute),
            "period"   : int(r.period),
            "outcome"  : str(r.outcome)  if r.outcome  else None,
            "team_side": str(r.teamSide) if r.teamSide else None,
        })
    return rows


def get_event_chain(ttl_path: str, event_uri: str, depth: int = 5) -> list[dict]:
    """Walk PRECEDED_BY backwards up to `depth` steps from event_uri."""
    g = _load(ttl_path)
    chain = []
    cur   = event_uri
    for _ in range(depth):
        q = """
        PREFIX ekg: <http://soccerekg.org/ontology#>
        SELECT ?prev ?type ?hasTime ?minute ?period WHERE {
            <%s> ekg:PRECEDED_BY ?prev .
            ?prev a ?type ;
                  ekg:hasTime   ?hasTime ;
                  ekg:hasMinute ?minute ;
                  ekg:hasPeriod ?period .
            FILTER (?type != <http://www.w3.org/2002/07/owl#NamedIndividual>)
        }
        LIMIT 1
        """ % cur
        results = list(g.query(q))
        if not results:
            break
        r = results[0]
        type_name = str(r.type).split("#")[-1]
        if type_name in ("ActionEvent", "CardEvent"):
            # pick a more specific type if possible
            for t in g.objects(r.prev, RDF.type):
                tn = str(t).split("#")[-1]
                if tn not in ("ActionEvent", "CardEvent",
                               "NamedIndividual", "Class"):
                    type_name = tn
                    break
        chain.append({
            "event" : str(r.prev),
            "type"  : type_name,
            "time"  : str(r.hasTime),
            "minute": float(r.minute),
            "period": int(r.period),
        })
        cur = str(r.prev)
    return chain


def get_triggered_card(ttl_path: str, event_uri: str) -> list[dict]:
    """Check whether this foul event TRIGGERED a card."""
    g = _load(ttl_path)
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    SELECT ?card ?type ?hasTime ?player WHERE {
        <%s> ekg:TRIGGERED ?card .
        ?card a ?type ;
              ekg:hasTime ?hasTime .
        OPTIONAL { ?player ekg:PERFORMED ?card }
        FILTER (?type != <http://www.w3.org/2002/07/owl#NamedIndividual>)
    }
    """ % event_uri
    rows = []
    for r in g.query(q):
        type_name = str(r.type).split("#")[-1]
        if type_name in ("ActionEvent", "CardEvent"):
            continue
        player_label = None
        if r.player:
            lbl = list(g.objects(r.player, RDFS.label))
            player_label = str(lbl[0]) if lbl else str(r.player).split("#")[-1]
        rows.append({
            "card"  : str(r.card),
            "type"  : type_name,
            "time"  : str(r.hasTime),
            "player": player_label,
        })
    return rows


def get_match_state(ttl_path: str, match_uri: str) -> dict:
    """Current score (goal count), half, and team names."""
    g = _load(ttl_path)

    # home / away team names
    home_team = away_team = None
    for ht in g.objects(INST[match_uri.split("#")[-1].split("/")[-1]],
                         EKG.hasHomeTeam):
        lbl = list(g.objects(ht, RDFS.label))
        home_team = str(lbl[0]) if lbl else str(ht).split("/")[-1]
    for at in g.objects(INST[match_uri.split("#")[-1].split("/")[-1]],
                         EKG.hasAwayTeam):
        lbl = list(g.objects(at, RDFS.label))
        away_team = str(lbl[0]) if lbl else str(at).split("/")[-1]

    # count goals per team using INVOLVED_IN
    q_goals = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    SELECT ?team (COUNT(?e) AS ?n) WHERE {
        ?e a ekg:GoalEvent ;
           ekg:IN_MATCH <%s> ;
           ekg:INVOLVED_IN ?team .
    } GROUP BY ?team
    """ % match_uri
    home_goals = away_goals = 0
    for r in g.query(q_goals):
        lbl = list(g.objects(r.team, RDFS.label))
        tname = str(lbl[0]) if lbl else ""
        n = int(r.n)
        if home_team and home_team.lower() in tname.lower():
            home_goals = n
        elif away_team and away_team.lower() in tname.lower():
            away_goals = n

    # most recent goal full_text for score string
    q_ft = """
    PREFIX ekg: <http://soccerekg.org/ontology#>
    SELECT ?ft ?period ?minute WHERE {
        ?e a ekg:GoalEvent ;
           ekg:IN_MATCH <%s> ;
           ekg:hasPeriod ?period ;
           ekg:hasMinute ?minute .
        OPTIONAL { ?e ekg:hasFullText ?ft }
    }
    ORDER BY DESC(?period) DESC(?minute)
    LIMIT 1
    """ % match_uri
    last_full_text = None
    for r in g.query(q_ft):
        last_full_text = str(r.ft) if r.ft else None

    return {
        "home_team"      : home_team,
        "away_team"      : away_team,
        "home_goals"     : home_goals,
        "away_goals"     : away_goals,
        "score_full_text": last_full_text,
    }


def get_team_events(ttl_path: str, team_uri: str) -> dict:
    """Count of each event type INVOLVED_IN this team."""
    g = _load(ttl_path)
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    SELECT ?type (COUNT(?e) AS ?n) WHERE {
        ?e ekg:INVOLVED_IN <%s> ;
           a               ?type .
    } GROUP BY ?type
    """ % team_uri
    counts = {}
    for r in g.query(q):
        type_name = str(r.type).split("#")[-1]
        if type_name in ("ActionEvent", "CardEvent", "NamedIndividual", "Class"):
            continue
        counts[type_name] = int(r.n)
    return counts


# ── dispatch table ─────────────────────────────────────────────────────────

TOOL_FN = {
    "get_player_history"  : get_player_history,
    "get_event_chain"     : get_event_chain,
    "get_triggered_card"  : get_triggered_card,
    "get_match_state"     : get_match_state,
    "get_team_events"     : get_team_events,
}

# ── OpenAI tool schemas ────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name"       : "get_player_history",
            "description": "All events performed by a player, ordered by time.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "ttl_path"  : {"type": "string", "description": "Path to ekg.ttl"},
                    "player_uri": {"type": "string", "description": "Full URI of the player node"},
                },
                "required": ["ttl_path", "player_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_event_chain",
            "description": "Walk PRECEDED_BY backwards from an event to show what led up to it.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "ttl_path" : {"type": "string"},
                    "event_uri": {"type": "string", "description": "Full URI of the current event"},
                    "depth"    : {"type": "integer", "default": 5},
                },
                "required": ["ttl_path", "event_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_triggered_card",
            "description": "Check if a foul event triggered a yellow or red card.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "ttl_path" : {"type": "string"},
                    "event_uri": {"type": "string"},
                },
                "required": ["ttl_path", "event_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_match_state",
            "description": "Current score, goal count, and team names for the match.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "ttl_path" : {"type": "string"},
                    "match_uri": {"type": "string", "description": "Full URI of the match node"},
                },
                "required": ["ttl_path", "match_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_team_events",
            "description": "Count of each event type linked to a team via INVOLVED_IN.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "ttl_path": {"type": "string"},
                    "team_uri": {"type": "string", "description": "Full URI of the team node"},
                },
                "required": ["ttl_path", "team_uri"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT HINTS
# ═══════════════════════════════════════════════════════════════════════════

def _suggested_tools(action: str) -> str:
    suggestions = {
        "Goal"     : "get_player_history, get_event_chain, get_match_state",
        "Shot"     : "get_player_history, get_team_events, get_event_chain",
        "Foul"     : "get_player_history, get_triggered_card, get_team_events",
        "Free_Kick": "get_event_chain, get_team_events",
        "Corner"   : "get_event_chain, get_match_state",
    }
    return suggestions.get(action, "get_match_state")


# ═══════════════════════════════════════════════════════════════════════════
# LLM AGENT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Always respond in English only. Never use any other language.

You are a live football commentator with access to a
knowledge graph (KG) of the match. You produce punchy, accurate commentary.

CRITICAL RULES — follow these strictly:
1. Only state facts that came from tool call results. Never invent stats,
   scores, or history. If a tool returns empty, describe only what you know.
2. Never say a player scored their "Nth goal" or "committed their Nth foul"
   unless get_player_history confirms it by counting previous events of the
   same type. If the player is unknown, describe the moment only.
3. NEVER say a player 'scored' or 'it's a goal' unless the current event
   type is GoalEvent. hasOutcome='goal' on a ShotEvent only means that shot
   went in — it does not make the shot a goal event. Describe Shots as
   attempts, efforts, or strikes — never as goals.
4. 1-3 sentences per event. Punchy, not an essay.
5. Tone: excited but grounded, like a real TV commentator.
6. Reference history naturally when tools confirm it:
   "his second goal tonight", "the man who fouled him earlier",
   "under pressure again from this player".
7. Three tiers — handle all three without speculating:
   - Named player + history  → rich narrative ("Armstrong again…")
   - Team known, no player   → team/temporal ("Forest pressing…")
   - Neither                 → pure moment ("Play breaks down…")
8. Do NOT say "According to the knowledge graph" or mention tools.
   Speak as if you simply know the match.
"""


def _execute_tool(name: str, args: dict, ttl_path: str) -> str:
    """Call the Python tool function and return JSON string result."""
    fn = TOOL_FN.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    # always inject the live ttl_path (ignore whatever path the model passes)
    args = dict(args)
    args["ttl_path"] = ttl_path
    try:
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _resolve_uris(event, ttl_path: str) -> tuple[str | None, str | None]:
    """
    Look up the event URI and match URI from ekg.ttl using
    hasTime + hasEventType as the key. Returns (event_uri, match_uri).
    """
    gametime    = getattr(event, "gametime", None)
    action_type = getattr(event, "action",   None)
    if not gametime or not action_type:
        return None, None
    g = _load(ttl_path)
    q = """
    PREFIX ekg: <http://soccerekg.org/ontology#>
    SELECT ?e ?match WHERE {
        ?e ekg:hasTime      "%s" ;
           ekg:hasEventType "%s" ;
           ekg:IN_MATCH     ?match .
    }
    LIMIT 1
    """ % (gametime, action_type)
    for r in g.query(q):
        return str(r.e), str(r.match)
    return None, None


def agent_commentate(event, ttl_path: str) -> str:
    """
    Tool-calling agent loop.  Sends messages to localhost:8001 and
    executes tool calls until finish_reason='stop' or MAX_TOOL_ROUNDS.
    Returns the final commentary string.
    """
    action     = getattr(event, "action",   "Unknown")
    gametime   = getattr(event, "gametime", "?")
    player     = getattr(event, "player",   None)
    team_name  = getattr(event, "team", None)
    event_uri, match_uri = _resolve_uris(event, ttl_path)
    description = getattr(event, "description", "") or ""

    suggested = _suggested_tools(action)

    user_content = (
        f"Event: {action} at {gametime}\n"
        f"Player: {player or 'unknown'}\n"
        f"Team: {team_name or 'unknown'}\n"
        f"VLM description: {description[:200] if description else 'none'}\n"
        f"Event URI: {event_uri or 'unknown'}\n"
        f"Match URI: {match_uri or 'unknown'}\n"
        f"TTL path: {ttl_path}\n"
        f"\nSuggested tools (you may call any): {suggested}\n"
        f"Generate live commentary for this event."
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            resp = requests.post(
                LLM_URL,
                json={
                    "model"   : LLM_MODEL,
                    "messages": messages,
                    "tools"   : TOOLS,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data   = resp.json()
            choice = data["choices"][0]
        except Exception as e:
            return f"[commentator] LLM error: {e}"

        finish = choice.get("finish_reason", "stop")
        msg    = choice["message"]

        if finish == "tool_calls" and msg.get("tool_calls"):
            # append assistant message with tool_calls
            messages.append({
                "role"      : "assistant",
                "content"   : msg.get("content"),
                "tool_calls": msg["tool_calls"],
            })
            # execute each tool and append results
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    fn_args = {}
                result_str = _execute_tool(fn_name, fn_args, ttl_path)
                messages.append({
                    "role"        : "tool",
                    "tool_call_id": tc["id"],
                    "content"     : result_str,
                })
        else:
            # finish_reason == "stop" or no tool calls
            return (msg.get("content") or "").strip()

    # exhausted rounds — return whatever content is in the last message
    return (msg.get("content") or "[commentator] max rounds reached").strip()


# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND THREAD
# ═══════════════════════════════════════════════════════════════════════════

def _handle_event(event, ttl_path: str):
    action   = getattr(event, "action",   "?")
    gametime = getattr(event, "gametime", "?")

    commentary = agent_commentate(event, ttl_path)

    print(f"\n[COMMENTARY] {gametime} {action}")
    print(commentary)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{gametime}] {action} | {commentary}\n")


def _commentator_loop(ttl_path: str):
    while True:
        event = event_queue.get()
        try:
            _handle_event(event, ttl_path)
        except Exception:
            traceback.print_exc()
        finally:
            event_queue.task_done()


def start_commentator(ttl_path: str):
    """Start the background commentary thread.  Call once at pipeline startup."""
    t = threading.Thread(target=_commentator_loop, args=(ttl_path,), daemon=True)
    t.start()
    print(f"[commentator] started → {ttl_path}")


def log_match_boundary(match_name: str, log_path: Path | None = None):
    """
    Write a section header to the shared log so multi-match runs can be
    split per-match by evaluate_commentary.py. Drain any pending events
    first so the boundary lands after the previous match's commentary.
    """
    try:
        event_queue.join()   # wait for queued events to be written
    except Exception:
        pass
    target = Path(log_path) if log_path else LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(f"\n=== MATCH: {match_name} ===\n")


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE MODE — generate commentary from an already-built KG
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys
    import re
    from types import SimpleNamespace
    from rdflib import RDFS

    _BASE_DIR   = Path(__file__).resolve().parent.parent.parent
    _DATA_DIR   = _BASE_DIR / "data"
    _TTL_DEFAULT = _BASE_DIR / "data" / "kg_output" / "ekg.ttl"

    ap = argparse.ArgumentParser(
        description="Generate AI commentary from an existing EKG (post-hoc)."
    )
    ap.add_argument("--ttl",     default=str(_TTL_DEFAULT), help="Path to ekg.ttl")
    ap.add_argument("--match",   help="Partial match name filter (e.g. 'Blackburn')")
    ap.add_argument("--all",     action="store_true", help="Process all matches in KG")
    ap.add_argument("--out-dir", help="Override output directory for all matches")
    ap.add_argument("--force",   action="store_true",
                    help="Overwrite ai_commentary.json if it already exists")
    args = ap.parse_args()

    if not args.all and not args.match:
        ap.print_help()
        sys.exit(1)

    ttl_path = args.ttl
    if not Path(ttl_path).exists():
        print(f"TTL not found: {ttl_path}")
        sys.exit(1)

    # ── discover matches in KG ───────────────────────────────────────────────
    g = _load(ttl_path)
    q_matches = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?match ?label WHERE {
        ?match a ekg:Match .
        OPTIONAL { ?match rdfs:label ?label }
    }
    """
    all_matches = []
    for r in g.query(q_matches):
        label = str(r.label) if r.label else str(r.match).split("/")[-1]
        all_matches.append((str(r.match), label))

    if args.match:
        all_matches = [(u, l) for u, l in all_matches
                       if args.match.lower() in l.lower()]

    if not all_matches:
        print("No matches found in KG" +
              (f" matching '{args.match}'" if args.match else "") + ".")
        sys.exit(1)

    print(f"Found {len(all_matches)} match(es) in KG.\n")

    def _find_match_folder(label: str, match_uri: str) -> Path:
        """Map a KG match label/URI back to a data/ folder."""
        # 1. exact folder name
        candidate = _DATA_DIR / label
        if candidate.is_dir():
            return candidate
        # 2. folder name contains the label (or significant words of it)
        for folder in sorted(_DATA_DIR.iterdir()):
            if not folder.is_dir():
                continue
            if label.lower() in folder.name.lower():
                return folder
            # partial: check that date + both team slugs appear in folder name
            slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            if slug[:12] in folder.name.lower():   # first 12 chars cover date
                return folder
        # 3. derive from URI slug
        uri_slug = match_uri.rstrip("/").split("/")[-1]
        for folder in sorted(_DATA_DIR.iterdir()):
            if folder.is_dir() and uri_slug[:12] in folder.name.replace(" - ", "_").lower():
                return folder
        return _DATA_DIR / label  # fallback (may not exist)

    # ── process each match ───────────────────────────────────────────────────
    for match_uri, match_label in all_matches:
        match_folder = _find_match_folder(match_label, match_uri)
        out_dir  = Path(args.out_dir) if args.out_dir else match_folder
        out_json = out_dir / "ai_commentary.json"
        out_log  = out_dir / "commentary_log.txt"

        print(f"{'─'*62}")
        print(f"Match : {match_label}")
        print(f"Folder: {match_folder}")

        if out_json.exists() and not args.force:
            print(f"[SKIP] ai_commentary.json already exists (use --force to overwrite)\n")
            continue

        # query events ordered by period then minute
        q_events = """
        PREFIX ekg:  <http://soccerekg.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?e ?type ?hasTime ?minute ?period ?desc ?playerLabel ?teamLabel WHERE {
            ?e ekg:IN_MATCH <%s> ;
               ekg:hasEventType ?type ;
               ekg:hasTime      ?hasTime ;
               ekg:hasMinute    ?minute ;
               ekg:hasPeriod    ?period .
            OPTIONAL { ?e ekg:hasDescription ?desc }
            OPTIONAL {
                ?player ekg:PERFORMED ?e ;
                        rdfs:label    ?playerLabel .
            }
            OPTIONAL {
                ?e ekg:INVOLVED_IN ?team .
                ?team rdfs:label   ?teamLabel .
            }
        }
        ORDER BY ?period ?minute
        """ % match_uri

        events = []
        for r in g.query(q_events):
            events.append({
                "uri"        : str(r.e),
                "action"     : str(r.type),
                "gametime"   : str(r.hasTime),
                "minute"     : float(r.minute),
                "period"     : int(r.period),
                "description": str(r.desc) if r.desc else "",
                "player"     : str(r.playerLabel) if r.playerLabel else None,
                "team"       : str(r.teamLabel)   if r.teamLabel   else None,
            })

        if not events:
            print("  [skip] No events found in KG for this match.\n")
            continue

        print(f"Events: {len(events)}\n")
        out_dir.mkdir(parents=True, exist_ok=True)

        output = []
        with open(out_log, "w", encoding="utf-8") as log_f:
            for i, ev in enumerate(events):
                event_obj = SimpleNamespace(
                    action      = ev["action"],
                    gametime    = ev["gametime"],
                    player      = ev["player"],
                    team        = ev["team"],
                    description = ev["description"],
                )
                text     = agent_commentate(event_obj, ttl_path)
                half_str = "1H" if ev["period"] == 1 else "2H"
                print(
                    f"  [{i+1:02d}/{len(events):02d}] "
                    f"{int(ev['minute'])}' {half_str} "
                    f"{ev['action']:<10} → \"{text[:120]}\""
                )
                log_f.write(f"[{ev['gametime']}] {ev['action']} | {text}\n")
                output.append({
                    "minute"    : int(ev["minute"]),
                    "half"      : ev["period"],
                    "event_type": ev["action"],
                    "player"    : ev["player"],
                    "team"      : ev["team"],
                    "ai_text"   : text,
                })

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n  Saved {len(output)} entries → {out_json}\n")
