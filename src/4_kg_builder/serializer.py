# serializer.py

from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD
from rdflib.namespace import Namespace

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")


def event_to_context(event_uri: URIRef, g: Graph, n_preceding: int = 3) -> dict:
    """
    Extract structured context for one event from the A-Box.
    Returns a dict — good for inspection and for building LLM prompts.
    """
    def lit(uri, prop):
        vals = list(g.objects(uri, prop))
        return str(vals[0]) if vals else None

    ctx = {
        "uri"        : str(event_uri),
        "event_type" : lit(event_uri, EKG.hasEventType),
        "time"       : lit(event_uri, EKG.hasTime),
        "minute"     : lit(event_uri, EKG.hasMinute),
        "period"     : lit(event_uri, EKG.hasPeriodNumber),
        "description": lit(event_uri, EKG.hasDescription),
        "full_text"  : lit(event_uri, EKG.hasFullText),
        "jersey"     : lit(event_uri, EKG.detectedJersey),
        "is_matched" : lit(event_uri, EKG.isMatched),
        "player"     : None,
        "player_team": None,
        "assist"     : None,
        "card"       : None,
        "preceded_by": [],
    }

    # player
    players = list(g.subjects(EKG.performed, event_uri))
    if players:
        p_uri         = players[0]
        ctx["player"] = lit(p_uri, RDFS.label) or str(p_uri).split("#")[-1]
        # team via direct playsFor triple
        team_uris = list(g.objects(p_uri, EKG.playsFor))
        if team_uris:
            ctx["player_team"] = lit(team_uris[0], RDFS.label)

    # assist
    assists = list(g.objects(event_uri, EKG.assistedBy))
    if assists:
        ctx["assist"] = lit(assists[0], RDFS.label) or str(assists[0]).split("#")[-1]

    # triggered card
    cards = list(g.objects(event_uri, EKG.triggered))
    if cards:
        ctx["card"] = lit(cards[0], EKG.hasEventType)

    # preceding events (walk precededBy chain)
    current = event_uri
    for _ in range(n_preceding):
        prev_list = list(g.objects(current, EKG.precededBy))
        if not prev_list:
            break
        prev = prev_list[0]
        prev_players = list(g.subjects(EKG.performed, prev))
        prev_player  = None
        if prev_players:
            prev_player = lit(prev_players[0], RDFS.label)
        ctx["preceded_by"].append({
            "type"  : lit(prev, EKG.hasEventType),
            "time"  : lit(prev, EKG.hasTime),
            "minute": lit(prev, EKG.hasMinute),
            "player": prev_player,
        })
        current = prev

    return ctx


def context_to_text(ctx: dict) -> str:
    """Convert extracted context dict → natural language block for LLM prompt."""
    lines = []

    period_str = {"1": "first half", "2": "second half"}.get(str(ctx.get("period", "")), "")
    minute_str = f"{float(ctx['minute']):.1f}'" if ctx.get("minute") else ctx.get("time", "?")

    lines.append(f"EVENT: {ctx['event_type']} at {minute_str}" +
                 (f" ({period_str})" if period_str else ""))

    if ctx.get("player"):
        team_str = f" ({ctx['player_team']})" if ctx.get("player_team") else ""
        lines.append(f"PLAYER: {ctx['player']}{team_str}")

    if ctx.get("assist"):
        lines.append(f"ASSIST: {ctx['assist']}")

    if ctx.get("card"):
        lines.append(f"CARD: {ctx['card']} issued following this event")

    if ctx.get("full_text"):
        lines.append(f"ESPN: {ctx['full_text']}")

    if ctx.get("description"):
        lines.append(f"VLM: {ctx['description']}")

    if ctx.get("preceded_by"):
        lines.append("RECENT EVENTS (before this):")
        for prev in reversed(ctx["preceded_by"]):
            t = f"{float(prev['minute']):.1f}'" if prev.get("minute") else prev.get("time", "?")
            p = f" by {prev['player']}" if prev.get("player") else ""
            lines.append(f"  {t}  {prev['type']}{p}")

    return "\n".join(lines)


def serialization_debug(g: Graph) -> list:
    """
    Run serialization on all PlayerAction nodes. Flag events with thin context.
    Returns list of (event_uri, time, issues).
    """
    issues = []
    events = list(g.subjects(RDF.type, EKG.PlayerAction))

    print(f"\n── Serialization debug: {len(events)} PlayerAction nodes ──")
    for ev in events:
        ctx       = event_to_context(ev, g)
        ev_issues = []

        if not ctx["player"]:
            ev_issues.append("NO PLAYER — commentator can't name who did it")
        if not ctx["minute"]:
            ev_issues.append("NO hasMinute — can't say 'in the 67th minute'")
        if not ctx["description"] and not ctx["full_text"]:
            ev_issues.append("NO TEXT — neither hasDescription nor hasFullText")
        if not ctx["preceded_by"]:
            ev_issues.append("NO precededBy — no build-up narrative possible")

        if ev_issues:
            t = ctx.get("time", "?")
            issues.append((str(ev), t, ev_issues))

    if not issues:
        print("  ✓ All events have sufficient context for commentary")
    else:
        print(f"  {len(issues)} events with thin context:")
        for uri, t, evissues in issues[:10]:
            name = uri.split("#")[-1]
            print(f"  ✗ {t:<8} {name}")
            for i in evissues:
                print(f"          → {i}")

    return issues
