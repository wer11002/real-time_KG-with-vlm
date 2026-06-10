"""
server.py — FastAPI backend for Soccer EKG live visualization

Endpoints:
  GET  /api/snapshot  → all events so far (reads full events_stream.jsonl)
  GET  /api/graph     → parse ekg.ttl → {nodes, links} for static visualization
  WS   /ws            → new events live (tails events_stream.jsonl, 200ms poll)
  GET  /              → serves React app from frontend/dist if built

Run:
  cd src/5_web_viz && python server.py
  # or: cd src/5_web_viz && uvicorn server:app --port 8002
"""

import asyncio
import json
import sys
import uvicorn
from pathlib import Path

from fastapi import FastAPI, APIRouter, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
STREAM_PATH = BASE_DIR / "data" / "kg_output" / "events_stream.jsonl"
DIST_DIR    = Path(__file__).resolve().parent / "frontend" / "dist"

app = FastAPI()

# All REST API routes go through this router so they are guaranteed to be
# checked before any static-file mount that lives at the root path.
api = APIRouter(prefix="/api")


# ── REST ───────────────────────────────────────────────────────────────────

@api.get("/snapshot")
async def snapshot():
    """Return all events ingested so far as a JSON array."""
    if not STREAM_PATH.exists():
        return JSONResponse({"events": []})
    events = []
    with open(STREAM_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return JSONResponse({"events": events})


@api.get("/graph")
async def get_graph():
    """Parse ekg.ttl → {nodes, links} for static KG visualization."""
    from rdflib import Graph, RDF, RDFS, Namespace, Literal  # noqa

    TTL_PATH = BASE_DIR / "data" / "ekg.ttl"
    if not TTL_PATH.exists():
        TTL_PATH = BASE_DIR / "ekg.ttl"   # fallback: project root
    if not TTL_PATH.exists():
        return JSONResponse({"nodes": [], "links": [], "error": "ekg.ttl not found"})

    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")

    EKG = Namespace("http://soccerekg.org/ontology#")

    EVENT_TYPES = [
        "Goal", "Shot", "Foul", "Corner",
        "FreeKick", "Substitution", "OffsideCalled",
    ]

    def local_name(uri):
        s = str(uri)
        return s.split("#")[-1] if "#" in s else s.split("/")[-1]

    def get_label(subj):
        lbl = g.value(subj, RDFS.label)
        return str(lbl) if lbl else local_name(subj).replace("_", " ").title()

    nodes, links = [], []
    node_map, link_keys = {}, set()

    def add_node(uri, node):
        k = str(uri)
        if k not in node_map:
            node_map[k] = node
            nodes.append(node)

    def add_link(src_uri, tgt_uri, label):
        s = node_map.get(str(src_uri))
        t = node_map.get(str(tgt_uri))
        if not s or not t:
            return
        key = f"{s['id']}|{t['id']}|{label}"
        if key not in link_keys:
            link_keys.add(key)
            links.append({"source": s["id"], "target": t["id"], "label": label})

    # Team-side map: scan event hasTeamSide + involvedTeam
    team_side_map = {}
    for event in g.subjects(EKG.hasTeamSide, None):
        side = str(g.value(event, EKG.hasTeamSide))
        for team in g.subjects(EKG.involvedTeam, event):
            team_side_map.setdefault(str(team), side)

    # Player-side map: scan performed → event hasTeamSide
    player_side_map = {}
    for player in g.subjects(RDF.type, EKG.Player):
        for event in g.objects(player, EKG.performed):
            side_lit = g.value(event, EKG.hasTeamSide)
            if side_lit:
                player_side_map[str(player)] = str(side_lit)
                break

    # Match nodes
    for subj in g.subjects(RDF.type, EKG.Match):
        uid = local_name(subj)
        add_node(subj, {
            "id": uid, "nodeType": "match",
            "label": get_label(subj),
            "rawData": {"match_id": uid},
        })

    # Team nodes
    for subj in g.subjects(RDF.type, EKG.Team):
        uid = local_name(subj)
        side = team_side_map.get(str(subj), "home")
        add_node(subj, {
            "id": uid, "nodeType": "team", "side": side,
            "label": get_label(subj),
            "rawData": {"team_id": uid, "side": side},
        })

    # Player nodes
    for subj in g.subjects(RDF.type, EKG.Player):
        uid = local_name(subj)
        side = player_side_map.get(str(subj), "home")
        jersey = g.value(subj, EKG.hasJerseyNumber)
        add_node(subj, {
            "id": uid, "nodeType": "player", "side": side,
            "label": get_label(subj),
            "rawData": {
                "player_id": uid,
                "jersey": str(jersey) if jersey else None,
                "side": side,
            },
        })

    # Event nodes
    for evt_type in EVENT_TYPES:
        for subj in g.subjects(RDF.type, EKG[evt_type]):
            uid = local_name(subj)
            time_raw = g.value(subj, EKG.hasTime)
            label = f"{evt_type} {time_raw or ''}".strip()
            raw = {"event_id": uid, "event_type": evt_type}
            for pred, obj in g.predicate_objects(subj):
                if isinstance(obj, Literal):
                    raw[local_name(pred)] = str(obj)
            add_node(subj, {"id": uid, "nodeType": evt_type, "label": label, "rawData": raw})

    # Edges
    for s, _, o in g.triples((None, EKG.performed, None)):
        add_link(s, o, "performed")
    for s, _, o in g.triples((None, EKG.involvedTeam, None)):
        add_link(s, o, "involvedTeam")
    for s, _, o in g.triples((None, EKG.inMatch, None)):
        add_link(s, o, "inMatch")
    for s, _, o in g.triples((None, EKG.precededBy, None)):
        add_link(s, o, "precededBy")

    return JSONResponse({"nodes": nodes, "links": links})


# Register the API router on the app (before any static mount)
app.include_router(api)


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        snapshot_end = 0
        if STREAM_PATH.exists():
            with open(STREAM_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        await ws.send_text(stripped)
                snapshot_end = f.tell()

        last_pos = snapshot_end
        while True:
            if STREAM_PATH.exists():
                size = STREAM_PATH.stat().st_size
                if size < last_pos:
                    last_pos = 0
                if size > last_pos:
                    with open(STREAM_PATH, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            stripped = line.strip()
                            if stripped:
                                await ws.send_text(stripped)
                        last_pos = f.tell()
            await asyncio.sleep(0.2)

    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Static (production build) ──────────────────────────────────────────────
# Mounted AFTER the API router so API routes always win.

if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
else:
    @app.get("/")
    async def dev_info():
        return JSONResponse({
            "status": "backend running on :8002",
            "next": "cd src/5_web_viz/frontend && npm install && npm run dev",
            "api_graph": "http://localhost:8002/api/graph",
            "ws": "ws://localhost:8002/ws",
        })


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(BASE_DIR))
    print(f"  Stream path : {STREAM_PATH}")
    print(f"  Frontend    : {DIST_DIR} ({'built' if DIST_DIR.exists() else 'run npm run dev'})")
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)
