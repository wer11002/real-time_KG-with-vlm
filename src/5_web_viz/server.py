"""
server.py — FastAPI backend for Soccer EKG live visualization

Endpoints:
  GET  /api/snapshot  → all events so far (reads full events_stream.jsonl)
  WS   /ws            → new events live (tails events_stream.jsonl, 200ms poll)
  GET  /              → serves React app from frontend/dist if built

Run:
  cd src/5_web_viz && python server.py
  # or: cd src/5_web_viz && uvicorn server:app --port 8000
"""

import asyncio
import json
import sys
import uvicorn
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
STREAM_PATH = BASE_DIR / "data" / "kg_output" / "events_stream.jsonl"
DIST_DIR    = Path(__file__).resolve().parent / "frontend" / "dist"

app = FastAPI()


# ── REST ───────────────────────────────────────────────────────────────────

@app.get("/api/snapshot")
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


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # Step 1: send full snapshot (all existing lines) on connect
        snapshot_end = 0
        if STREAM_PATH.exists():
            with open(STREAM_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        await ws.send_text(stripped)
                snapshot_end = f.tell()

        # Step 2: tail for new lines
        # Re-opens the file each poll to correctly handle truncation
        # (clear_stream() at pipeline restart shrinks the file).
        last_pos = snapshot_end
        while True:
            if STREAM_PATH.exists():
                size = STREAM_PATH.stat().st_size
                if size < last_pos:
                    last_pos = 0  # file truncated — pipeline restarted
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

@app.get("/")
async def serve_index():
    if (DIST_DIR / "index.html").exists():
        return FileResponse(str(DIST_DIR / "index.html"))
    return JSONResponse({
        "status": "backend running on :8000",
        "next": "cd src/5_web_viz/frontend && npm install && npm run dev",
        "ws": "ws://localhost:8000/ws",
        "snapshot": "http://localhost:8000/api/snapshot",
    })


@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    target = DIST_DIR / full_path
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    if (DIST_DIR / "index.html").exists():
        return FileResponse(str(DIST_DIR / "index.html"))
    return JSONResponse({"error": "not found"}, status_code=404)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(BASE_DIR))
    print(f"  Stream path : {STREAM_PATH}")
    print(f"  Frontend    : {DIST_DIR} ({'built' if DIST_DIR.exists() else 'run npm run dev'})")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
