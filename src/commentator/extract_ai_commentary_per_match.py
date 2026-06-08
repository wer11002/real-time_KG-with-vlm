"""
extract_ai_commentary_per_match.py
──────────────────────────────────
Splits data/commentator_output/commentary_log.txt into per-match
ai_commentary.json files using the EKG as the source of truth.

For every match in data/kg_output/ekg.ttl we:
  1. Query its events (type, minute, period, player, team, gametime).
  2. Walk commentary_log.txt in order, claiming each log line for the
     first unused KG event of the same (period, event_type) whose
     minute is within ±1 of the line's minute.
  3. Write data/<match folder>/ai_commentary.json in the same schema
     produced by generate_ground_truth_commentary.py
     (fields: minute, half, event_type, player, team, human_text).

Run after main.py has produced both ekg.ttl and commentary_log.txt.

Usage:
    python src/commentator/extract_ai_commentary_per_match.py
"""

import json
import re
import sys
from pathlib import Path

from rdflib import Graph

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data"
TTL_PATH  = DATA_DIR / "kg_output" / "ekg.ttl"
LOG_PATH  = DATA_DIR / "commentator_output" / "commentary_log.txt"

LOG_LINE = re.compile(
    r"\[(\d+)(?:st|nd|rd|th)\s+(\d+):(\d+)\]\s+(\w+)\s+\|\s*(.+)"
)
TOL_MIN  = 1   # minute tolerance when pairing KG events with log lines


# ── log parsing ────────────────────────────────────────────────────────────

def parse_commentary_log(path: Path) -> list[dict]:
    """
    Parse '[1st 18:05] Goal | text' lines from the shared log.
    '=== MATCH: <name> ===' headers do not match LOG_LINE and are skipped
    naturally — we attribute lines via KG membership, not headers.
    """
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(errors="ignore").splitlines():
        m = LOG_LINE.match(raw.strip())
        if not m:
            continue
        half, mins, secs, etype, text = m.groups()
        out.append({
            "half"      : int(half),
            "minute"    : int(mins),
            "second"    : int(secs),
            "event_type": etype.strip(),
            "text"      : text.strip(),
            "used"      : False,
        })
    return out


# ── SPARQL ─────────────────────────────────────────────────────────────────

def load_matches(g: Graph) -> list[tuple[str, str]]:
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?match ?label WHERE {
        ?match a ekg:Match .
        OPTIONAL { ?match rdfs:label ?label }
    }
    """
    return [
        (str(r.match), str(r.label) if r.label else "")
        for r in g.query(q)
    ]


def load_events_for_match(g: Graph, match_uri: str) -> list[dict]:
    """
    All events in one match, ordered by (period, minute, gametime, URI).
    Both IS_PERFORMED_BY and the inverse PERFORMED are checked so we work
    whichever direction the schema actually stores.
    """
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?e ?type ?minute ?period ?gametime ?playerLabel ?teamLabel WHERE {
        ?e ekg:IN_MATCH      <%s> ;
           ekg:hasEventType  ?type ;
           ekg:hasMinute     ?minute ;
           ekg:hasPeriod     ?period ;
           ekg:hasTime       ?gametime .
        OPTIONAL {
            { ?e ekg:IS_PERFORMED_BY ?p } UNION { ?p ekg:PERFORMED ?e }
            ?p rdfs:label ?playerLabel .
        }
        OPTIONAL {
            ?e ekg:INVOLVED_IN ?t .
            ?t rdfs:label      ?teamLabel .
        }
    }
    ORDER BY ?period ?minute ?gametime ?e
    """ % match_uri

    events = []
    for r in g.query(q):
        events.append({
            "uri"       : str(r.e),
            "event_type": str(r.type),
            "minute"    : int(float(r.minute)),
            "half"      : int(r.period),
            "gametime"  : str(r.gametime),
            "player"    : str(r.playerLabel) if r.playerLabel else "",
            "team"      : str(r.teamLabel)   if r.teamLabel   else "",
        })
    return events


# ── KG match URI → data/<folder> ───────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_folder_for_match(match_uri: str, label: str,
                          data_dir: Path) -> Path | None:
    candidates = [f for f in data_dir.iterdir() if f.is_dir()]

    # 1. exact rdfs:label match
    if label:
        for f in candidates:
            if f.name == label:
                return f
        for f in candidates:
            if label.lower() in f.name.lower():
                return f

    # 2. URI slug → folder (e.g. match_2019_10_01_blackburn_rovers_nottingham_forest)
    slug = match_uri.rstrip("/").split("/")[-1].split("#")[-1]
    if slug.startswith("match_"):
        slug = slug[len("match_"):]
    slug_norm = _norm(slug)
    for f in candidates:
        if _norm(f.name) == slug_norm:
            return f
    for f in candidates:
        fnorm = _norm(f.name)
        if slug_norm and (slug_norm in fnorm or fnorm in slug_norm):
            return f

    # 3. date prefix + team slug match
    m = re.match(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})_?(.*)", slug)
    if m:
        date_str   = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        teams_norm = _norm(m.group(4))
        for f in candidates:
            if not f.name.startswith(date_str):
                continue
            fnorm = _norm(f.name)
            if not teams_norm or teams_norm in fnorm:
                return f

    return None


# ── pairing KG events with log lines (in-order consumption) ────────────────

def consume_match(kg_events: list[dict], log_lines: list[dict],
                  tol_min: int = TOL_MIN) -> list[dict]:
    output = []
    for ev in kg_events:
        match_idx = None
        for i, line in enumerate(log_lines):
            if line["used"]:
                continue
            if line["half"] != ev["half"]:
                continue
            if line["event_type"].lower() != ev["event_type"].lower():
                continue
            if abs(line["minute"] - ev["minute"]) > tol_min:
                continue
            match_idx = i
            break
        if match_idx is None:
            continue
        log_lines[match_idx]["used"] = True
        output.append({
            "minute"    : ev["minute"],
            "half"      : ev["half"],
            "event_type": ev["event_type"],
            "player"    : ev["player"],
            "team"      : ev["team"],
            "human_text": log_lines[match_idx]["text"],
        })
    return output


# ── main ───────────────────────────────────────────────────────────────────

def main():
    if not TTL_PATH.exists():
        print(f"TTL not found: {TTL_PATH}")
        sys.exit(1)
    if not LOG_PATH.exists():
        print(f"Commentary log not found: {LOG_PATH}")
        sys.exit(1)

    print(f"Loading KG from {TTL_PATH} …")
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    print(f"  {len(g):,} triples loaded")

    print(f"Loading commentary log from {LOG_PATH} …")
    log_lines = parse_commentary_log(LOG_PATH)
    print(f"  {len(log_lines):,} log lines parsed")

    matches = load_matches(g)
    if not matches:
        print("No matches found in KG.")
        sys.exit(1)
    print(f"\nFound {len(matches)} match(es) in KG.\n")

    skipped = []
    for match_uri, label in matches:
        events = load_events_for_match(g, match_uri)
        folder = find_folder_for_match(match_uri, label, DATA_DIR)
        name   = label or match_uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

        if folder is None:
            skipped.append(name)
            print(f"  [skip] {name[:50]:<52} : "
                  f"{len(events):3d} KG events — no matching folder")
            continue

        ai_entries = consume_match(events, log_lines)
        out_path   = folder / "ai_commentary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ai_entries, f, indent=2, ensure_ascii=False)

        print(f"  {folder.name[:50]:<52} : "
              f"{len(events):3d} KG events → "
              f"{len(ai_entries):3d} matched → ai_commentary.json saved")

    unused = sum(1 for l in log_lines if not l["used"])
    print(f"\n  Unclaimed log lines: {unused}/{len(log_lines)} "
          f"(no KG event within ±{TOL_MIN} min of same type)")
    if skipped:
        print(f"  Matches with no folder match ({len(skipped)}):")
        for s in skipped:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
