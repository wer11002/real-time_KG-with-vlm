"""
scrape_ground_truth_sources.py
─────────────────────────────────────────────────────────────────────
Scrape ground-truth football commentary for the 7 EFL Championship
matches played on 2019-10-01.

Active sources (default run):
    1. Sky Sports match report      (URL hardcoded — 7/7)
    2. BBC Sport match report       (2/7 hardcoded + range probe)
    3. ESPN full commentary thread  (6/7 hardcoded + 1 discovered)

Optional (off by default; pass --source club or --source reddit):
    4. Club official site reports
    5. Reddit thread (Arctic Shift)

Output per match folder under data/<match>/groundtruth_sources/:
    sky_report.txt                — Sky Sports article body
    bbc_report.txt                — BBC Sport article body
    espn_full_commentary.json     — full commentary feed as JSON
    sources_status.json           — per-source success/failure log

Every network call is wrapped in try/except. The script never crashes
on a bad URL — it logs the failure and moves on to the next source.

CLI:
    python src/commentator/scrape_ground_truth_sources.py
    python src/commentator/scrape_ground_truth_sources.py --match "Blackburn"
    python src/commentator/scrape_ground_truth_sources.py --source espn
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib  import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")
HDR = {
    "User-Agent"     : UA,
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}

REQ_TIMEOUT   = 15
SLEEP_BETWEEN = 0.6


# ════════════════════════════════════════════════════════════════════
# HARDCODED MATCH-LEVEL TABLES (keyed by match folder name)
# ════════════════════════════════════════════════════════════════════
# Each value is what we need to construct the source URL directly,
# bypassing the previous (broken) URL-discovery logic. Add rows as new
# IDs are found by hand.

SKY_KNOWN_IDS: dict[str, tuple[str, int]] = {
    "2019-10-01 - Blackburn Rovers - Nottingham Forest":
        ("blackburn-vs-n-forest",   409456),
    "2019-10-01 - Hull City - Sheffield Wednesday":
        ("hull-city-vs-sheff-wed",  409457),
    "2019-10-01 - Leeds United - West Bromwich":
        ("leeds-vs-w-brom",         409458),
    "2019-10-01 - Middlesbrough - Preston North End":
        ("mboro-vs-preston",        409459),
    "2019-10-01 - Reading - Fulham":
        ("reading-vs-fulham",       409460),
    "2019-10-01 - Stoke City - Huddersfield Town":
        ("stoke-vs-huddsfld",       409461),
    "2019-10-01 - Brentford - Bristol City":
        ("brentford-vs-bristol-c",  409464),
}

BBC_KNOWN_IDS: dict[str, int] = {
    "2019-10-01 - Leeds United - West Bromwich":    49805857,
    "2019-10-01 - Stoke City - Huddersfield Town":  49805862,
}
BBC_PROBE_RANGE = range(49805850, 49805900)

ESPN_GAME_IDS: dict[str, int] = {
    "2019-10-01 - Blackburn Rovers - Nottingham Forest":  544482,
    "2019-10-01 - Brentford - Bristol City":              544474,
    "2019-10-01 - Hull City - Sheffield Wednesday":       544484,
    "2019-10-01 - Leeds United - West Bromwich":          544485,
    "2019-10-01 - Reading - Fulham":                      544479,
    "2019-10-01 - Stoke City - Huddersfield Town":        544480,
    # "2019-10-01 - Middlesbrough - Preston North End": discovered at runtime
}


# Optional sources (off by default) — preserved from the previous version.
CLUB_SITES = {
    "Blackburn Rovers"    : "https://www.rovers.co.uk",
    "Nottingham Forest"   : "https://www.nottinghamforest.co.uk",
    "Brentford"           : "https://www.brentfordfc.com",
    "Bristol City"        : "https://www.bcfc.co.uk",
    "Hull City"           : "https://www.hullcitytigers.com",
    "Sheffield Wednesday" : "https://www.swfc.co.uk",
    "Leeds United"        : "https://www.leedsunited.com",
    "West Bromwich Albion": "https://www.wba.co.uk",
    "West Bromwich"       : "https://www.wba.co.uk",
    "Middlesbrough"       : "https://www.mfc.co.uk",
    "Preston North End"   : "https://www.pnefc.net",
    "Reading"             : "https://www.readingfc.co.uk",
    "Fulham"              : "https://www.fulhamfc.com",
    "Stoke City"          : "https://www.stokecityfc.com",
    "Huddersfield Town"   : "https://www.htafc.com",
}


# ════════════════════════════════════════════════════════════════════
# COMMON UTILITIES
# ════════════════════════════════════════════════════════════════════

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_match_folder(folder: Path):
    parts = folder.name.split(" - ", 2)
    if len(parts) != 3:
        return None
    date, t1, t2 = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    return date.strip(), t1.strip(), t2.strip()


def find_matches(filter_name: str | None) -> list[tuple[Path, str, str, str]]:
    out = []
    for f in sorted(DATA_DIR.iterdir()):
        if not f.is_dir():
            continue
        parsed = parse_match_folder(f)
        if not parsed:
            continue
        if filter_name and filter_name.lower() not in f.name.lower():
            continue
        date, t1, t2 = parsed
        out.append((f, date, t1, t2))
    return out


def fetch(url: str) -> tuple[int, str | None]:
    try:
        r = requests.get(url, headers=HDR, timeout=REQ_TIMEOUT)
        return r.status_code, (r.text if r.status_code == 200 else None)
    except Exception:
        return 0, None


def extract_text(html: str, selectors: list[str]) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return "\n".join(
                line.strip()
                for line in node.get_text("\n").splitlines()
                if line.strip()
            )
    return None


def _lookup_by_folder(folder_name: str, table: dict):
    """Exact-then-case-insensitive lookup of a folder name in a hardcoded
    table. Returns None if no row exists."""
    if folder_name in table:
        return table[folder_name]
    lc = folder_name.lower()
    for k, v in table.items():
        if k.lower() == lc:
            return v
    return None


# ════════════════════════════════════════════════════════════════════
# SOURCE 1 — Sky Sports
# ════════════════════════════════════════════════════════════════════

SKY_SELECTORS = [
    "div.sdc-article-body",
    "div.sdc-site-article__body",
    "div.article__body",
    "article",
    "div.page-content",
]


def _extract_sky_text(html: str) -> str:
    """
    Try each Sky selector in order and keep the one that yields the
    longest cleaned text. If every selector misses, concatenate all <p>
    tags inside <main>. Returns "" when nothing usable is found.
    """
    soup = BeautifulSoup(html, "lxml")
    best = ""
    for sel in SKY_SELECTORS:
        node = soup.select_one(sel)
        if not node:
            continue
        text = "\n".join(
            line.strip()
            for line in node.get_text("\n").splitlines()
            if line.strip()
        )
        if len(text) > len(best):
            best = text
    if best:
        return best
    main = soup.find("main")
    if main:
        paras = [p.get_text(strip=True) for p in main.find_all("p")]
        paras = [p for p in paras if p]
        if paras:
            return "\n".join(paras)
    return ""


def scrape_sky(folder: Path) -> dict:
    out_path = folder / "groundtruth_sources" / "sky_report.txt"
    entry    = _lookup_by_folder(folder.name, SKY_KNOWN_IDS)
    if not entry:
        return {"status": "not_found",
                "reason": "no Sky entry — add to SKY_KNOWN_IDS"}

    sky_slug, gid = entry
    # Sky migrated their match-report URLs over the years. Try the
    # report path first, then the bare ID path, then the preview path
    # (preview pages typically have ~600-1200 words of build-up text).
    urls = [
        f"https://www.skysports.com/football/{sky_slug}/report/{gid}",
        f"https://www.skysports.com/football/{sky_slug}/{gid}",
        f"https://www.skysports.com/football/{sky_slug}/preview/{gid}",
    ]

    last_html = None
    last_url  = None
    for url in urls:
        code, html = fetch(url)
        time.sleep(SLEEP_BETWEEN)
        # Inline diagnostic so failing runs explain themselves.
        title = ""
        if html:
            t = BeautifulSoup(html, "lxml").find("title")
            title = t.get_text(strip=True) if t else ""
        print(f"  [sky-debug] {url} → status={code} "
              f"bytes={len(html) if html else 0} title={title[:60]!r}")
        if code != 200 or not html:
            continue
        last_html = html
        last_url  = url
        text = _extract_sky_text(html)
        if text and len(text) > 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            return {"status": "ok", "url": url, "bytes": len(text)}

    # We hit 200 but every selector returned nothing — save the raw
    # HTML so you can grep for the real article container manually.
    if last_html:
        debug_path = folder / "groundtruth_sources" / "sky_DEBUG.html"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(last_html, encoding="utf-8")
        return {"status"    : "no_content",
                "url"       : last_url,
                "debug_html": str(debug_path)}
    return {"status": "blocked_or_404", "tried": urls}


# ════════════════════════════════════════════════════════════════════
# SOURCE 2 — BBC Sport (with ID range probe for unknowns)
# ════════════════════════════════════════════════════════════════════

def _bbc_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    t    = soup.find("title")
    return t.get_text() if t else ""


def discover_bbc_ids(unknown: list[tuple[str, str, str]]) -> dict[str, int]:
    """
    For matches we don't have a hardcoded BBC ID for, probe every ID in
    BBC_PROBE_RANGE in turn, parse the <title>, and check whether both
    team-name tokens appear. Returns {folder_name: id} for any hits.

    We probe each ID once and try to match it against ALL remaining
    unknowns in one pass, so the worst-case cost is len(range) HTTP
    requests regardless of how many matches we're hunting for.
    """
    if not unknown:
        return {}
    print(f"  [bbc-discover] probing {len(BBC_PROBE_RANGE)} IDs "
          f"for {len(unknown)} unknown match(es) …")
    pending = list(unknown)   # mutable copy
    found   : dict[str, int] = {}

    for gid in BBC_PROBE_RANGE:
        if not pending:
            break
        url = f"https://www.bbc.co.uk/sport/football/{gid}"
        code, html = fetch(url)
        time.sleep(SLEEP_BETWEEN)
        if code != 200 or not html:
            continue
        title_lc = _bbc_title(html).lower()
        if not title_lc:
            continue
        # Look for the first unknown whose teams both appear in the title.
        for entry in list(pending):
            folder_name, t1, t2 = entry
            t1_tokens = [w for w in re.split(r"\W+", t1.lower()) if len(w) > 3]
            t2_tokens = [w for w in re.split(r"\W+", t2.lower()) if len(w) > 3]
            if (any(w in title_lc for w in t1_tokens)
                and any(w in title_lc for w in t2_tokens)):
                found[folder_name] = gid
                pending.remove(entry)
                print(f"  [bbc-discover] {folder_name} → {gid}")
                break
    if pending:
        print(f"  [bbc-discover] still missing: "
              f"{', '.join(name for name, _, _ in pending)}")
    return found


def scrape_bbc(folder: Path, bbc_ids: dict[str, int]) -> dict:
    out_path = folder / "groundtruth_sources" / "bbc_report.txt"
    gid      = _lookup_by_folder(folder.name, bbc_ids)
    if not gid:
        return {"status": "not_found",
                "reason": "no BBC article ID hardcoded or discovered"}

    url        = f"https://www.bbc.co.uk/sport/football/{gid}"
    code, html = fetch(url)
    time.sleep(SLEEP_BETWEEN)
    if code == 403:
        return {"status": "blocked", "url": url, "error": "403"}
    if code != 200 or not html:
        return {"status": "blocked_or_404", "url": url,
                "error": f"HTTP {code}"}
    text = extract_text(html, [
        "article",
        "main",
        "[data-component='text-block']",
    ])
    if not text:
        return {"status": "no_content", "url": url}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {"status": "ok", "url": url, "bytes": len(text)}


# ════════════════════════════════════════════════════════════════════
# SOURCE 3 — ESPN full commentary (JSON API)
# ════════════════════════════════════════════════════════════════════
# The /soccer/commentary/_/gameId/<id> HTML page is a React shell with
# no commentary in the initial markup, so the old HTML parser returned
# zero hits. We hit ESPN's public site.api.espn.com summary endpoint
# instead, which serves the full commentary as JSON.

ESPN_API_URL = ("https://site.api.espn.com/apis/site/v2/"
                "sports/soccer/eng.2/summary?event={gid}")

# Candidate IDs to probe when ESPN_GAME_IDS doesn't list a match
# (currently only Middlesbrough vs Preston is unknown).
ESPN_PROBE_IDS = (544475, 544476, 544477, 544478, 544481, 544483, 544486)


def _minute_absolute(raw: str) -> int | None:
    """'63'' → 63, '45'+2'' → 47, '90+3' → 93."""
    s = re.sub(r"[^\d+]", "", str(raw))
    if not s:
        return None
    if "+" in s:
        a, _, b = s.partition("+")
        try:
            return int(a) + (int(b) if b else 0)
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def _event_type_from_text(text: str) -> str:
    """First-words heuristic — used when ESPN's `type.text` is absent."""
    t    = text.strip().lower()
    head = t[:60]
    if t.startswith("goal"):
        return "Goal"
    if "yellow card" in head or "booked" in head or "shown a yellow" in head:
        return "YellowCard"
    if "red card" in head or "sent off" in head:
        return "RedCard"
    if t.startswith("attempt") or t.startswith("shot") or "header" in head:
        return "Shot"
    if t.startswith("foul"):
        return "Foul"
    if t.startswith("corner"):
        return "Corner"
    if t.startswith("free kick") or t.startswith("free-kick"):
        return "Free_Kick"
    if t.startswith("substitution") or "comes on" in head or "comes off" in head:
        return "Substitution"
    if t.startswith("offside"):
        return "Offside"
    return "Other"


def _normalize_event_type(type_text: str, full_text: str) -> str:
    """Map ESPN's type.text (e.g. 'Attempt Saved', 'Goal', 'Yellow Card')
    to one of our KG action types. Falls back to text-based heuristic
    if no obvious keyword fires."""
    t = (type_text or "").lower()
    if "goal" in t and "no goal" not in t:
        return "Goal"
    if "yellow" in t:
        return "YellowCard"
    if "red" in t and "card" in t:
        return "RedCard"
    if "attempt" in t or "shot" in t or "header" in t:
        return "Shot"
    if "foul" in t:
        return "Foul"
    if "corner" in t:
        return "Corner"
    if "free kick" in t or "free-kick" in t:
        return "Free_Kick"
    if "substitut" in t:
        return "Substitution"
    if "offside" in t:
        return "Offside"
    return _event_type_from_text(full_text)


def _fetch_espn_json(gid: int) -> tuple[int, dict | None]:
    """GET the summary JSON for one ESPN game ID. (status, parsed_json)."""
    url = ESPN_API_URL.format(gid=gid)
    try:
        r = requests.get(url, headers={"User-Agent": UA},
                         timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return r.status_code, None
        return 200, r.json()
    except Exception:
        return 0, None


def _extract_team_names(data: dict) -> list[str]:
    """Pull team display names out of the summary's header.competitions."""
    names = []
    for comp in (data.get("header", {}) or {}).get("competitions", []):
        for c in comp.get("competitors", []) or []:
            team = c.get("team", {}) or {}
            for k in ("displayName", "name", "shortDisplayName", "abbreviation"):
                v = team.get(k)
                if v:
                    names.append(str(v))
    return names


def _parse_espn_json_commentary(data: dict) -> list[dict]:
    """
    Find the commentary list inside ESPN's summary JSON. The current API
    nests it under 'commentary'; older variants used 'plays' or
    'header.competitions[0].details'. Each entry has type, text,
    clock, and period.
    """
    candidates: list = []
    if isinstance(data.get("commentary"), list):
        candidates = data["commentary"]
    elif isinstance(data.get("plays"), list):
        candidates = data["plays"]
    else:
        for comp in (data.get("header", {}) or {}).get("competitions", []):
            details = comp.get("details")
            if isinstance(details, list):
                candidates = details
                break

    out: list[dict] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        text = (entry.get("text") or "").strip()
        if not text:
            continue

        clock      = entry.get("clock")  if isinstance(entry.get("clock"),  dict) else {}
        period     = entry.get("period") if isinstance(entry.get("period"), dict) else {}
        etype      = entry.get("type")   if isinstance(entry.get("type"),   dict) else {}
        clock_val  = clock.get("value")
        clock_disp = clock.get("displayValue", "")

        try:
            period_num = int(period.get("number", 1))
        except (TypeError, ValueError):
            period_num = 1

        if isinstance(clock_val, (int, float)):
            abs_min = int(clock_val)
        else:
            abs_min = _minute_absolute(clock_disp) or 0

        minute_str = clock_disp or f"{abs_min}'"
        guess = _normalize_event_type(etype.get("text", ""), text)

        out.append({
            "minute"          : minute_str,
            "minute_absolute" : abs_min,
            "period"          : period_num,
            "text"            : text,
            "event_type_guess": guess,
        })
    return out


def _discover_espn_id_via_api(t1: str, t2: str) -> int | None:
    """
    Try each ID in ESPN_PROBE_IDS against the summary API and confirm
    the response's competitors contain both team names. Returns the
    first match or None.
    """
    t1_tokens = [w for w in re.split(r"\W+", t1.lower()) if len(w) > 3]
    t2_tokens = [w for w in re.split(r"\W+", t2.lower()) if len(w) > 3]
    print(f"  [espn-discover] probing {len(ESPN_PROBE_IDS)} IDs for {t1} vs {t2} …")
    for gid in ESPN_PROBE_IDS:
        code, data = _fetch_espn_json(gid)
        time.sleep(SLEEP_BETWEEN)
        if code != 200 or not isinstance(data, dict):
            continue
        teams_lc = " ".join(_extract_team_names(data)).lower()
        if (any(w in teams_lc for w in t1_tokens) and
            any(w in teams_lc for w in t2_tokens)):
            print(f"  [espn-discover] → matched gameId {gid}")
            return gid
    print(f"  [espn-discover] no probe ID matched")
    return None


def scrape_espn(folder: Path, date: str, t1: str, t2: str) -> dict:
    out_path = folder / "groundtruth_sources" / "espn_full_commentary.json"
    gid      = _lookup_by_folder(folder.name, ESPN_GAME_IDS)
    if not gid:
        gid = _discover_espn_id_via_api(t1, t2)
    if not gid:
        return {"status": "not_found",
                "reason": "no ESPN game ID hardcoded or discoverable"}

    url        = ESPN_API_URL.format(gid=gid)
    code, data = _fetch_espn_json(gid)
    time.sleep(SLEEP_BETWEEN)
    if code != 200 or not isinstance(data, dict):
        return {"status": "blocked_or_404", "url": url,
                "error":  f"HTTP {code}"}

    items = _parse_espn_json_commentary(data)
    if not items:
        # Persist the raw response so the JSON shape can be inspected.
        debug_path = folder / "groundtruth_sources" / "espn_DEBUG.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        return {"status"    : "no_content",
                "url"       : url,
                "game_id"   : gid,
                "debug_json": str(debug_path),
                "top_keys"  : list(data.keys())[:12]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return {"status" : "ok",
            "url"    : url,
            "game_id": gid,
            "lines"  : len(items)}


# ════════════════════════════════════════════════════════════════════
# OPTIONAL SOURCES — kept available via --source but not on default
# ════════════════════════════════════════════════════════════════════

def _looks_like_match_report(href: str, opponent: str, date: str) -> bool:
    if not href:
        return False
    h = href.lower()
    opp_tokens = [w for w in re.split(r"\W+", opponent.lower()) if len(w) > 3]
    if not any(w in h for w in opp_tokens):
        return False
    if "match-report" in h or "highlights" in h or "report" in h:
        return True
    year = date.split("-")[0]
    return f"/{year}/" in h


def _find_club_report_url(site_root: str, opponent: str,
                          date: str) -> str | None:
    year, month_num, _ = date.split("-")
    month_name = datetime(int(year), int(month_num), 1).strftime("%B").lower()
    archives = [
        f"{site_root}/news/{year}/{month_name}/",
        f"{site_root}/news/{year}/",
        f"{site_root}/news/",
        f"{site_root}/",
    ]
    for url in archives:
        code, html = fetch(url)
        time.sleep(SLEEP_BETWEEN)
        if code != 200 or not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            if _looks_like_match_report(a["href"], opponent, date):
                href = a["href"]
                if href.startswith("/"):
                    href = site_root + href
                if href.startswith("http"):
                    return href
    return None


def scrape_club(folder: Path, team: str, opponent: str,
                date: str) -> dict:
    out_name = f"club_report_{slug(team)}.txt"
    out_path = folder / "groundtruth_sources" / out_name
    site     = CLUB_SITES.get(team)
    if not site:
        return {"status": "not_found",
                "reason": f"no site root mapped for '{team}'"}
    url = _find_club_report_url(site, opponent, date)
    if not url:
        return {"status": "not_found",
                "reason": f"no match-report link on {site}"}
    code, html = fetch(url)
    time.sleep(SLEEP_BETWEEN)
    if code != 200 or not html:
        return {"status": "blocked_or_404", "url": url,
                "error": f"HTTP {code}"}
    text = extract_text(html, [
        "article", ".article-body", ".news-article", ".article__body", "main",
    ])
    if not text:
        return {"status": "no_content", "url": url}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {"status": "ok", "url": url, "bytes": len(text)}


ARCTIC_POSTS    = "https://arctic-shift.photon-reddit.com/api/posts/search"
ARCTIC_COMMENTS = "https://arctic-shift.photon-reddit.com/api/comments/search"


def _epoch(date_str: str, day_offset: int) -> int:
    dt = (datetime.strptime(date_str, "%Y-%m-%d")
                  .replace(tzinfo=timezone.utc))
    return int(dt.timestamp()) + day_offset * 86400


def _arctic_get(url: str, params: dict) -> list[dict]:
    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": UA},
                         timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        if isinstance(data, dict):
            return data.get("data") or []
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _link_id_for(post: dict) -> str | None:
    pid = post.get("id") or post.get("name")
    if not pid:
        return None
    pid = str(pid)
    return pid if pid.startswith("t3_") else f"t3_{pid}"


def scrape_reddit(folder: Path, date: str, t1: str, t2: str) -> dict:
    out_path = folder / "groundtruth_sources" / "reddit_thread.json"
    after    = _epoch(date, -1)
    before   = _epoch(date, +1)
    posts: list[dict] = []
    for sub in ("Championship", "soccer"):
        for q in (t1, t2):
            posts.extend(_arctic_get(ARCTIC_POSTS, {
                "subreddit": sub, "after": after, "before": before,
                "q": q, "limit": 25,
            }))
            time.sleep(SLEEP_BETWEEN)
    seen, unique = set(), []
    for p in posts:
        key = p.get("id") or p.get("permalink")
        if key and key not in seen:
            seen.add(key); unique.append(p)
    threads, total_comments = [], 0
    for p in unique:
        link_id = _link_id_for(p)
        if not link_id:
            continue
        comments_raw = _arctic_get(ARCTIC_COMMENTS, {"link_id": link_id, "limit": 500})
        time.sleep(SLEEP_BETWEEN)
        comments = [{
            "author"     : c.get("author"),
            "created_utc": c.get("created_utc"),
            "body"       : c.get("body", ""),
        } for c in comments_raw if c.get("body")]
        threads.append({
            "post_title": p.get("title", ""),
            "post_url"  : p.get("permalink") or p.get("url", ""),
            "comments"  : comments,
        })
        total_comments += len(comments)
    if not threads:
        return {"status": "not_found",
                "reason": "no posts in Arctic Shift window"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(threads, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return {"status": "ok",
            "post_count":    len(threads),
            "comment_count": total_comments}


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

def process_match(folder: Path, date: str, t1: str, t2: str,
                  sources: set[str], bbc_ids: dict[str, int]) -> dict:
    print(f"\n{'─'*72}")
    print(f"  MATCH : {folder.name}")
    print(f"{'─'*72}")
    out_root = folder / "groundtruth_sources"
    out_root.mkdir(parents=True, exist_ok=True)
    status: dict = {}

    if "sky" in sources:
        print("  [sky] scraping …")
        status["sky"] = scrape_sky(folder)
        print(f"        → {status['sky']}")

    if "bbc" in sources:
        print("  [bbc] scraping …")
        status["bbc"] = scrape_bbc(folder, bbc_ids)
        print(f"        → {status['bbc']}")

    if "espn" in sources:
        print("  [espn] scraping …")
        status["espn"] = scrape_espn(folder, date, t1, t2)
        print(f"        → {status['espn']}")

    if "club" in sources:
        print(f"  [club:{t1}] scraping …")
        status[f"club_{slug(t1)}"] = scrape_club(folder, t1, t2, date)
        print(f"        → {status[f'club_{slug(t1)}']}")
        print(f"  [club:{t2}] scraping …")
        status[f"club_{slug(t2)}"] = scrape_club(folder, t2, t1, date)
        print(f"        → {status[f'club_{slug(t2)}']}")

    if "reddit" in sources:
        print("  [reddit] scraping …")
        status["reddit"] = scrape_reddit(folder, date, t1, t2)
        print(f"        → {status['reddit']}")

    (out_root / "sources_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return status


# ════════════════════════════════════════════════════════════════════
# AGGREGATE TABLE
# ════════════════════════════════════════════════════════════════════

def _short_name(folder_name: str) -> str:
    """'2019-10-01 - Blackburn Rovers - Nottingham Forest'
       → 'Blackburn Rovers - Nottingham Forest'"""
    parts = folder_name.split(" - ", 1)
    return parts[1] if len(parts) == 2 else folder_name


def print_aggregate(rows: list[tuple[str, dict]]):
    print(f"\n{'═'*82}")
    print("  AGGREGATE — ground truth source coverage")
    print(f"{'═'*82}")
    print(f"  {'Match':<42} {'Sky':>4} {'BBC':>4} {'ESPN full':<18}")
    print(f"  {'─'*42} {'─'*4} {'─'*4} {'─'*18}")

    sky_ok = bbc_ok = espn_ok = 0
    sky_chars = bbc_chars = espn_lines = 0
    for name, status in rows:
        sky  = status.get("sky")  or {}
        bbc  = status.get("bbc")  or {}
        espn = status.get("espn") or {}
        sky_mark = "✓" if sky.get("status")  == "ok" else "—"
        bbc_mark = "✓" if bbc.get("status")  == "ok" else "—"
        if espn.get("status") == "ok":
            espn_mark   = f"✓ ({espn.get('lines', 0)} lines)"
            espn_ok    += 1
            espn_lines += int(espn.get("lines", 0))
        else:
            espn_mark = "—"
        if sky.get("status") == "ok":
            sky_ok    += 1
            sky_chars += int(sky.get("bytes", 0))
        if bbc.get("status") == "ok":
            bbc_ok    += 1
            bbc_chars += int(bbc.get("bytes", 0))
        print(f"  {_short_name(name)[:42]:<42} "
              f"{sky_mark:>4} {bbc_mark:>4} {espn_mark:<18}")

    total = len(rows)
    print(f"  {'─'*42} {'─'*4} {'─'*4} {'─'*18}")
    print(f"  {'Total':<42} {f'{sky_ok}/{total}':>4} "
          f"{f'{bbc_ok}/{total}':>4} {f'{espn_ok}/{total}':<18}")

    print(f"\n  Total raw commentary fetched:")
    print(f"    Sky reports     : {sky_chars:>9,} chars")
    print(f"    BBC reports     : {bbc_chars:>9,} chars")
    print(f"    ESPN commentary : {espn_lines:>9,} entries")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Scrape Sky / BBC / ESPN-full commentary for the "
                    "7 EFL Championship matches of 2019-10-01.",
    )
    ap.add_argument("--match",  help="Partial match-folder name filter")
    ap.add_argument("--source", choices=["sky", "bbc", "espn",
                                         "club", "reddit"],
                    help="Only scrape this single source (default: "
                         "sky + bbc + espn)")
    args = ap.parse_args()

    matches = find_matches(args.match)
    if not matches:
        print(f"No match folders found under {DATA_DIR}.")
        sys.exit(1)
    sources = ({args.source} if args.source
               else {"sky", "bbc", "espn"})

    print(f"Found {len(matches)} match folder(s).")
    print(f"Sources: {', '.join(sorted(sources))}")

    # BBC: run discovery probe ONCE up-front for any matches whose
    # ID isn't in the hardcoded table — then pass the merged map into
    # each per-match call.
    bbc_ids = dict(BBC_KNOWN_IDS)
    if "bbc" in sources:
        unknown = [(f.name, t1, t2) for f, _, t1, t2 in matches
                   if _lookup_by_folder(f.name, bbc_ids) is None]
        bbc_ids.update(discover_bbc_ids(unknown))

    rows = []
    for folder, date, t1, t2 in matches:
        status = process_match(folder, date, t1, t2, sources, bbc_ids)
        rows.append((folder.name, status))

    print_aggregate(rows)


if __name__ == "__main__":
    main()
