"""
scrape_ground_truth_sources.py
─────────────────────────────────────────────────────────────────────
Scrape ground-truth football commentary for the 7 EFL Championship
matches played on 2019-10-01, from four web sources:

    1. Sky Sports match report
    2. BBC Sport match report
    3. Each team's official club website
    4. Reddit thread comments (Arctic Shift archive API)

Output (per match folder under data/<match>/groundtruth_sources/):
    sky_report.txt          — Sky Sports article body
    bbc_report.txt          — BBC Sport article body
    club_report_<slug>.txt  — one per team's official site
    reddit_thread.json      — Reddit posts + comments
    sources_status.json     — which sources succeeded/failed

Every network call is wrapped in try/except. The script never crashes
on a single bad URL; it logs the failure to sources_status.json and
moves on to the next source/match.

CLI:
    python src/commentator/scrape_ground_truth_sources.py
    python src/commentator/scrape_ground_truth_sources.py --match "Blackburn"
    python src/commentator/scrape_ground_truth_sources.py --source sky
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
SLEEP_BETWEEN = 0.6   # polite gap between every outbound request


# ════════════════════════════════════════════════════════════════════
# TEAM METADATA — site roots, Sky/BBC ID overrides
# ════════════════════════════════════════════════════════════════════

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

# Sky Sports uses short, inconsistent abbreviations. Where unknown we
# fall back to lowercased-hyphenated full name.
SKY_SLUGS = {
    "Nottingham Forest"   : "n-forest",
    "Blackburn Rovers"    : "blackburn",
    "Brentford"           : "brentford",
    "Bristol City"        : "bristol-city",
    "Hull City"           : "hull",
    "Sheffield Wednesday" : "sheff-wed",
    "Leeds United"        : "leeds",
    "West Bromwich Albion": "west-brom",
    "West Bromwich"       : "west-brom",
    "Middlesbrough"       : "middlesbrough",
    "Preston North End"   : "preston",
    "Reading"             : "reading",
    "Fulham"              : "fulham",
    "Stoke City"          : "stoke",
    "Huddersfield Town"   : "huddersfield",
}

# Sky Sports / BBC game IDs verified by hand. Sky URL = .../<a-vs-b>/<id>;
# BBC URL = .../sport/football/<id>. Add more here as you find them.
SKY_KNOWN_IDS = {
    frozenset(("Blackburn Rovers", "Nottingham Forest")): "409456",
}
BBC_KNOWN_IDS = {
    frozenset(("Leeds United", "West Bromwich Albion")): "49805857",
    frozenset(("Leeds United", "West Bromwich"))       : "49805857",
    frozenset(("Stoke City",   "Huddersfield Town"))   : "49805862",
}


# ════════════════════════════════════════════════════════════════════
# COMMON UTILITIES
# ════════════════════════════════════════════════════════════════════

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_match_folder(folder: Path):
    """'YYYY-MM-DD - Team1 - Team2' → (date, t1, t2) or None."""
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
    """GET a URL; return (status_code, text) or (0, None) on exception."""
    try:
        r = requests.get(url, headers=HDR, timeout=REQ_TIMEOUT)
        return r.status_code, (r.text if r.status_code == 200 else None)
    except Exception:
        return 0, None


def extract_text(html: str, selectors: list[str]) -> str | None:
    """Try CSS selectors in order; return first match's text or None."""
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


# ════════════════════════════════════════════════════════════════════
# SOURCE 1 — Sky Sports
# ════════════════════════════════════════════════════════════════════

def _sky_slug(team: str) -> str:
    return SKY_SLUGS.get(team, team.lower().replace(" ", "-"))


def _sky_candidate_urls(t1: str, t2: str) -> list[str]:
    """Build URLs for both home/away orderings if we have an ID."""
    s1, s2 = _sky_slug(t1), _sky_slug(t2)
    key    = frozenset((t1, t2))
    gid    = SKY_KNOWN_IDS.get(key)
    if not gid:
        return []
    return [
        f"https://www.skysports.com/football/{s1}-vs-{s2}/{gid}",
        f"https://www.skysports.com/football/{s2}-vs-{s1}/{gid}",
    ]


def scrape_sky(folder: Path, t1: str, t2: str) -> dict:
    out_path = folder / "groundtruth_sources" / "sky_report.txt"
    urls     = _sky_candidate_urls(t1, t2)
    if not urls:
        return {"status": "not_found",
                "reason": "no Sky ID hardcoded — add to SKY_KNOWN_IDS"}
    for url in urls:
        code, html = fetch(url)
        time.sleep(SLEEP_BETWEEN)
        if code != 200 or not html:
            continue
        text = extract_text(html, [
            ".sdc-article-body",
            "article",
            ".article-body",
        ])
        if not text:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return {"status": "ok", "url": url, "bytes": len(text)}
    return {"status": "blocked_or_404", "tried": urls}


# ════════════════════════════════════════════════════════════════════
# SOURCE 2 — BBC Sport
# ════════════════════════════════════════════════════════════════════

def _bbc_candidate_urls(t1: str, t2: str) -> list[str]:
    key = frozenset((t1, t2))
    gid = BBC_KNOWN_IDS.get(key)
    if not gid:
        return []
    return [f"https://www.bbc.co.uk/sport/football/{gid}"]


def scrape_bbc(folder: Path, t1: str, t2: str) -> dict:
    out_path = folder / "groundtruth_sources" / "bbc_report.txt"
    urls     = _bbc_candidate_urls(t1, t2)
    if not urls:
        return {"status": "not_found",
                "reason": "no BBC article ID hardcoded — add to BBC_KNOWN_IDS"}
    for url in urls:
        code, html = fetch(url)
        time.sleep(SLEEP_BETWEEN)
        if code == 403:
            return {"status": "blocked", "url": url, "error": "403"}
        if code != 200 or not html:
            continue
        text = extract_text(html, [
            "article",
            "main",
            "[data-component='text-block']",
        ])
        if not text:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return {"status": "ok", "url": url, "bytes": len(text)}
    return {"status": "blocked_or_404", "tried": urls}


# ════════════════════════════════════════════════════════════════════
# SOURCE 3 — Club official sites
# ════════════════════════════════════════════════════════════════════

def _looks_like_match_report(href: str, opponent: str, date: str) -> bool:
    """
    Heuristic: a club's match-report link almost always contains either
    'match-report' or 'highlights' AND mentions the opponent (or one of
    its long words) somewhere in the URL.
    """
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
                "reason": f"no plausible match-report link on {site}"}
    code, html = fetch(url)
    time.sleep(SLEEP_BETWEEN)
    if code != 200 or not html:
        return {"status": "blocked_or_404", "url": url,
                "error": f"HTTP {code}"}
    text = extract_text(html, [
        "article",
        ".article-body",
        ".news-article",
        ".article__body",
        "main",
    ])
    if not text:
        return {"status": "no_content", "url": url}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {"status": "ok", "url": url, "bytes": len(text)}


# ════════════════════════════════════════════════════════════════════
# SOURCE 4 — Reddit (Arctic Shift archive)
# ════════════════════════════════════════════════════════════════════

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
    """Reddit's comments-search wants the post's full name 't3_<id>'."""
    pid = post.get("id") or post.get("name")
    if not pid:
        return None
    pid = str(pid)
    return pid if pid.startswith("t3_") else f"t3_{pid}"


def scrape_reddit(folder: Path, date: str, t1: str, t2: str) -> dict:
    out_path = folder / "groundtruth_sources" / "reddit_thread.json"
    after    = _epoch(date, -1)
    before   = _epoch(date, +1)

    # Search both r/Championship and r/soccer for either team name.
    posts: list[dict] = []
    for sub in ("Championship", "soccer"):
        for q in (t1, t2):
            posts.extend(_arctic_get(ARCTIC_POSTS, {
                "subreddit": sub,
                "after"    : after,
                "before"   : before,
                "q"        : q,
                "limit"    : 25,
            }))
            time.sleep(SLEEP_BETWEEN)

    # Dedup posts by id/permalink
    seen, unique = set(), []
    for p in posts:
        key = p.get("id") or p.get("permalink")
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    threads, total_comments = [], 0
    for p in unique:
        link_id = _link_id_for(p)
        if not link_id:
            continue
        comments_raw = _arctic_get(ARCTIC_COMMENTS, {
            "link_id": link_id,
            "limit"  : 500,
        })
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
    return {"status"       : "ok",
            "post_count"   : len(threads),
            "comment_count": total_comments}


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

def process_match(folder: Path, date: str, t1: str, t2: str,
                  sources: set[str]) -> dict:
    print(f"\n{'─'*72}")
    print(f"  MATCH : {folder.name}")
    print(f"{'─'*72}")
    out_root = folder / "groundtruth_sources"
    out_root.mkdir(parents=True, exist_ok=True)
    status: dict = {}

    if "sky" in sources:
        print("  [sky] scraping …")
        status["sky"] = scrape_sky(folder, t1, t2)
        print(f"        → {status['sky']}")

    if "bbc" in sources:
        print("  [bbc] scraping …")
        status["bbc"] = scrape_bbc(folder, t1, t2)
        print(f"        → {status['bbc']}")

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

def _check(s: dict | None) -> str:
    if not s:
        return "-"
    return "✓" if s.get("status") == "ok" else "✗"


def print_aggregate(rows: list[tuple[str, dict]]):
    print(f"\n{'═'*82}")
    print("  AGGREGATE — ground truth source coverage")
    print(f"{'═'*82}")
    print(f"  {'Match':<40} {'Sky':>4} {'BBC':>4} {'Club×2':>8} {'Reddit':>10}")
    print(f"  {'─'*40} {'─'*4} {'─'*4} {'─'*8} {'─'*10}")
    for name, status in rows:
        sky       = _check(status.get("sky"))
        bbc       = _check(status.get("bbc"))
        club_keys = [k for k in status if k.startswith("club_")]
        club_chk  = " ".join(_check(status[k]) for k in club_keys) or "-"
        red       = status.get("reddit", {})
        red_str   = (f"✓ ({red.get('comment_count', 0)})"
                     if red.get("status") == "ok" else "✗")
        print(f"  {name[:40]:<40} {sky:>4} {bbc:>4} {club_chk:>8} {red_str:>10}")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Scrape ground-truth football commentary "
                    "from Sky, BBC, club sites, and Reddit.",
    )
    ap.add_argument("--match",  help="Partial match-folder name filter")
    ap.add_argument("--source", choices=["sky", "bbc", "club", "reddit"],
                    help="Only scrape this single source (default: all)")
    args = ap.parse_args()

    matches = find_matches(args.match)
    if not matches:
        print(f"No match folders found under {DATA_DIR}.")
        sys.exit(1)
    sources = {args.source} if args.source else {"sky", "bbc", "club", "reddit"}

    print(f"Found {len(matches)} match folder(s).")
    print(f"Sources: {', '.join(sorted(sources))}")

    rows = []
    for folder, date, t1, t2 in matches:
        status = process_match(folder, date, t1, t2, sources)
        rows.append((folder.name, status))

    print_aggregate(rows)


if __name__ == "__main__":
    main()
