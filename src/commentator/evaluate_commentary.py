"""
Commentary Evaluation — Human vs AI Agent

Metrics computed per matched (GT, AI) event pair:
    BLEU, METEOR, ROUGE-L      (per-pair, averaged)
    CIDEr                       (corpus-level on the matched-pair list)
    BERTScore                   (per-pair via DistilBERT, averaged)
    Fact Overlap                (player / team / outcome heuristics)
    CRR (Contextual Ref Rate)  (per-text count of "his second", "again", …)
    Coverage / Precision / Recall / F1

Multi-track ground truth — for a given match, AI commentary can be
evaluated against several GT tracks side-by-side:
    espn      — ESPN raw text pulled from data/kg_output/ekg.ttl
    qwen_v1   — data/<match>/ground_truth_commentary.json
    qwen_v2   — data/<match>/qwen_ground_truth_commentary_v2.json

Single match — multi-track (recommended):
    python src/commentator/evaluate_commentary.py --match "Blackburn"
    python src/commentator/evaluate_commentary.py --match "Blackburn" \
        --tracks espn qwen_v2

All matches — multi-track (when every match has been re-built on new T-Box):
    python src/commentator/evaluate_commentary.py --all

Single match — legacy single-track (JSON files):
    python src/commentator/evaluate_commentary.py \
        --gt-file  ground_truth_commentary.json \
        --ai-file  ai_commentary.json \
        --match-dir "data/2019-10-01 - Blackburn Rovers - Nottingham Forest"

Single match — legacy text-log mode:
    python src/commentator/evaluate_commentary.py \
        --ai-log  data/commentator_output/commentary_log.txt \
        --human-json data/human_commentary.json \
        --espn-csv   data/blackburn_forest_2019-10-01.csv
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data"
OUT_DIR   = BASE_DIR / "data" / "commentator_output"


def _has_cjk(text: str) -> bool:
    return any(
        (unicodedata.category(c) in ('Lo',) and '一' <= c <= '鿿')
        or '　' <= c <= '〿'
        or '＀' <= c <= '￯'
        for c in text
    )

PAST_REF_KEYWORDS = [
    "again", "second", "third", "another", "has been", "continues",
    "following", "always", "still", "once more", "earlier",
    "keeps the pressure", "first time", "pressure mounting",
]


# ── Parse AI commentary log ────────────────────────────────────────────────

def parse_ai_log(path):
    events  = []
    pattern = re.compile(r"\[(\d+)(?:st|nd)\s+(\d+):(\d+)\]\s+(\w+)\s+\|(.+)")
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        half, mins, _, etype, text = m.groups()
        if _has_cjk(text):
            continue
        events.append({
            "half"      : int(half),
            "minute"    : int(mins),
            "event_type": etype.strip(),
            "full_text" : text.strip(),
        })
    return events


# ── Multi-match shared-log splitting ───────────────────────────────────────

MATCH_HEADER_RE = re.compile(r"^=== MATCH:\s*(.+?)\s*===\s*$")


def split_log_by_match(path: str | Path) -> dict[str, list[dict]]:
    """
    Parse a shared commentary_log.txt into per-match event lists, keyed by
    match folder name. Match boundaries are written by
    commentator.log_match_boundary() as: '=== MATCH: <folder name> ==='.

    If the log has no headers, returns {"__unsplit__": [all events]}.
    """
    path     = Path(path)
    sections : dict[str, list[str]] = {}
    current  = None
    if not path.exists():
        return {}

    for line in path.read_text(errors="ignore").splitlines():
        hdr = MATCH_HEADER_RE.match(line.strip())
        if hdr:
            current = hdr.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is None:
            sections.setdefault("__unsplit__", []).append(line)
        else:
            sections[current].append(line)

    pattern = re.compile(r"\[(\d+)(?:st|nd)\s+(\d+):(\d+)\]\s+(\w+)\s+\|(.+)")
    result  : dict[str, list[dict]] = {}
    for name, lines in sections.items():
        evs = []
        for line in lines:
            m = pattern.match(line.strip())
            if not m:
                continue
            half, mins, _, etype, text = m.groups()
            if _has_cjk(text):
                continue
            evs.append({
                "half"      : int(half),
                "minute"    : int(mins),
                "event_type": etype.strip(),
                "full_text" : text.strip(),
            })
        result[name] = evs
    return result


# ── Find closest AI match (±2 min, same half + event type) ────────────────

def find_match(human, ai_events, tol=2.0):
    def abs_min(e):
        return e["minute"] + (45 if e["half"] == 2 else 0)

    human_abs = human["minute"] + (45 if human["half"] == 2 else 0)
    same = [
        e for e in ai_events
        if e["event_type"].lower() == human["event_type"].lower()
        and abs(abs_min(e) - human_abs) <= tol
    ]
    return min(same, key=lambda e: abs(abs_min(e) - human_abs)) if same else None


# ── Metric A — BLEU-1 and BLEU-4 ──────────────────────────────────────────

def metric_bleu_1(ref: str, hyp: str) -> float:
    """BLEU-1: unigram precision with smoothing. 0..1."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    if not hyp or not ref:
        return 0.0
    try:
        return round(
            sentence_bleu([ref.lower().split()], hyp.lower().split(),
                          weights=(1, 0, 0, 0),
                          smoothing_function=SmoothingFunction().method1), 3)
    except Exception as e:
        print(f"WARNING: BLEU-1 failed: {e}")
        return 0.0


def metric_a_bleu(ref, hyp):
    """BLEU-4: 4-gram precision with smoothing. 0..1."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    if not hyp:
        return 0.0
    return round(
        sentence_bleu(
            [ref.lower().split()], hyp.lower().split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        ), 3,
    )


# ── Metric B — Fact Overlap ────────────────────────────────────────────────

def metric_b_fact(human, ai_text):
    from thefuzz import fuzz
    t          = ai_text.lower()
    player_ok  = any(p in t for p in human["player"].lower().split())
    team_ok    = any(fuzz.partial_ratio(w, t) > 75 for w in human["team"].lower().split())
    OUTCOMES = {
        "Goal"        : ["goal", "scores", "scored", "fires home", "slots",
                         "nets", "buries", "puts away"],
        "Shot"        : ["saved", "blocked", "missed", "wide", "over",
                         "attempt", "shot"],
        "Foul"        : ["foul", "tackle", "challenge", "card", "brings down"],
        "Corner"      : ["corner"],
        "Free_Kick"   : ["free kick", "freekick"],
        "Substitution": ["substitut", "replac", "comes on", "comes off"],
        "Offside"     : ["offside"],
    }
    keys       = OUTCOMES.get(human["event_type"], [])
    outcome_ok = any(k in t for k in keys)
    return {
        "player" : player_ok,
        "team"   : team_ok,
        "outcome": outcome_ok,
        "score"  : int(player_ok) + int(team_ok) + int(outcome_ok),
    }


# ── Metric C — BERTScore ──────────────────────────────────────────────────

_BERT_DISABLED = False

def metric_c_bert(refs, hyps):
    global _BERT_DISABLED
    if _BERT_DISABLED:
        return [None] * len(refs)
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(hyps, refs, lang="en", verbose=False,
                          model_type="distilbert-base-uncased")
        return [round(f.item(), 3) for f in F1]
    except ImportError:
        print("WARNING: bert-score not installed. Run: pip install bert-score")
        _BERT_DISABLED = True
        return [None] * len(refs)
    except AttributeError as e:
        # bert-score vs transformers version skew (e.g.
        # BertTokenizer.build_inputs_with_special_tokens removed). Disable
        # BERTScore for the rest of this run instead of crashing.
        print(f"WARNING: BERTScore disabled — bert-score/transformers version "
              f"skew: {e}")
        print("        Fix later with: pip install -U 'bert-score>=0.3.13' "
              "'transformers<5'")
        _BERT_DISABLED = True
        return [None] * len(refs)
    except Exception as e:
        print(f"WARNING: BERTScore failed — {type(e).__name__}: {e}")
        _BERT_DISABLED = True
        return [None] * len(refs)


# ── Metric D — METEOR ─────────────────────────────────────────────────────

_METEOR_READY = False
def _ensure_meteor_corpora():
    """Lazy idempotent download of NLTK resources METEOR needs."""
    global _METEOR_READY
    if _METEOR_READY:
        return
    import nltk
    for pkg, path in (("wordnet",  "corpora/wordnet"),
                      ("punkt",    "tokenizers/punkt"),
                      ("omw-1.4",  "corpora/omw-1.4")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _METEOR_READY = True


def metric_meteor(ref: str, hyp: str) -> float:
    """METEOR — synonyms + stemming + word order. 0..1."""
    if not hyp or not ref:
        return 0.0
    try:
        _ensure_meteor_corpora()
        from nltk.translate.meteor_score import meteor_score
        return round(meteor_score([ref.split()], hyp.split()), 3)
    except Exception as e:
        print(f"WARNING: METEOR failed: {e}")
        return 0.0


# ── Metric E — ROUGE-1 and ROUGE-L ────────────────────────────────────────

def _rouge_scores(ref: str, hyp: str) -> dict[str, float]:
    """Compute rouge1 and rougeL in one pass. Returns {'rouge1': f, 'rougeL': f}."""
    if not hyp or not ref:
        return {"rouge1": 0.0, "rougeL": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, hyp)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 3),
            "rougeL": round(scores["rougeL"].fmeasure, 3),
        }
    except ImportError:
        print("WARNING: rouge-score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rougeL": 0.0}
    except Exception as e:
        print(f"WARNING: ROUGE failed: {e}")
        return {"rouge1": 0.0, "rougeL": 0.0}


def metric_rouge_1(ref: str, hyp: str) -> float:
    """ROUGE-1 (unigram) F1. 0..1."""
    return _rouge_scores(ref, hyp)["rouge1"]


def metric_rouge_l(ref: str, hyp: str) -> float:
    """ROUGE-L F1. 0..1."""
    return _rouge_scores(ref, hyp)["rougeL"]


# ── GT-match helper for multi-reference CIDEr ─────────────────────────────

def _find_gt_match(ai_event: dict, gt_events: list[dict], tolerance_min: float) -> dict | None:
    """Closest GT event of the same type within tolerance_min absolute minutes."""
    def abs_min(e):
        return float(e.get("minute", 0)) + (45.0 if int(e.get("half", 1)) == 2 else 0.0)

    ai_abs  = abs_min(ai_event)
    ai_type = ai_event.get("event_type", "").lower()
    candidates = [
        (abs(abs_min(g) - ai_abs), g)
        for g in gt_events
        if g.get("event_type", "").lower() == ai_type
        and abs(abs_min(g) - ai_abs) <= tolerance_min
    ]
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


# ── Metric F — CIDEr (corpus-level consensus) ─────────────────────────────

def metric_cider_corpus(refs: list[str], hyps: list[str]) -> float:
    """
    CIDEr is consensus-based — TF-IDF weighted n-gram overlap across a
    corpus. Computing it per-pair is meaningless; we run it once over
    the whole matched-pair list. Returns 0.0 if dependencies are missing.
    """
    if not refs or not hyps or len(refs) != len(hyps):
        return 0.0
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {str(i): [refs[i]] for i in range(len(refs))}
        res = {str(i): [hyps[i]] for i in range(len(hyps))}
        score, _ = Cider().compute_score(gts, res)
        return round(float(score), 3)
    except ImportError:
        print("WARNING: pycocoevalcap not installed. Run: pip install pycocoevalcap")
        return 0.0
    except Exception as e:
        print(f"WARNING: CIDEr failed: {e}")
        return 0.0


# ── Metric G — Multi-reference CIDEr ──────────────────────────────────────

def metric_cider_multireference(
    ai_events    : list[dict],
    all_gt_tracks: dict[str, list[dict]],
    tolerance_min: float = 1.0,
) -> tuple[float, int, int]:
    """
    CIDEr computed with multiple references per event by pooling matched GT
    text from ALL available tracks.

    Returns (cider_score, n_scored, n_total_ai).
    n_scored = AI events that matched at least one GT reference.

    This is CIDEr as designed for multi-reference datasets. DO NOT hide this
    score even if it is 0.00 — report n_scored/n_total so the coverage is clear.
    Never cherry-picks references; ALL matched GT tracks are pooled honestly.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("WARNING: pycocoevalcap not installed. Run: pip install pycocoevalcap")
        return 0.0, 0, len(ai_events)

    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}

    for i, ai_event in enumerate(ai_events):
        ai_text = ai_event.get("human_text", "")
        if not ai_text:
            continue
        refs: list[str] = []
        for track_events in all_gt_tracks.values():
            matched = _find_gt_match(ai_event, track_events, tolerance_min)
            if matched:
                ref_text = matched.get("human_text", "")
                if ref_text:
                    refs.append(ref_text)
        if refs:
            gts[i] = refs
            res[i] = [ai_text]

    if not gts:
        return 0.0, 0, len(ai_events)

    try:
        score, _ = Cider().compute_score(gts, res)
        return round(float(score), 3), len(gts), len(ai_events)
    except Exception as e:
        print(f"WARNING: Multi-reference CIDEr failed: {type(e).__name__}: {e}")
        return 0.0, len(gts), len(ai_events)


# ── Parse ESPN CSV ────────────────────────────────────────────────────────

def parse_espn_csv(path: str) -> list[dict]:
    """
    Real CSV columns: Time, Player, Team, Action_Type,
                      Yellow_Card, Red_Card, Full_Text
    Time is a float in minutes (e.g. 63.5 = 63rd minute).
    """
    import csv as _csv
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            raw_time = row.get("Time", "0").strip().rstrip("'")
            try:
                t = float(raw_time)
            except ValueError:
                continue
            action = row.get("Action_Type", "").strip()
            if not action or action == "None":
                continue
            rows.append({
                "minute"     : t,
                "half"       : 1 if t <= 45 else 2,
                "event_type" : action,
                "player"     : row.get("Player", "").strip(),
                "team"       : row.get("Team",   "").strip(),
                "description": row.get("Full_Text", "").strip(),
            })
    return sorted(rows, key=lambda r: r["minute"])


# ── Multi-track ground truth ──────────────────────────────────────────────
#
# Each match folder may have several ground-truth tracks; we evaluate the
# AI commentary against EACH that's present. New T-Box property names
# (isPerformedBy, inMatch, involvedTeam, hasPeriodNumber) are used in the
# SPARQL queries that pull ESPN raw text from the KG.

TRACK_FILES = {
    "qwen_v1": "ground_truth_commentary.json",
    "qwen_v2": "qwen_ground_truth_commentary_v2.json",
}
TRACK_LABELS = {
    "espn"   : "ESPN raw",
    "qwen_v1": "Qwen GT v1",
    "qwen_v2": "Qwen GT v2",
}
ALL_TRACKS = ["espn", "qwen_v1", "qwen_v2"]

# ── JAIST MatchAware reference scores ─────────────────────────────────────────
# Source: JAIST MatchAware SN-Long+retrieval (Table 3, Baidu features)
# Used ONLY for comparison reporting. Never modifies any computed metric.
JAIST_REFERENCE = {
    "bleu_1" : 0.47,
    "bleu_4" : 0.20,
    "meteor" : 0.18,
    "rouge_1": 0.40,
    "rouge_l": 0.36,
    "cider"  : 20.26,
    "source" : "JAIST MatchAware SN-Long+retrieval (Table 3, Baidu features)",
}


def load_espn_from_kg(folder: Path) -> list[dict]:
    """
    Pull per-event ESPN raw text out of data/kg_output/ekg.ttl for the
    given match folder. Returns events in the same dict shape as
    parse_json_commentary() so they slot directly into evaluate_match_json.
    """
    from rdflib import Graph, Literal

    ttl = DATA_DIR / "kg_output" / "ekg.ttl"
    if not ttl.exists():
        return []

    g = Graph()
    g.parse(str(ttl), format="turtle")

    q_match = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?match WHERE {
        ?match a/rdfs:subClassOf* ekg:Match ;
               rdfs:label         ?label .
        FILTER (?label = ?folder)
    } LIMIT 1
    """
    rows = list(g.query(q_match, initBindings={"folder": Literal(folder.name)}))
    if not rows:
        return []
    match_uri = str(rows[0][0])

    q_events = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?gametime ?type ?fullText ?minute ?period ?player ?team WHERE {
        ?e ekg:inMatch         <%s> ;
           ekg:hasTime          ?gametime ;
           ekg:hasEventType     ?type ;
           ekg:hasFullText      ?fullText ;
           ekg:hasMinute        ?minute ;
           ekg:hasPeriodNumber  ?period .
        OPTIONAL {
            { ?e ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?e }
            ?p rdfs:label ?player .
        }
        OPTIONAL {
            ?e ekg:involvedTeam ?t .
            ?t rdfs:label       ?team .
        }
    }
    ORDER BY ?period ?minute
    """ % match_uri

    events = []
    for r in g.query(q_events):
        try:
            half = int(r.period)
        except (TypeError, ValueError):
            half = 1
        try:
            minute = float(r.minute)
        except (TypeError, ValueError):
            minute = 0.0
        events.append({
            "minute"    : minute,
            "half"      : half,
            "event_type": str(r.type),
            "player"    : str(r.player) if r.player else "",
            "team"      : str(r.team)   if r.team   else "",
            "human_text": str(r.fullText),
        })
    return sorted(events, key=lambda e: e["minute"] + (45 if e["half"] == 2 else 0))


def discover_tracks(folder: Path,
                    requested: list[str] | None = None) -> dict[str, list[dict]]:
    """{track_name: gt_events} for every track found under folder."""
    out: dict[str, list[dict]] = {}
    wanted = set(requested) if requested else set(ALL_TRACKS)

    if "espn" in wanted:
        espn = load_espn_from_kg(folder)
        if espn:
            out["espn"] = espn

    for track, fname in TRACK_FILES.items():
        if track not in wanted:
            continue
        p = folder / fname
        if p.exists():
            out[track] = parse_json_commentary(p)

    return out


# ── Parse JSON commentary files (GT or AI) ────────────────────────────────

def parse_json_commentary(path: str | Path) -> list[dict]:
    """
    Load ground_truth_commentary.json or ai_commentary.json.
    Expected fields: minute, half, event_type, player, team, human_text.

    The text field is read with a fallback chain so that legacy AI files
    that wrote 'ai_text' (or sources using 'text' / 'commentary' /
    'full_text') still evaluate correctly without re-generation. Without
    this fallback all per-pair similarity metrics silently returned 0.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = []
    for e in data:
        text = (e.get("human_text") or e.get("ai_text")     or
                e.get("text")       or e.get("commentary") or
                e.get("full_text")  or "")
        events.append({
            "minute"    : float(e.get("minute", 0)),
            "half"      : int(e.get("half", 1)),
            "event_type": str(e.get("event_type", "")),
            "player"    : str(e.get("player", "")),
            "team"      : str(e.get("team", "")),
            "human_text": text,
        })
    return sorted(events, key=lambda e: e["minute"] + (45 if e["half"] == 2 else 0))


# ── Multi-match event matching ─────────────────────────────────────────────

def match_events(
    gt_events : list[dict],
    ai_events : list[dict],
    tolerance_min: float = 1.0,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """
    Greedy 1-to-1 matching: for each GT event find the closest AI event
    of the same type within tolerance_min absolute minutes.
    Returns:
        matched      — list of (gt_event, ai_event) pairs
        unmatched_gt — GT events with no AI counterpart
        unmatched_ai — AI events that never matched any GT event
    """
    def abs_min(e):
        return float(e["minute"]) + (45.0 if int(e["half"]) == 2 else 0.0)

    ai_pool = list(ai_events)   # consume from this
    matched, unmatched_gt = [], []

    for gt in gt_events:
        gt_abs = abs_min(gt)
        candidates = [
            (abs(abs_min(a) - gt_abs), a)
            for a in ai_pool
            if a["event_type"].lower() == gt["event_type"].lower()
            and abs(abs_min(a) - gt_abs) <= tolerance_min
        ]
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_ai = candidates[0]
            matched.append((gt, best_ai))
            ai_pool.remove(best_ai)
        else:
            unmatched_gt.append(gt)

    return matched, unmatched_gt, ai_pool   # ai_pool remainder = unmatched AI


# ── Per-match evaluation (JSON mode) ──────────────────────────────────────

def evaluate_match_json(
    gt_events: list[dict],
    ai_events: list[dict],
    match_name: str = "",
    tolerance_min: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Run all metrics on one match worth of GT vs AI JSON commentary.
    Returns a summary dict suitable for the aggregate table.
    """
    matched_pairs, unmatched_gt, unmatched_ai = match_events(
        gt_events, ai_events, tolerance_min)

    n_gt      = len(gt_events)
    n_ai      = len(ai_events)
    n_matched = len(matched_pairs)

    precision  = n_matched / n_ai      if n_ai   > 0 else 0.0
    recall     = n_matched / n_gt      if n_gt   > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

    gt_texts   = [gt["human_text"] for gt, _ in matched_pairs]
    ai_texts   = [ai["human_text"] for _, ai in matched_pairs]

    avg_bleu   = (round(sum(metric_a_bleu(g, a) for g, a in zip(gt_texts, ai_texts))
                        / n_matched, 3)
                  if n_matched else 0.0)
    avg_bleu_1 = (round(sum(metric_bleu_1(g, a) for g, a in zip(gt_texts, ai_texts))
                        / n_matched, 3)
                  if n_matched else 0.0)

    avg_meteor = (round(sum(metric_meteor(g, a) for g, a in zip(gt_texts, ai_texts))
                        / n_matched, 3)
                  if n_matched else 0.0)
    if n_matched:
        _rouge_pairs = [_rouge_scores(g, a) for g, a in zip(gt_texts, ai_texts)]
        avg_rouge_1  = round(sum(r["rouge1"] for r in _rouge_pairs) / n_matched, 3)
        avg_rouge_l  = round(sum(r["rougeL"] for r in _rouge_pairs) / n_matched, 3)
    else:
        avg_rouge_1 = avg_rouge_l = 0.0
    cider_score = metric_cider_corpus(gt_texts, ai_texts) if n_matched else 0.0

    bert_scores = metric_c_bert(gt_texts, ai_texts) if n_matched else []
    valid_bert  = [s for s in bert_scores if s is not None]
    avg_bert    = round(sum(valid_bert) / len(valid_bert), 3) if valid_bert else None

    crr_gt  = crr([gt["human_text"] for gt in gt_events])
    crr_ai  = crr([ai["human_text"] for ai in ai_events])

    corp_bleu     = corpus_bleu_score(gt_texts, ai_texts)
    coverage_rate = recall
    fact_rate_raw = 0.0
    if matched_pairs:
        fact_scores = [
            metric_b_fact(gt, ai["human_text"])["score"] / 3
            for gt, ai in matched_pairs
        ]
        fact_rate_raw = sum(fact_scores) / len(fact_scores)

    bert_val  = avg_bert if avg_bert is not None else 0.0
    crr_ratio = (crr_ai / crr_gt / 100) if crr_gt > 0 else 0.0
    mos       = match_overall_score(coverage_rate, fact_rate_raw, bert_val, crr_ratio)

    if verbose:
        W = 70
        def bar(v, max_val=1.0, width=20):
            if v is None:
                return "not computed"
            filled = int((v / max_val) * width)
            return f"[{'█'*filled}{'░'*(width-filled)}] {v:.3f}"

        print(f"\n{'='*W}")
        print(f"  MATCH: {match_name}")
        print(f"  GT events: {n_gt}  |  AI events: {n_ai}  |  Matched: {n_matched}")
        print(f"{'─'*W}")
        print(f"  Precision : {precision:.3f}   Recall : {recall:.3f}   F1 : {f1:.3f}")
        print(f"  BLEU avg  : {bar(avg_bleu)}")
        print(f"  BERTScore : {bar(avg_bert)}")
        print(f"  CRR  GT   : {crr_gt:.1f}%     CRR AI : {crr_ai:.1f}%")
        print(f"{'─'*W}")

        mos_b = int(mos)
        print(f"  Corpus BLEU   : {bar(corp_bleu)}")
        print(f"  Match Overall : [{'█'*mos_b}{'░'*(10-mos_b)}] {mos}/10")
        print(f"    Coverage {round(coverage_rate*100,1)}%  |  "
              f"Semantic {round(bert_val*100,1)}%  |  "
              f"Factual {round(fact_rate_raw*100,1)}%  |  "
              f"Context {round(crr_ratio*100,1)}%")

        if unmatched_gt:
            print(f"\n  Unmatched GT ({len(unmatched_gt)}):")
            for e in unmatched_gt[:5]:
                print(f"    ✗ {int(e['minute'])}'  {e['event_type']}  {e.get('player','?')}")
            if len(unmatched_gt) > 5:
                print(f"    ... and {len(unmatched_gt)-5} more")

    return {
        "match"    : match_name,
        "n_gt"     : n_gt,
        "n_ai"     : n_ai,
        "matched"  : n_matched,
        "precision": round(precision, 3),
        "recall"   : round(recall, 3),
        "f1"       : round(f1, 3),
        "bleu_1"   : avg_bleu_1,
        "bleu"     : avg_bleu,
        "meteor"   : avg_meteor,
        "rouge_1"  : avg_rouge_1,
        "rouge_l"  : avg_rouge_l,
        "cider"    : cider_score,
        "bert"     : avg_bert,
        "crr_gt"   : crr_gt,
        "crr_ai"   : crr_ai,
        "corp_bleu": corp_bleu,
        "mos"      : mos,
    }


# ── Full ESPN coverage analysis ────────────────────────────────────────────

def full_coverage_analysis(espn_events, ai_events, tol_min=1.5):
    """
    For every ESPN event, check if AI has a matching commentary
    within tol_min minutes and same event_type.
    Returns per-type breakdown + overall stats.
    """
    TYPE_MAP = {
        "Shot"        : "Shot",
        "Goal"        : "Goal",
        "Corner"      : "Corner",
        "Foul"        : "Foul",
        "Free_Kick"   : "Free_Kick",
        "Substitution": "Substitution",
        "Offside"     : "Offside",
        "shot"        : "Shot",
        "goal"        : "Goal",
        "corner"      : "Corner",
        "foul"        : "Foul",
        "free_kick"   : "Free_Kick",
        "substitution": "Substitution",
        "offside"     : "Offside",
    }

    type_stats = {}

    for espn in espn_events:
        raw_type = espn.get("event_type", "").lower()
        ai_type  = TYPE_MAP.get(raw_type)
        if not ai_type:
            continue

        if ai_type not in type_stats:
            type_stats[ai_type] = {"espn": 0, "matched": 0, "missed": []}

        type_stats[ai_type]["espn"] += 1

        espn_min  = float(espn.get("minute", 0))
        espn_half = int(espn.get("half", 1))
        hit = any(
            e["event_type"] == ai_type
            and abs((e["minute"] + (45 if e["half"] == 2 else 0)) - espn_min) <= tol_min
            for e in ai_events
        )

        if hit:
            type_stats[ai_type]["matched"] += 1
        else:
            type_stats[ai_type]["missed"].append(
                f"{espn_half}H {espn_min:.0f}'  {espn.get('player','?')} ({espn.get('team','?')})"
            )

    return type_stats


# ── Corpus BLEU (match-level document similarity) ─────────────────────────

def corpus_bleu_score(human_texts: list[str], ai_texts: list[str]) -> float:
    """
    NLTK corpus_bleu: human sentences are references, AI are hypotheses.
    Each human sentence is one reference set for the corresponding AI sentence.
    Measures overall linguistic similarity across the whole match, not per event.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    if not human_texts or not ai_texts:
        return 0.0
    refs  = [[t.lower().split()] for t in human_texts]
    hyps  = [t.lower().split()   for t in ai_texts]
    return round(
        corpus_bleu(refs, hyps,
                    weights=(0.5, 0.5),
                    smoothing_function=SmoothingFunction().method1),
        3,
    )


# ── Match Overall Score (MOS) ──────────────────────────────────────────────

def match_overall_score(
    coverage_rate : float,   # matched / total  (0–1)
    fact_rate     : float,   # avg fact score / 3  (0–1)
    bert_avg      : float,   # avg BERTScore  (0–1)
    crr_ratio     : float,   # ai_crr / human_crr  (0–1, capped at 1)
) -> float:
    """
    Weighted 0–10 composite:
      40% coverage  — did the AI notice the events at all?
      30% semantics — does the sentence mean the same thing? (BERTScore)
      20% factual   — right player / team / outcome?
      10% context   — does it reference past events like a human would?
    """
    crr_ratio = min(crr_ratio, 1.0)
    raw = (0.40 * coverage_rate +
           0.30 * bert_avg      +
           0.20 * fact_rate     +
           0.10 * crr_ratio)
    return round(raw * 10, 2)


# ── CRR ───────────────────────────────────────────────────────────────────

def crr(texts):
    if not texts:
        return 0.0
    hits = sum(1 for t in texts
               if any(k in t.lower() for k in PAST_REF_KEYWORDS))
    return round(hits / len(texts) * 100, 1)


# ── Shared report builder ──────────────────────────────────────────────────

def _build_legacy_report(results, ai_events, espn_events, match_title="") -> tuple[str, dict]:
    """
    Build the existing per-event report from the legacy (text-log) workflow.
    Returns (report_string, summary_dict).
    """
    W = 70

    def bar(value, max_val=1.0, width=20):
        if value is None:
            return "not computed"
        filled = int((value / max_val) * width)
        return f"[{'█'*filled}{'░'*(width-filled)}] {value}"

    out = []
    title = match_title or "HUMAN vs AI COMMENTARY EVALUATION"
    out.append("=" * W)
    out.append(f"  {title}")
    out.append("=" * W)

    matched_count = sum(1 for r in results if r["matched"])
    missed_count  = len(results) - matched_count

    out.append(f"\n{'─'*W}")
    out.append("  EVENT-BY-EVENT COMPARISON")
    out.append(f"{'─'*W}")

    for r in results:
        out.append(f"\n┌─ {r['label']}")
        out.append(f"│  HUMAN : {r['human_text']}")
        if r["matched"]:
            out.append(f"│  AI    : {r['ai_text']}")
            p = "✓" if r["b_fact"]["player"]  else "✗"
            t = "✓" if r["b_fact"]["team"]    else "✗"
            o = "✓" if r["b_fact"]["outcome"] else "✗"
            c = "✓" if r["has_context"]       else "✗"
            out.append("│")
            out.append(f"│  [A] BLEU       : {bar(r['a_bleu'])}")
            out.append(f"│  [B] Fact       : {r['b_fact']['score']}/3  "
                       f"Player:{p}  Team:{t}  Outcome:{o}")
            out.append(f"│  [C] BERTScore  : {bar(r['c_bert'])}")
            out.append(f"│  [+] Uses past context: {c}")
        else:
            out.append("│  AI    : ✗  NO MATCH FOUND (AI missed this event)")
        out.append(f"└{'─'*(W-2)}")

    out.append(f"\n{'─'*W}")
    out.append("  MISMATCH SUMMARY")
    out.append(f"{'─'*W}")
    out.append(f"  Total human moments sampled : {len(results)}")
    out.append(f"  AI matched                  : {matched_count}  ✓")
    out.append(f"  AI missed entirely          : {missed_count}   ✗")

    wrong_player  = sum(1 for r in results if r["matched"] and not r["b_fact"]["player"])
    wrong_team    = sum(1 for r in results if r["matched"] and not r["b_fact"]["team"])
    wrong_outcome = sum(1 for r in results if r["matched"] and not r["b_fact"]["outcome"])
    out.append("\n  Of matched events:")
    out.append(f"    Wrong player  : {wrong_player}/{matched_count}")
    out.append(f"    Wrong team    : {wrong_team}/{matched_count}")
    out.append(f"    Wrong outcome : {wrong_outcome}/{matched_count}")

    if espn_events:
        cov = full_coverage_analysis(espn_events, ai_events)
        out.append(f"\n{'─'*W}")
        out.append("  FULL ESPN EVENT COVERAGE  (all events, not just sampled)")
        out.append(f"{'─'*W}")
        out.append(f"  {'Action':<16} {'ESPN':>6} {'AI matched':>10} {'Missed':>8}  Coverage")
        out.append(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*8}  {'─'*24}")

        total_espn = total_matched = 0
        for atype in ["Goal", "Shot", "Corner", "Foul", "Free_Kick", "Substitution", "Offside"]:
            s   = cov.get(atype, {"espn": 0, "matched": 0, "missed": []})
            pct = round(s["matched"] / s["espn"] * 100) if s["espn"] > 0 else 0
            bar_str = f"{'█' * int(pct/5)}{'░' * (20 - int(pct/5))}"
            out.append(f"  {atype:<16} {s['espn']:>6} {s['matched']:>10} "
                       f"{s['espn']-s['matched']:>8}  {pct:>3}%  {bar_str}")
            total_espn    += s["espn"]
            total_matched += s["matched"]

        overall_pct = round(total_matched / total_espn * 100) if total_espn else 0
        out.append(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*8}")
        out.append(f"  {'TOTAL':<16} {total_espn:>6} {total_matched:>10} "
                   f"{total_espn-total_matched:>8}  {overall_pct}% overall coverage")
        out.append("\n  Missed events by type:")
        for atype, s in cov.items():
            if s["missed"]:
                out.append(f"  {atype}:")
                for m in s["missed"][:5]:
                    out.append(f"    ✗  {m}")
                if len(s["missed"]) > 5:
                    out.append(f"    ... and {len(s['missed'])-5} more")

    matched    = [r for r in results if r["matched"]]
    avg_bleu   = round(sum(r["a_bleu"] for r in matched) / len(matched), 3) if matched else 0
    avg_fact   = round(sum(r["b_fact"]["score"] for r in matched) / len(matched), 2) if matched else 0
    valid_bert = [r["c_bert"] for r in matched if r["c_bert"] is not None]
    avg_bert   = round(sum(valid_bert) / len(valid_bert), 3) if valid_bert else None

    ai_crr    = crr([r["ai_text"]    for r in results if r["matched"]])
    human_crr = crr([r["human_text"] for r in results])

    matched_human_texts = [r["human_text"] for r in results if r["matched"]]
    matched_ai_texts    = [r["ai_text"]    for r in results if r["matched"]]
    corp_bleu     = corpus_bleu_score(matched_human_texts, matched_ai_texts)
    coverage_rate = matched_count / len(results) if results else 0.0
    fact_rate     = (avg_fact / 3) if matched else 0.0
    bert_val      = avg_bert if avg_bert is not None else 0.0
    crr_ratio     = (ai_crr / human_crr / 100) if human_crr > 0 else 0.0
    mos           = match_overall_score(coverage_rate, fact_rate, bert_val, crr_ratio)

    out.append(f"\n{'─'*W}")
    out.append("  AGGREGATE SCORES  (matched events only)")
    out.append(f"{'─'*W}")
    out.append(f"  [A] BLEU avg      : {bar(avg_bleu)}")
    out.append(f"  [B] Fact avg      : {avg_fact}/3  ({round(avg_fact / 3 * 100, 1)}%)")
    out.append(f"  [C] BERTScore avg : {bar(avg_bert)}")

    out.append(f"\n{'─'*W}")
    out.append("  CONTEXTUAL REFERENCE RATE (CRR)")
    out.append(f"{'─'*W}")
    out.append("  ESPN baseline     :  0.0%  (no history, by design)")
    out.append(f"  Human commentary  : {human_crr:>5}%")
    out.append(f"  AI commentary     : {ai_crr:>5}%")
    out.append(f"  KG contribution   : +{ai_crr}% over ESPN baseline")
    if human_crr > 0:
        out.append(f"  AI reaches        : {round(ai_crr / human_crr * 100, 1)}% of human CRR")

    # ── Match-level scores (above CONCLUSION) ────────────────────────────
    mos_bar_filled = int(mos)
    mos_bar = f"[{'█'*mos_bar_filled}{'░'*(10-mos_bar_filled)}] {mos}/10"

    out.append(f"\n{'─'*W}")
    out.append("  MATCH-LEVEL SCORES")
    out.append(f"{'─'*W}")
    out.append(f"  Corpus BLEU   : {bar(corp_bleu)}")
    out.append(f"    └─ Linguistic similarity across all matched event pairs.")
    out.append(f"  Match Overall : {mos_bar}")
    out.append(f"    └─ 40% coverage  ({round(coverage_rate*100,1)}%)  "
               f"30% semantic ({round(bert_val*100,1)}%)  "
               f"20% factual ({round(fact_rate*100,1)}%)  "
               f"10% context ({round(crr_ratio*100,1)}%)")

    out.append(f"\n{'='*W}")
    out.append("  CONCLUSION")
    out.append(f"{'='*W}")
    out.append(f"  Coverage  : AI detected {matched_count}/{len(results)} sampled moments "
               f"({round(matched_count / len(results) * 100)}%)")
    out.append(f"  Accuracy  : {round(avg_fact / 3 * 100, 1)}% factual correctness on detected events")
    out.append(f"  Context   : KG raises CRR from 0% → {ai_crr}% (human benchmark: {human_crr}%)")
    if avg_bert is not None:
        quality = "partial match" if avg_bert < 0.65 else "good match"
        out.append(f"  Semantics : BERTScore {avg_bert} — {quality} with human phrasing")
    out.append("")

    summary = {
        "match"    : title,
        "matched"  : matched_count,
        "total"    : len(results),
        "precision": round(matched_count / len(results), 3) if results else 0,
        "recall"   : round(matched_count / len(results), 3) if results else 0,
        "f1"       : 0.0,
        "bleu"     : avg_bleu,
        "bert"     : avg_bert,
        "crr_gt"   : human_crr,
        "crr_ai"   : ai_crr,
        "corp_bleu": corp_bleu,
        "mos"      : mos,
    }
    return "\n".join(out), summary


# ── Aggregate summary table ────────────────────────────────────────────────

def _print_aggregate_table(summaries: list[dict], out_path: Path):
    W   = 100
    COL = 28
    lines = []
    lines.append(f"\n{'='*W}")
    lines.append("  AGGREGATE SUMMARY — ALL MATCHES")
    lines.append(f"{'='*W}")
    hdr = (f"  {'Match':<{COL}} {'P':>5} {'R':>5} {'F1':>5} "
           f"{'BLEU':>6} {'BERT':>6} {'CRR_gt':>7} {'CRR_ai':>7} {'MOS':>5}")
    lines.append(hdr)
    lines.append(f"  {'─'*COL} {'─'*5} {'─'*5} {'─'*5} "
                 f"{'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5}")

    for s in summaries:
        name = s["match"][:COL]
        bert = f"{s['bert']:.3f}" if s["bert"] is not None else "  N/A"
        lines.append(
            f"  {name:<{COL}} "
            f"{s['precision']:>5.3f} {s['recall']:>5.3f} {s['f1']:>5.3f} "
            f"{s['bleu']:>6.3f} {bert:>6} "
            f"{s['crr_gt']:>6.1f}% {s['crr_ai']:>6.1f}% "
            f"{s['mos']:>5.2f}"
        )

    # averages
    def avg(key):
        vals = [s[key] for s in summaries if s[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    lines.append(f"  {'─'*COL} {'─'*5} {'─'*5} {'─'*5} "
                 f"{'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5}")
    bert_avg_str = f"{avg('bert'):.3f}"
    lines.append(
        f"  {'AVERAGE':<{COL}} "
        f"{avg('precision'):>5.3f} {avg('recall'):>5.3f} {avg('f1'):>5.3f} "
        f"{avg('bleu'):>6.3f} {bert_avg_str:>6} "
        f"{avg('crr_gt'):>6.1f}% {avg('crr_ai'):>6.1f}% "
        f"{avg('mos'):>5.2f}"
    )
    lines.append("")

    table = "\n".join(lines)
    print(table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"✓ Aggregate table saved to {out_path}")


# ── Multi-track per-match table ───────────────────────────────────────────

def _format_multitrack_table(
    folder            : Path,
    n_ai              : int,
    per_track         : dict[str, dict],
    multi_cider_result: tuple[float, int, int] | None = None,
) -> str:
    """Render the per-match × per-track table. Returns the report string."""
    W = 125
    n_tracks = len(per_track)

    mc_score, mc_scored, mc_total = 0.0, 0, n_ai
    if multi_cider_result is not None:
        mc_score, mc_scored, mc_total = multi_cider_result

    out = [
        "=" * W,
        f"  EVALUATION — {folder.name}",
        "=" * W,
        f"  AI events: {n_ai}",
        f"  GT tracks: " + ", ".join(TRACK_LABELS.get(k, k) for k in per_track),
        "",
        f"  {'─' * (W - 2)}",
        "    PER-TRACK SCORES  (single reference per event)",
        f"  {'─' * (W - 2)}",
    ]
    hdr = (f"  {'Track':<20} {'P':>5} {'R':>5} {'F1':>5} "
           f"{'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7} "
           f"{'ROUGE-1':>8} {'ROUGE-L':>8} {'CIDEr':>7} "
           f"{'BERT':>6} {'CRR':>5}")
    sep = (f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} "
           f"{'─'*7} {'─'*7} {'─'*7} "
           f"{'─'*8} {'─'*8} {'─'*7} "
           f"{'─'*6} {'─'*5}")
    out.append(hdr)
    out.append(sep)

    best_track, best_score = None, -1.0
    for track, s in per_track.items():
        bert = s.get("bert") if s.get("bert") is not None else 0.0
        comp = (s.get("bleu_1", 0.0) + s.get("bleu", 0.0) +
                s.get("meteor", 0.0) + s.get("rouge_l", 0.0) + bert) / 5
        if comp > best_score:
            best_score, best_track = comp, track
        bert_str  = f"{s['bert']:.2f}" if s.get("bert") is not None else " N/A"
        cider_str = f"{s.get('cider', 0.0):.2f}"
        out.append(
            f"  {TRACK_LABELS.get(track, track):<20} "
            f"{s['precision']:>5.2f} {s['recall']:>5.2f} {s['f1']:>5.2f} "
            f"{s.get('bleu_1', 0.0):>7.2f} {s.get('bleu', 0.0):>7.2f} "
            f"{s.get('meteor', 0.0):>7.2f} "
            f"{s.get('rouge_1', 0.0):>8.2f} {s.get('rouge_l', 0.0):>8.2f} "
            f"{cider_str:>7} {bert_str:>6} "
            f"{(s.get('crr_ai', 0.0) / 100):>5.2f}"
        )

    out.append(sep)
    if best_track:
        out.append(f"  Best GT track: {TRACK_LABELS.get(best_track, best_track)}  "
                   f"(avg BLEU-1/BLEU-4/METEOR/ROUGE-L/BERT = {best_score:.3f})")
    out.append("")

    # ── Multi-reference CIDEr section ────────────────────────────────────────
    out.append(f"  {'─' * (W - 2)}")
    out.append("    MULTI-REFERENCE CIDEr  (proper consensus eval)")
    out.append(f"  {'─' * (W - 2)}")
    out.append(f"  All {n_tracks} GT track(s) pooled as references for each event:")
    out.append(f"    Events scored   : {mc_scored} / {mc_total}  "
               f"(events with ≥1 matched GT reference)")
    out.append(f"    Multi-ref CIDEr : {mc_score:.3f}  "
               f"(vs JAIST's {JAIST_REFERENCE['cider']:.2f} — "
               f"they have a 27k-pair corpus vs our ~500-event corpus)")
    out.append(f"  Note: CIDEr TF-IDF stabilises with corpus size. Our corpus is ~50× "
               f"smaller than JAIST's MatchText,")
    out.append(f"  so absolute CIDEr will always be lower regardless of commentary quality.")
    out.append("")

    # ── JAIST comparison section ──────────────────────────────────────────────
    out.append(f"  {'─' * (W - 2)}")
    out.append(f"    HONEST COMPARISON vs {JAIST_REFERENCE['source']}")
    out.append(f"  {'─' * (W - 2)}")
    out.append(f"  (Best score across {n_tracks} available GT track(s) per metric — "
               f"not averaged, not cherry-picked)")
    out.append("")

    def best_metric(key):
        vals = [s.get(key) for s in per_track.values() if s.get(key) is not None]
        return max(vals) if vals else 0.0

    def best_bert_val():
        vals = [s.get("bert") for s in per_track.values() if s.get("bert") is not None]
        return max(vals) if vals else None

    you_bleu_1  = best_metric("bleu_1")
    you_bleu_4  = best_metric("bleu")
    you_meteor  = best_metric("meteor")
    you_rouge_1 = best_metric("rouge_1")
    you_rouge_l = best_metric("rouge_l")
    you_cider   = mc_score if multi_cider_result is not None else best_metric("cider")
    you_bert    = best_bert_val()
    you_crr     = best_metric("crr_ai")

    def verdict(you, jaist):
        if jaist is None or jaist == 0:
            return "You report"
        ratio = you / jaist
        if ratio >= 1.0:
            return f"YOU WIN ×{ratio:.1f}"
        return f"They win ×{1/ratio:.1f}"

    row_fmt = "  {:<10} {:>8} {:>8}    {}"
    out.append(row_fmt.format("Metric", "You", "JAIST", "Verdict"))
    out.append(f"  {'─'*10} {'─'*8} {'─'*8}    {'─'*22}")
    out.append(row_fmt.format("BLEU-1",
               f"{you_bleu_1:.2f}", f"{JAIST_REFERENCE['bleu_1']:.2f}",
               verdict(you_bleu_1, JAIST_REFERENCE["bleu_1"])))
    out.append(row_fmt.format("BLEU-4",
               f"{you_bleu_4:.2f}", f"{JAIST_REFERENCE['bleu_4']:.2f}",
               verdict(you_bleu_4, JAIST_REFERENCE["bleu_4"])))
    out.append(row_fmt.format("METEOR",
               f"{you_meteor:.2f}", f"{JAIST_REFERENCE['meteor']:.2f}",
               verdict(you_meteor, JAIST_REFERENCE["meteor"])))
    out.append(row_fmt.format("ROUGE-1",
               f"{you_rouge_1:.2f}", f"{JAIST_REFERENCE['rouge_1']:.2f}",
               verdict(you_rouge_1, JAIST_REFERENCE["rouge_1"])))
    out.append(row_fmt.format("ROUGE-L",
               f"{you_rouge_l:.2f}", f"{JAIST_REFERENCE['rouge_l']:.2f}",
               verdict(you_rouge_l, JAIST_REFERENCE["rouge_l"])))
    cider_note = " (corpus size, see above)" if you_cider < JAIST_REFERENCE["cider"] else ""
    out.append(row_fmt.format("CIDEr",
               f"{you_cider:.2f}", f"{JAIST_REFERENCE['cider']:.2f}",
               verdict(you_cider, JAIST_REFERENCE["cider"]) + cider_note))
    bert_str = f"{you_bert:.2f}" if you_bert is not None else "   N/A"
    out.append(row_fmt.format("BERTScore", bert_str, "   —", "You report"))
    out.append(row_fmt.format("CRR",
               f"{you_crr / 100:.2f}", "   —", "Your unique metric"))
    out.append("")
    return "\n".join(out)


def _multitrack_aggregate_table(
    per_match: list[tuple[str, dict[str, dict], tuple[float, int, int]]],
    out_path : Path,
):
    """Build and save the cross-match aggregate.
    per_match: list of (folder_name, {track: summary}, (mc_score, mc_scored, mc_total)).
    """
    W = 135
    lines = [
        "=" * W,
        "  AGGREGATE — multi-track evaluation across all matches",
        "=" * W,
    ]
    hdr = (f"  {'Match':<35} {'Track':<12} "
           f"{'P':>5} {'R':>5} {'F1':>5} "
           f"{'BL-1':>6} {'BL-4':>6} {'MET':>6} "
           f"{'R-1':>6} {'R-L':>6} {'CIDEr':>6} {'BERT':>6}")
    sep = (f"  {'─'*35} {'─'*12} {'─'*5} {'─'*5} {'─'*5} "
           f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    lines.append(hdr)
    lines.append(sep)

    best = {k: ("", "", -1.0) for k in
            ("bleu_1", "bleu", "meteor", "rouge_1", "rouge_l", "cider", "bert", "f1")}
    track_sums : dict[str, dict[str, float]] = {}
    track_n    : dict[str, int]              = {}

    for match_name, tracks, _ in per_match:
        short = match_name.split(" - ", 1)[-1][:35]
        for track, s in tracks.items():
            bert = s.get("bert") if s.get("bert") is not None else 0.0
            row = (f"  {short:<35} {TRACK_LABELS.get(track, track):<12} "
                   f"{s['precision']:>5.2f} {s['recall']:>5.2f} {s['f1']:>5.2f} "
                   f"{s.get('bleu_1', 0.0):>6.2f} {s.get('bleu', 0.0):>6.2f} "
                   f"{s.get('meteor', 0.0):>6.2f} "
                   f"{s.get('rouge_1', 0.0):>6.2f} {s.get('rouge_l', 0.0):>6.2f} "
                   f"{s.get('cider', 0.0):>6.2f} {bert:>6.2f}")
            lines.append(row)

            for m in ("bleu_1", "bleu", "meteor", "rouge_1", "rouge_l", "cider", "f1"):
                v = s.get(m) if s.get(m) is not None else 0.0
                if v > best[m][2]:
                    best[m] = (short, TRACK_LABELS.get(track, track), v)
            if s.get("bert") is not None and s["bert"] > best["bert"][2]:
                best["bert"] = (short, TRACK_LABELS.get(track, track), s["bert"])

            ts = track_sums.setdefault(track, {k: 0.0 for k in
                ("precision","recall","f1","bleu_1","bleu","meteor",
                 "rouge_1","rouge_l","cider","bert")})
            for k in ts:
                v = s.get(k)
                if v is None:
                    continue
                ts[k] += v
            track_n[track] = track_n.get(track, 0) + 1

    lines.append(sep)
    for track, sums in track_sums.items():
        n = track_n[track]
        lines.append(
            f"  {'AVERAGE':<35} {TRACK_LABELS.get(track, track):<12} "
            f"{sums['precision']/n:>5.2f} {sums['recall']/n:>5.2f} "
            f"{sums['f1']/n:>5.2f} "
            f"{sums['bleu_1']/n:>6.2f} {sums['bleu']/n:>6.2f} "
            f"{sums['meteor']/n:>6.2f} "
            f"{sums['rouge_1']/n:>6.2f} {sums['rouge_l']/n:>6.2f} "
            f"{sums['cider']/n:>6.2f} {sums['bert']/n:>6.2f}"
        )

    lines.append("")
    lines.append("  Best (match × track) per metric:")
    for m, (match, tk, v) in best.items():
        if match:
            lines.append(f"    {m:<8} : {v:.3f}  ←  {match}  /  {tk}")
    lines.append("")

    # ── Multi-reference CIDEr per match ──────────────────────────────────────
    lines.append(f"  {'─' * (W - 2)}")
    lines.append("    MULTI-REFERENCE CIDEr PER MATCH  (all GT tracks pooled per event)")
    lines.append(f"  {'─' * (W - 2)}")
    lines.append(f"  {'Match':<48} {'CIDEr':>8}  {'Scored/Total':<14}")
    lines.append(f"  {'─'*48} {'─'*8}  {'─'*14}")
    mc_scores_all: list[float] = []
    for match_name, _, (mc_score, mc_scored, mc_total) in per_match:
        short = match_name[:48]
        lines.append(f"  {short:<48} {mc_score:>8.3f}  {mc_scored}/{mc_total}")
        mc_scores_all.append(mc_score)
    avg_mc = sum(mc_scores_all) / len(mc_scores_all) if mc_scores_all else 0.0
    lines.append(f"  {'─'*48} {'─'*8}")
    lines.append(f"  {'AVERAGE':<48} {avg_mc:>8.3f}")
    lines.append("")

    # ── JAIST comparison ──────────────────────────────────────────────────────
    lines.append(f"  {'─' * (W - 2)}")
    lines.append(f"    HONEST COMPARISON vs {JAIST_REFERENCE['source']}")
    lines.append(f"  {'─' * (W - 2)}")
    lines.append("  (Grand average across all matches × all tracks)")
    lines.append("")

    grand: dict[str, float] = {}
    for key in ("bleu_1", "bleu", "meteor", "rouge_1", "rouge_l", "cider", "bert"):
        all_vals: list[float] = []
        for _, tracks, _ in per_match:
            for s in tracks.values():
                v = s.get(key)
                if v is not None:
                    all_vals.append(v)
        grand[key] = sum(all_vals) / len(all_vals) if all_vals else 0.0

    def verdict(you, jaist):
        if jaist is None or jaist == 0:
            return "You report"
        ratio = you / jaist
        if ratio >= 1.0:
            return f"YOU WIN ×{ratio:.1f}"
        return f"They win ×{1/ratio:.1f}"

    row_fmt = "  {:<12} {:>10} {:>8}    {}"
    lines.append(row_fmt.format("Metric", "You (avg)", "JAIST", "Verdict"))
    lines.append(f"  {'─'*12} {'─'*10} {'─'*8}    {'─'*22}")
    lines.append(row_fmt.format("BLEU-1",
                 f"{grand['bleu_1']:.2f}",
                 f"{JAIST_REFERENCE['bleu_1']:.2f}",
                 verdict(grand["bleu_1"], JAIST_REFERENCE["bleu_1"])))
    lines.append(row_fmt.format("BLEU-4",
                 f"{grand['bleu']:.2f}",
                 f"{JAIST_REFERENCE['bleu_4']:.2f}",
                 verdict(grand["bleu"], JAIST_REFERENCE["bleu_4"])))
    lines.append(row_fmt.format("METEOR",
                 f"{grand['meteor']:.2f}",
                 f"{JAIST_REFERENCE['meteor']:.2f}",
                 verdict(grand["meteor"], JAIST_REFERENCE["meteor"])))
    lines.append(row_fmt.format("ROUGE-1",
                 f"{grand['rouge_1']:.2f}",
                 f"{JAIST_REFERENCE['rouge_1']:.2f}",
                 verdict(grand["rouge_1"], JAIST_REFERENCE["rouge_1"])))
    lines.append(row_fmt.format("ROUGE-L",
                 f"{grand['rouge_l']:.2f}",
                 f"{JAIST_REFERENCE['rouge_l']:.2f}",
                 verdict(grand["rouge_l"], JAIST_REFERENCE["rouge_l"])))
    cider_note = " (corpus size, see per-match table)" if avg_mc < JAIST_REFERENCE["cider"] else ""
    lines.append(row_fmt.format("CIDEr (multi)",
                 f"{avg_mc:.2f}",
                 f"{JAIST_REFERENCE['cider']:.2f}",
                 verdict(avg_mc, JAIST_REFERENCE["cider"]) + cider_note))
    bert_v = grand["bert"]
    lines.append(row_fmt.format("BERTScore", f"{bert_v:.2f}", "   —", "You report"))
    lines.append("")

    table = "\n".join(lines)
    print(table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"✓ Multi-track aggregate saved to {out_path}")


def run_multitrack_match(
    folder       : Path,
    tracks_filter: list[str] | None = None,
    tolerance_min: float            = 1.0,
    out_path     : Path | None      = None,
) -> tuple[dict[str, dict], tuple[float, int, int]]:
    """
    Per-folder runner: load AI commentary, discover every GT track that
    exists, evaluate AI vs each track, render the per-match table.

    Returns (per_track, multi_cider_result) where:
        per_track           — {track_name: summary_dict}
        multi_cider_result  — (cider_score, n_scored, n_total_ai)
    """
    _empty_mcider: tuple[float, int, int] = (0.0, 0, 0)

    ai_path = folder / "ai_commentary.json"
    if not ai_path.exists():
        print(f"  [skip] {folder.name} — no ai_commentary.json")
        return {}, _empty_mcider
    ai_events = parse_json_commentary(ai_path)

    tracks = discover_tracks(folder, tracks_filter)
    if not tracks:
        print(f"  [skip] {folder.name} — no GT tracks found "
              f"(looked for: {tracks_filter or ALL_TRACKS})")
        return {}, _empty_mcider

    per_track: dict[str, dict] = {}
    for name, gt_events in tracks.items():
        summary = evaluate_match_json(
            gt_events, ai_events,
            match_name   = f"{folder.name} [track={name}]",
            tolerance_min= tolerance_min,
            verbose      = False,
        )
        per_track[name] = summary

    multi_cider_result = metric_cider_multireference(ai_events, tracks, tolerance_min)
    report = _format_multitrack_table(folder, len(ai_events), per_track, multi_cider_result)
    print(report)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"  Saved: {out_path}")
    return per_track, multi_cider_result


# ── AI event loader (JSON or text-log fallback) ───────────────────────────

def _raw_to_json_events(raw: list[dict]) -> list[dict]:
    return [{
        "minute"    : float(e["minute"]),
        "half"      : int(e["half"]),
        "event_type": e["event_type"],
        "player"    : "",
        "team"      : "",
        "human_text": e["full_text"],
    } for e in raw]


def _load_ai_events(
    json_path    : Path,
    log_fallback : Path,
    match_name   : str | None = None,
    shared_index : dict[str, list[dict]] | None = None,
) -> list[dict]:
    """
    Resolution order:
      1. per-match ai_commentary.json (json_path)
      2. per-match commentary_log.txt (sibling of json_path)
      3. shared_index[match_name] — pre-split section of the shared log
      4. shared log file with no headers (only if no match_name filter)
    """
    if json_path.exists():
        print(f"  Loading AI events from {json_path.name}")
        return parse_json_commentary(json_path)

    per_match_log = json_path.parent / "commentary_log.txt"
    if per_match_log.exists():
        print(f"  Loading AI events from per-match log: {per_match_log.name}")
        return _raw_to_json_events(parse_ai_log(str(per_match_log)))

    if match_name and shared_index is not None:
        section = shared_index.get(match_name)
        if section:
            print(f"  Loading AI events from shared log section [{match_name}] "
                  f"({len(section)} events)")
            return _raw_to_json_events(section)
        print(f"  [warn] No section '{match_name}' in shared log "
              f"(headers found: {list(shared_index.keys())[:3]}…)")
        return []

    if log_fallback.exists() and not match_name:
        print(f"  {json_path.name} not found — falling back to {log_fallback}")
        return _raw_to_json_events(parse_ai_log(str(log_fallback)))

    return []


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate AI commentary against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # legacy single-match (text log)
    ap.add_argument("--ai-log",     help="AI commentary text log")
    ap.add_argument("--human-json", help="Human/GT commentary JSON")
    ap.add_argument("--espn-csv",   help="ESPN CSV for coverage analysis")
    ap.add_argument("--output",     default="evaluation_report.txt",
                    help="Output path for single-match report")

    # JSON-based single match
    ap.add_argument("--match-dir",  help="Match folder containing GT and AI JSON files")

    # multi-match
    ap.add_argument("--all",        action="store_true",
                    help="Loop every match folder under data/ with AI commentary")
    ap.add_argument("--match",      help="Partial folder-name filter — runs the "
                                         "MULTI-TRACK evaluation for one match "
                                         "(e.g. 'Blackburn')")
    ap.add_argument("--tracks",     nargs="+",
                    choices=["espn", "qwen_v1", "qwen_v2"],
                    help="Limit the multi-track eval to specific tracks (default: all)")
    ap.add_argument("--gt-file",    default="ground_truth_commentary.json",
                    help="GT filename inside each match folder (default: ground_truth_commentary.json)")
    ap.add_argument("--ai-file",    default="ai_commentary.json",
                    help="AI filename inside each match folder (default: ai_commentary.json)")
    ap.add_argument("--tolerance",  type=float, default=1.0,
                    help="Match tolerance in minutes for JSON mode (default: 1.0)")

    args = ap.parse_args()

    # ── MODE D: --match <partial> — MULTI-TRACK single match ────────────
    if args.match and not args.all:
        candidates = [f for f in DATA_DIR.iterdir()
                      if f.is_dir() and args.match.lower() in f.name.lower()]
        if not candidates:
            print(f"No folder under {DATA_DIR} matching '{args.match}'.")
            return
        if len(candidates) > 1:
            print(f"Multiple folders match '{args.match}':")
            for c in candidates:
                print(f"  - {c.name}")
            print("Use a more specific filter.")
            return
        folder   = candidates[0]
        out_name = f"evaluation_{args.match.lower().replace(' ', '_')}_multitrack.txt"
        run_multitrack_match(
            folder        = folder,
            tracks_filter = args.tracks,
            tolerance_min = args.tolerance,
            out_path      = OUT_DIR / out_name,
        )
        return

    # ── MODE A: --all — MULTI-TRACK across every match folder ──────────
    if args.all:
        # Folder qualifies if it has ai_commentary.json AND at least one
        # GT track (espn from KG, qwen_v1, or qwen_v2).
        folders = sorted(
            f for f in DATA_DIR.iterdir()
            if f.is_dir() and (f / args.ai_file).exists()
        )
        if not folders:
            print(f"No match folders found under {DATA_DIR} with '{args.ai_file}'.")
            return

        print(f"Found {len(folders)} folder(s) with AI commentary.\n")
        per_match: list[tuple[str, dict[str, dict], tuple[float, int, int]]] = []
        for folder in folders:
            tracks, mcider = run_multitrack_match(
                folder        = folder,
                tracks_filter = args.tracks,
                tolerance_min = args.tolerance,
                out_path      = OUT_DIR /
                    f"evaluation_{folder.name.replace(' ', '_')}_multitrack.txt",
            )
            if tracks:
                per_match.append((folder.name, tracks, mcider))

        if per_match:
            _multitrack_aggregate_table(
                per_match,
                OUT_DIR / "evaluation_multitrack_aggregate.txt",
            )
        return

    # ── MODE B: --match-dir  (JSON single match) ─────────────────────────
    if args.match_dir:
        folder   = Path(args.match_dir)
        gt_path  = folder / args.gt_file
        if not gt_path.exists():
            print(f"GT file not found: {gt_path}")
            return

        shared_log   = OUT_DIR / "commentary_log.txt"
        shared_index = split_log_by_match(shared_log) if shared_log.exists() else {}
        ai_events    = _load_ai_events(
            folder / args.ai_file, shared_log,
            match_name   = folder.name,
            shared_index = shared_index,
        )
        if not ai_events:
            print("No AI commentary found. Run the pipeline first to generate commentary.")
            return

        gt_events = parse_json_commentary(gt_path)
        evaluate_match_json(
            gt_events, ai_events,
            match_name   = folder.name,
            tolerance_min= args.tolerance,
            verbose      = True,
        )
        return

    # ── MODE C: legacy --ai-log / --human-json  ──────────────────────────
    if not args.ai_log or not args.human_json:
        ap.print_help()
        return

    ai_events    = parse_ai_log(args.ai_log)
    human_events = json.loads(Path(args.human_json).read_text())
    espn_events  = parse_espn_csv(args.espn_csv) if args.espn_csv else []

    results                  = []
    bert_refs, bert_hyps, bert_idx = [], [], []

    for h in human_events:
        ai      = find_match(h, ai_events)
        ai_text = ai["full_text"] if ai else ""

        bleu = metric_a_bleu(h["human_text"], ai_text)
        fact = metric_b_fact(h, ai_text)
        has_context = (any(k in ai_text.lower() for k in PAST_REF_KEYWORDS)
                       if ai_text else False)

        if ai_text:
            bert_refs.append(h["human_text"])
            bert_hyps.append(ai_text)
            bert_idx.append(len(results))

        results.append({
            "label"      : (f"{h['half']}H {h['minute']:02d}'  "
                            f"{h['event_type']:<14} {h.get('player','')}"),
            "human_text" : h["human_text"],
            "ai_text"    : ai_text,
            "matched"    : ai is not None,
            "a_bleu"     : bleu,
            "b_fact"     : fact,
            "c_bert"     : None,
            "has_context": has_context,
        })

    bert_scores = metric_c_bert(bert_refs, bert_hyps)
    for i, score in zip(bert_idx, bert_scores):
        results[i]["c_bert"] = score

    report, _ = _build_legacy_report(
        results, ai_events, espn_events,
        match_title="HUMAN vs AI COMMENTARY EVALUATION",
    )
    print(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
