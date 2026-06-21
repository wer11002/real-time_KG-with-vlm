"""
experiments/shared/runner.py — shared pipeline runner for ablation experiments.

run_experiment() executes two subprocesses per experiment, then evaluates:
  Step 1: python main.py [--match X] --exp-name NAME [extra flags]
          → runs VLM + KG pipeline; commentator writes commentary_log.txt
  Step 2: python src/commentator/extract_ai_commentary_per_match.py
              --exp-name NAME [--match X]
          → reads KG + log; writes ai_commentary_<NAME>.json per match folder
  Step 3: evaluate each match using evaluate_match_json; aggregate metrics
  Step 4: append results to experiments/results.json
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import BASE_DIR, DATA_DIR, RESULTS

# evaluate_commentary lives in src/commentator — add to path once at import time
sys.path.insert(0, str(BASE_DIR / "src" / "commentator"))
from evaluate_commentary import evaluate_match_json, parse_json_commentary


def run_experiment(
    name: str,
    match_filter: Optional[str] = None,
    extra_main_args: Optional[list] = None,
) -> dict:
    """
    Run one experiment level end-to-end.

    Args:
        name:            experiment name tag (e.g. "baseline", "cot", "kg_history")
        match_filter:    optional partial folder name — passed as --match to main.py
        extra_main_args: additional CLI flags for main.py (e.g. ["--use-cot"])

    Returns:
        dict keyed by match folder name, each value is a metrics dict.
    """
    extra_main_args = extra_main_args or []

    # ── Step 1: full VLM + KG pipeline ──────────────────────────────────────
    main_cmd = [sys.executable, str(BASE_DIR / "main.py")]
    if match_filter:
        main_cmd += ["--match", match_filter]
    main_cmd += ["--exp-name", name]
    main_cmd += extra_main_args

    print(f"\n[runner:{name}] Step 1 — pipeline")
    print(f"  {' '.join(main_cmd)}")
    subprocess.run(main_cmd, check=True)

    # ── Step 2: extract per-match AI commentary JSON ─────────────────────────
    extract_cmd = [
        sys.executable,
        str(BASE_DIR / "src" / "commentator" / "extract_ai_commentary_per_match.py"),
        "--exp-name", name,
    ]
    if match_filter:
        extract_cmd += ["--match", match_filter]

    print(f"\n[runner:{name}] Step 2 — extract ai_commentary_{name}.json")
    print(f"  {' '.join(extract_cmd)}")
    subprocess.run(extract_cmd, check=True)

    # ── Step 3: evaluate ─────────────────────────────────────────────────────
    print(f"\n[runner:{name}] Step 3 — evaluate")
    results: dict[str, dict] = {}

    for match_dir in sorted(DATA_DIR.iterdir()):
        if not match_dir.is_dir():
            continue
        if match_filter and match_filter.lower() not in match_dir.name.lower():
            continue

        ai_json = match_dir / f"ai_commentary_{name}.json"
        gt_json = match_dir / "human_commentary.json"

        if not ai_json.exists():
            print(f"  [skip] {match_dir.name[:55]}: no {ai_json.name}")
            continue
        if not gt_json.exists():
            print(f"  [skip] {match_dir.name[:55]}: no human_commentary.json")
            continue

        gt_events = parse_json_commentary(gt_json)
        ai_events = parse_json_commentary(ai_json)

        raw = evaluate_match_json(
            gt_events=gt_events,
            ai_events=ai_events,
            match_name=match_dir.name,
            verbose=False,
        )

        results[match_dir.name] = {
            "bleu_4"   : raw.get("bleu",      0.0),
            "meteor"   : raw.get("meteor",     0.0),
            "rouge_l"  : raw.get("rouge_l",    0.0),
            "cider"    : raw.get("cider",      0.0),
            "bertscore": raw.get("bert",       0.0) or 0.0,
            "crr"      : raw.get("crr_ai",     0.0),
            "f1"       : raw.get("f1",         0.0),
            "precision": raw.get("precision",  0.0),
            "recall"   : raw.get("recall",     0.0),
            "n_gt"     : raw.get("n_gt",       0),
            "n_ai"     : raw.get("n_ai",       0),
            "matched"  : raw.get("matched",    0),
        }
        m = results[match_dir.name]
        print(f"  {match_dir.name[:50]:<52} "
              f"BLEU4={m['bleu_4']:.3f} "
              f"METEOR={m['meteor']:.3f} "
              f"CIDEr={m['cider']:.3f} "
              f"CRR={m['crr']:.1f}%")

    # ── Step 4: persist to results.json ─────────────────────────────────────
    _save(name, results)
    return results


def _save(name: str, results: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    if RESULTS.exists():
        try:
            all_results = json.loads(RESULTS.read_text())
        except Exception:
            all_results = {}
    all_results[name] = results
    RESULTS.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n[runner] Saved → {RESULTS}")


def aggregate(results: dict[str, dict]) -> dict:
    """
    Compute macro-average across all matches.
    Returns {metric: avg_value, "n_matches": int}.
    """
    from .config import METRICS
    if not results:
        return {}
    n = len(results)
    agg: dict = {"n_matches": n}
    for m in METRICS + ["f1", "precision", "recall"]:
        vals = [v.get(m, 0.0) for v in results.values() if v.get(m) is not None]
        agg[m] = round(sum(vals) / len(vals), 3) if vals else 0.0
    return agg
