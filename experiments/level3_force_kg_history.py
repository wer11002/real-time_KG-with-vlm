"""
experiments/level3_force_kg_history.py — Level 3: Forced KG Player History

Before each LLM call, queries the KG for all prior events by the same player
in this match and injects them as an additional system message.
Enabled via EXP_FORCE_HISTORY=1 (set by --force-history flag in main.py).

If the player is unidentified, the injection step is silently skipped and
commentary falls back to the baseline behaviour.

Usage:
    # All matches
    python experiments/level3_force_kg_history.py

    # Single match (smoke test)
    python experiments/level3_force_kg_history.py --match "Blackburn"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.runner import aggregate, run_experiment

NAME = "kg_history"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default=None,
                    help="Partial match folder name (run one match only)")
    args = ap.parse_args()

    print(f"═══ Experiment: {NAME} ═══")
    results = run_experiment(
        name=NAME,
        match_filter=args.match,
        extra_main_args=["--force-history"],
    )

    if not results:
        print("No results — check that human_commentary.json exists in data/sn_long/*/")
        sys.exit(1)

    agg = aggregate(results)
    print(f"\n── Aggregate ({agg['n_matches']} matches) ──")
    print(f"  BLEU-4   : {agg.get('bleu_4', 0):.3f}")
    print(f"  METEOR   : {agg.get('meteor', 0):.3f}")
    print(f"  ROUGE-L  : {agg.get('rouge_l', 0):.3f}")
    print(f"  CIDEr    : {agg.get('cider', 0):.3f}")
    print(f"  BERTScore: {agg.get('bertscore', 0):.3f}")
    print(f"  CRR      : {agg.get('crr', 0):.1f}%")
    print(f"  F1       : {agg.get('f1', 0):.3f}")


if __name__ == "__main__":
    main()
