from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "sn_long"
RESULTS  = BASE_DIR / "experiments" / "results.json"

# Ordered list of metric keys used in runner output and compare table
METRICS = ["bleu_4", "meteor", "rouge_l", "cider", "bertscore", "crr"]
