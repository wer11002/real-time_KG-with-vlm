# Fix 012 — evaluate.py: Remove ESPN CSV Dependency

**Status:** Applied  
**Files changed:** `evaluate.py`

---

## Problem

`evaluate.py` required an ESPN CSV file (hardcoded path to
`data/blackburn_forest_2019-10-01.csv`) to load ground truth. When the CSV was
absent, the script crashed with `FileNotFoundError` before any evaluation ran.

Additionally, Pass was included in `KEY_ACTIONS` which produced false negatives:
Labels-ball.json annotates Pass events that the pipeline intentionally does not
detect (Pass was removed from `VALID_ACTIONS`).

---

## Fix

1. **ESPN CSV removed** — `evaluate.py` now uses `Labels-ball.json` only as
   ground truth. `load_ground_truth(labels_path)` takes no ESPN args.

2. **KEY_ACTIONS updated** — `Pass` removed:
   ```python
   KEY_ACTIONS = {"Shot", "Goal", "Foul", "Corner", "Free_Kick"}
   ```
   Note: Foul/Corner/Offside are not in Labels-ball.json, so GT for those is
   always 0. They appear in the table but produce no FN inflation.

3. **`--coverage-min` flag added** — filters GT events to only those within the
   pipeline's processed window (e.g. `--coverage-min 35` when testing 70 clips
   covering 35 minutes). Prevents FNs from unannotated future clips inflating the
   miss rate.

4. **Verbose TP output** — matches with `Δt > 1.0 min` are flagged with ⚠ large Δt.

---

## What this does not fix

`evaluate.py` still uses a 2-minute match tolerance for TP/FP decisions.
With timestamp accuracy now ±4s (Fix 001), this tolerance could produce false
positive TPs where two events in the same 2-minute window both claim the same GT.
Status: In plan.
