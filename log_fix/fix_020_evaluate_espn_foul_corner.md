# Fix 020 — evaluate.py: ESPN CSV for Foul/Corner GT + Tolerance 0.5 + Gate Logging

**Status:** Applied  
**Files changed:** `evaluate.py`, `src/3_buffer_matcher/align.py`

---

## Problems

**P1 — Foul/Corner phantom FPs.**  
Fix 012 removed ESPN CSV entirely. Labels-ball.json has no Foul or Corner
annotations, so GT = 0 for both. Every detected Foul or Corner was a FP by
definition — dragging Overall F1 down by phantom errors. The docstring still
claimed "ESPN CSV (22 fouls, 14 corners)" but the code did nothing of the kind.

**P2 — ±2 min tolerance is 30× too loose.**  
Pipeline timestamp accuracy is ±4s (Fix 001). A ±2 min window lets a shot at
9:14 match a GT shot at 11:14 — potentially two different events. The suspicious
TP at Δ=1.10 min (66 seconds) highlighted this. Shot F1 was inflated.

**P3 — Gated events invisible to inspection.**  
Fix 013 ESPN gate blocks low-confidence shots before KG ingestion with no logged
output. A gate that incorrectly blocks a real shot becomes a silent FN — recall
looks better than it is with no way to audit.

---

## Fixes

### evaluate.py

`load_gt_csv(csv_path, actions)` added — reads ESPN CSV, filters to Foul and
Corner only. `load_ground_truth(labels_path, csv_path=None)` updated to call it:

```python
gt = load_gt_labels(labels_path)          # Shot, Goal, Free_Kick
if csv_path and csv_path.exists():
    gt += load_gt_csv(csv_path, actions={"Foul", "Corner"})
```

Default CSV path: `data/blackburn_forest_2019-10-01.csv`.  
`--csv` flag added to override path. Missing CSV prints a warning but
doesn't crash — Foul/Corner GT reverts to 0 gracefully.

`per_action[action]["source"]` now shows "ESPN CSV" for Foul/Corner so the
table is self-documenting.

Default tolerance changed `2.0 → 0.5` (30 seconds). Matches ±4s pipeline
accuracy with headroom for clock rounding. `--tolerance` flag still available
to override.

### align.py

In `align_buffer()`, gated events now print to stdout:
```
[gated] 1st 17:33  Shot        conf=0.62  (ESPN nearby: not Shot/Goal)
```
This lets you inspect every gate decision after each ESPN tick without changing
any data flow.

---

## What this does not fix

ESPN CSV times use minute-level granularity ("23'") versus Labels-ball.json
millisecond precision. Foul/Corner matching is inherently less precise than
Shot matching. A Foul at "23'" could be anywhere in minute 23:00–23:59.
Tolerance 0.5 min = 30s is still a reasonable window for ESPN-sourced GT.
