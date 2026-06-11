# deprecated/

Scripts moved here during project cleanup on **2026-06-12**.
Nothing has been deleted — all history is preserved via `git mv`.

These files are **not imported by any active code** and are **not part of the
pipeline** (`main.py → src/`). They can be restored at any time with:

```bash
git mv deprecated/<filename> <original_path>
git commit -m "restore: <filename>"
```

---

## Files

### `evaluate.py`

**Moved from:** project root  
**Reason:** Superseded by `src/commentator/evaluate_commentary.py` which
provides multi-track GT evaluation, BLEU-1/BLEU-4/METEOR/ROUGE-1/ROUGE-L/
CIDEr/BERTScore/CRR, and a per-track JAIST comparison table. The root-level
`evaluate.py` only computed P/R/F1 for event detection against
`Labels-ball.json` + ESPN CSV (no text quality metrics).  
**Last active fix:** fix 020 (ESPN CSV restored for Foul/Corner GT)  
**Size:** 537 lines

---

### `validate_confidence.py`

**Moved from:** project root  
**Reason:** One-off confidence-formula validation script (added in fix ced8182).
The confidence threshold is now set and stable at 0.65; all confidence values
are stored in the KG as `ekg:hasConfidence`. There is nothing left to validate.  
**Last active fix:** ced8182 (initial creation)  
**Size:** 527 lines  
**Outputs (if ever re-run):** `data/validation/score_distribution.png`,
`data/validation/precision_recall_curve.png`,
`data/validation/threshold_f1_curve.png`

---

### `check_KG_for_pee_dodo.py`

**Moved from:** project root  
**Reason:** Standalone TA inspection script — 10 SPARQL queries over `ekg.ttl`
for demo/review purposes. Useful for ad-hoc inspection but not part of the
active pipeline. Can be run from `deprecated/` without changes (uses
`Path(__file__).resolve().parent.parent` to locate `data/kg_output/ekg.ttl`
after the move).  
**Last active fix:** da1b563 (absolute path fix)  
**Size:** ~200 lines  
**Note:** If needed for a TA demo this week, restore to root or run directly:
`python deprecated/check_KG_for_pee_dodo.py`

---

### `ekg_explorer.py`

**Moved from:** project root  
**Reason:** Replaced by `src/5_web_viz/server.py` which is the actively
maintained web visualisation. `ekg_explorer.py` was a Streamlit-based explorer
that pre-dates the current `src/` layout.  
**Last active fix:** 2f98733 (property rename sweep — touched but not actively used)  
**Size:** 1084 lines  
**To run the current visualiser instead:**
```bash
cd src/5_web_viz && python server.py
```
