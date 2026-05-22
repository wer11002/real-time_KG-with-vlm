"""
run_tdeed_test.py — T-DEED inference on 500 test frames (Step 2)

Must be run from the T-DEED root directory so relative module imports resolve:
    cd /work/s2616011/models/T-DEED
    python /work/s2616011/real-time_KG-with-vlm/src/tdeed_integration/run_tdeed_test.py
"""

import os, json, torch, numpy as np
from torch.utils.data import DataLoader
from dataset.frame import ActionSpotVideoDataset
from model.model import TDEEDModel
from util.io import load_text
from util.eval import process_frame_predictions, soft_non_maximum_supression

FRAME_DIR  = '/tmp/tdeed_test_frames'
VIDEO_NAME = 'test_20s'
CHECKPOINT = 'checkpoints/checkpoint_best.pt'
OUT_DIR    = '/tmp/tdeed_test_out'
os.makedirs(OUT_DIR, exist_ok=True)

# ── class list ─────────────────────────────────────────────────────────────
classes = {}
for i, x in enumerate(load_text('data/soccernetball/class.txt')):
    classes[x + '-left']  = (i * 2) + 1
    classes[x + '-right'] = (i * 2) + 2

# ── model ──────────────────────────────────────────────────────────────────
class Args:
    feature_arch  = 'rny002_gsf'
    temporal_arch = 'ed_sgp_mixer'
    clip_len      = 100
    modality      = 'rgb'
    num_classes   = 12
    crop_dim      = None
    batch_size    = 2
    n_layers      = 2

args   = Args()
model  = TDEEDModel(args=args)
ckpt   = torch.load(CHECKPOINT, map_location='cpu')
model.load(ckpt)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = model.to(device)
print(f"Model loaded on {device}")

# ── dataset / dataloader ───────────────────────────────────────────────────
dataset = ActionSpotVideoDataset(
    classes, FRAME_DIR, VIDEO_NAME, args, 'fps', True)
loader  = DataLoader(dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=2, pin_memory=(device == 'cuda'))

# ── inference ──────────────────────────────────────────────────────────────
all_preds = []
for batch in loader:
    frames = batch['frame'].to(device)
    with torch.no_grad():
        pred = model(frames)
    all_preds.append(pred.cpu())

pred_tensor = torch.cat(all_preds, dim=0)
results     = process_frame_predictions(dataset, classes, pred_tensor)
results     = soft_non_maximum_supression(results, window=4)

# ── filter to EKG-relevant classes ────────────────────────────────────────
EKG_CLASSES    = {'SHOT', 'GOAL', 'FREE KICK', 'CORNER', 'CORNER KICK'}
CONF_THRESHOLD = 0.25

all_preds_list = results.get('predictions', [])
ekg_detections = [
    p for p in all_preds_list
    if any(ec in p['label'].upper() for ec in EKG_CLASSES)
    and p['confidence'] >= CONF_THRESHOLD
]
high_conf = [p for p in all_preds_list if p['confidence'] >= 0.5]

# ── report ─────────────────────────────────────────────────────────────────
print(f"\n=== T-DEED EKG-RELEVANT DETECTIONS (conf >= {CONF_THRESHOLD}) ===")
print(f"Total predictions : {len(all_preds_list)}")
print(f"EKG-relevant      : {len(ekg_detections)}")
for p in sorted(ekg_detections, key=lambda x: x['position']):
    print(f"  {p['gameTime']:<12} {p['label']:<25} conf:{p['confidence']:.3f}"
          f"  team:{p.get('team','?')}  frame:{p['position']}")

print(f"\n=== ALL HIGH CONF (>= 0.5) ===")
for p in sorted(high_conf, key=lambda x: x['position']):
    print(f"  {p['gameTime']:<12} {p['label']:<25} conf:{p['confidence']:.3f}"
          f"  team:{p.get('team','?')}")

out_file = f'{OUT_DIR}/detections.json'
json.dump({'ekg_detections': ekg_detections, 'high_conf': high_conf},
          open(out_file, 'w'), indent=2)
print(f"\nSaved to {out_file}")
