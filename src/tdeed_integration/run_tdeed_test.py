"""
run_tdeed_test.py — T-DEED inference on 500 test frames (Step 2)

Must be run from the T-DEED root directory so relative module imports resolve:
    cd /work/s2616011/models/T-DEED
    python /work/s2616011/real-time_KG-with-vlm/src/tdeed_integration/run_tdeed_test.py
"""

import os, sys, json, torch, numpy as np
# run_all.sh does `cd T-DEED && python /full/path/this_script.py`
# Python adds the *script* directory to sys.path, not the CWD — fix that:
sys.path.insert(0, os.getcwd())

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
# Load checkpoint first — it may contain the training args so we don't have
# to guess which fields TDEEDModel.__init__ requires.
ckpt = torch.load(CHECKPOINT, map_location='cpu')
ckpt_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
print(f"Checkpoint keys: {ckpt_keys}")

if isinstance(ckpt, dict) and 'args' in ckpt:
    args = ckpt['args']
    # override batch_size so we don't OOM on a 20-s test
    args.batch_size = 2
    print(f"Args loaded from checkpoint: {vars(args)}")
else:
    # Fallback: define known attributes; __getattr__ catches anything else so
    # the script survives if the model needs additional fields.
    class Args:
        def __init__(self):
            self.feature_arch      = 'rny002_gsf'
            self.temporal_arch     = 'ed_sgp_mixer'
            self.clip_len          = 100
            self.modality          = 'rgb'
            self.num_classes       = 12
            self.crop_dim          = None
            self.batch_size        = 2
            self.n_layers          = 2
            self.radi_displacement = 2
            self.sgp_ks            = 3
            self.sgp_r             = 1
            self.event_team        = True   # SoccerNet ball: left/right side

        def __getattr__(self, name):
            # Only reached for attributes NOT set in __init__
            print(f"  WARNING: Args missing '{name}' — defaulting to None; "
                  f"add it to the fallback Args if the model crashes")
            return None

    args = Args()
    print("WARNING: args not found in checkpoint — using hardcoded defaults.")

model = TDEEDModel(args=args)
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
