"""
run_tdeed_test.py — T-DEED inference on 500 test frames (Step 2)

Must be run from the T-DEED root directory so relative module imports resolve:
    cd /work/s2616011/models/T-DEED
    python /work/s2616011/real-time_KG-with-vlm/src/tdeed_integration/run_tdeed_test.py
"""

import os, sys, re, json, torch, numpy as np
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

if isinstance(ckpt, dict) and 'args' in ckpt:
    args       = ckpt['args']
    args.batch_size = 2
    state_dict = ckpt.get('state_dict', ckpt)
    print(f"Args loaded from checkpoint: {vars(args)}")
else:
    # Checkpoint is a raw state_dict — reverse-engineer architecture params
    # from weight tensor shapes so we don't guess wrong and get size mismatches.
    state_dict = ckpt

    # sgp_ks: psi conv kernel size (directly readable)
    sgp_ks = state_dict['_temp_fine._sgp.0.psi.weight'].shape[2]
    # radi_displacement: convkw kernel = 2*radi+1  →  radi = (size-1)//2
    radi_displacement = (state_dict['_temp_fine._sgp.0.convkw.weight'].shape[2] - 1) // 2
    # n_layers: count SGP-Mixer blocks
    n_sgp_mixer = sum(
        1 for k in state_dict
        if re.match(r'_temp_fine\._sgpMixer\.\d+\.ln1\.weight$', k))

    print(f"Inferred from checkpoint shapes: "
          f"sgp_ks={sgp_ks}  radi_displacement={radi_displacement}  "
          f"n_sgp_mixer={n_sgp_mixer}")

    class Args:
        def __init__(self):
            self.feature_arch      = 'rny002_gsf'
            self.temporal_arch     = 'ed_sgp_mixer'
            self.clip_len          = 100
            self.modality          = 'rgb'
            self.num_classes       = 12
            self.crop_dim          = None
            self.batch_size        = 2
            self.n_layers          = n_sgp_mixer if n_sgp_mixer > 0 else 2
            self.radi_displacement = radi_displacement
            self.sgp_ks            = sgp_ks
            self.sgp_r             = 1
            self.event_team        = True

        def __getattr__(self, name):
            print(f"  WARNING: Args missing '{name}' — defaulting to None")
            return None

    args = Args()
    print(f"Fallback Args: { {k: v for k, v in vars(args).items()} }")

model = TDEEDModel(args=args)

# Try strict load first; fall back to shape-filtered strict=False if head
# architecture differs between checkpoint and current codebase version.
try:
    model.load(state_dict)
except RuntimeError as e:
    print(f"Strict load failed — retrying with shape-compatible keys only")
    print(f"  ({str(e)[:300]})")
    model_sd   = model._model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    missing    = [k for k in model_sd  if k not in compatible]
    extra      = [k for k in state_dict if k not in compatible]
    model._model.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)}/{len(model_sd)} tensors; "
          f"{len(missing)} random-initialised, {len(extra)} checkpoint-only skipped")
# TDEEDModel is a custom wrapper — eval/to/__call__ live on model._model (the Impl nn.Module)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model._model.eval()
model._model = model._model.to(device)
print(f"Model on {device}")

# ── dataset / dataloader ───────────────────────────────────────────────────
# ActionSpotVideoDataset(classes, label_file, frame_dir, modality, clip_len, ...)
# Frame naming: frame{N}.jpg (1-indexed, no zero-padding) per soccernetball convention
video_dir  = os.path.join(FRAME_DIR, VIDEO_NAME)
num_frames = len([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
print(f"Found {num_frames} frames in {video_dir}")

LABEL_FILE = '/tmp/tdeed_test_labels.json'
json.dump([{'video': VIDEO_NAME, 'num_frames': num_frames, 'fps': 25, 'events': []}],
          open(LABEL_FILE, 'w'))

dataset = ActionSpotVideoDataset(
    classes, LABEL_FILE, FRAME_DIR,
    modality='rgb', clip_len=args.clip_len,
    overlap_len=0, dataset='soccernetball')
loader  = DataLoader(dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=2, pin_memory=(device == 'cuda'))

# ── inference ──────────────────────────────────────────────────────────────
all_preds = []
for i, batch in enumerate(loader):
    frames = batch['frame'].to(device)
    with torch.no_grad():
        out = model._model(frames)
    # Inspect output structure on first batch to handle any return format
    if i == 0:
        if isinstance(out, tuple):
            print(f"  model output: tuple len={len(out)}  "
                  f"shapes={[x.shape if hasattr(x,'shape') else type(x) for x in out]}")
        else:
            print(f"  model output: {type(out)} shape={getattr(out,'shape',None)}")
    # Extract the classification score tensor (shape B×T×C or B×C)
    pred = out[0] if isinstance(out, tuple) else out
    if isinstance(pred, dict):
        pred = pred.get('cls', next(iter(pred.values())))
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
