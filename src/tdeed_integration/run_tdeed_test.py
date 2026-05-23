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

import torch.nn.functional as F
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
# Use base-only classes (values 1-12). The model im_feat head outputs 13
# columns (12 classes + 1 background). Left/right split (24 classes) would
# need 25 columns but the loaded head only has 13.
classes = {}
for i, x in enumerate(load_text('data/soccernetball/class.txt')):
    classes[x] = i + 1

print(f"Classes ({len(classes)}): {classes}")

# ── model ──────────────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location='cpu')

if isinstance(ckpt, dict) and 'args' in ckpt:
    args       = ckpt['args']
    args.batch_size = 2
    state_dict = ckpt.get('state_dict', ckpt)
    print(f"Args loaded from checkpoint: {vars(args)}")
else:
    state_dict = ckpt

    sgp_ks            = state_dict['_temp_fine._sgp.0.psi.weight'].shape[2]
    radi_displacement = (state_dict['_temp_fine._sgp.0.convkw.weight'].shape[2] - 1) // 2
    n_sgp_mixer = sum(
        1 for k in state_dict
        if re.match(r'_temp_fine\._sgpMixer\.\d+\.ln1\.weight$', k))

    print(f"Inferred: sgp_ks={sgp_ks}  radi_displacement={radi_displacement}  "
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

try:
    model.load(state_dict)
except RuntimeError as e:
    print(f"Strict load failed — remapping old _fc1/_fc2 head keys to current _fc_out")
    print(f"  ({str(e)[:200]})")
    # Checkpoint was saved with two heads (_fc1 = class, _fc2 = team).
    # Current codebase merged them into a single _fc_out.
    # Remap _fc1._fc_out → _fc_out so the real class-prediction weights load.
    remapped = {}
    for k, v in state_dict.items():
        if k == '_pred_fine._fc1._fc_out.weight':
            remapped['_pred_fine._fc_out.weight'] = v
        elif k == '_pred_fine._fc1._fc_out.bias':
            remapped['_pred_fine._fc_out.bias'] = v
        else:
            remapped[k] = v
    model_sd   = model._model.state_dict()
    compatible = {k: v for k, v in remapped.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    missing    = [k for k in model_sd  if k not in compatible]
    extra      = [k for k in remapped   if k not in compatible]
    model._model.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)}/{len(model_sd)} tensors; "
          f"{len(missing)} random-initialised, {len(extra)} checkpoint-only skipped")
    if missing:
        print(f"  Still random: {missing}")

# TDEEDModel is a custom wrapper — eval/to/__call__ live on model._model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model._model.eval()
model._model = model._model.to(device)
print(f"Model on {device}")

# ── dataset / dataloader ───────────────────────────────────────────────────
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

# ── pred_dict accumulator ──────────────────────────────────────────────────
# process_frame_predictions(dataset, classes, pred_dict) expects:
#   pred_dict = {video: (scores_arr, support_arr)}
#   scores_arr shape: (video_len, len(classes)+1)  ← index 0 = background
#   support_arr shape: (video_len,)  ← count of clips contributing to each frame
# It returns: list of {'video': str, 'events': [...], 'fps': float}
# Each event: {'label': str, 'frame': int, 'score': float}
n_cols = len(classes) + 1   # 13: background(0) + classes 1..12
pred_dict = {}
for video, video_len, _ in dataset.videos:
    pred_dict[video] = (
        np.zeros((video_len, n_cols), np.float32),
        np.zeros(video_len, np.int32))

print(f"pred_dict initialized: {len(pred_dict)} video(s), "
      f"shape=({list(pred_dict.values())[0][0].shape}), n_cols={n_cols}")

for i, batch in enumerate(loader):
    videos = batch['video']
    starts = batch['start'].numpy()
    frames = batch['frame'].to(device)

    with torch.no_grad():
        out, _ = model._model(frames)

    cls_logits = out['im_feat'] if isinstance(out, dict) else out  # (B, T, 13)
    cls_probs  = F.softmax(cls_logits, dim=-1).cpu().numpy()       # (B, T, 13)

    if i == 0:
        print(f"  im_feat shape: {cls_logits.shape}  n_cols={n_cols}")

    clip_len = cls_probs.shape[1]
    for j in range(len(videos)):
        video = videos[j]
        start = int(starts[j])
        vid_scores, vid_support = pred_dict[video]
        vid_len = len(vid_scores)

        s  = max(0, start)
        cs = max(0, -start)
        e  = min(start + clip_len, vid_len)
        ce = cs + (e - s)
        if ce > cs:
            chunk = cls_probs[j, cs:ce]            # (frames, 13)
            vid_scores[s:e]  += chunk
            vid_support[s:e] += (chunk.sum(axis=1) != 0).astype(np.int32)

# ── post-processing ────────────────────────────────────────────────────────
pred_events = process_frame_predictions(dataset, classes, pred_dict)
# pred_events: list of {'video': str, 'events': [...], 'fps': float}

results = soft_non_maximum_supression(pred_events, window=4)
# results: same structure, NMS applied

all_preds_list = [e for r in results for e in r.get('events', [])]
# each e: {'label': str, 'frame': int, 'score': float}

# ── filter to EKG-relevant classes ────────────────────────────────────────
EKG_CLASSES    = {'SHOT', 'GOAL', 'FREE KICK', 'FREE_KICK'}
CONF_THRESHOLD = 0.25

ekg_detections = [
    p for p in all_preds_list
    if any(ec in p['label'].upper() for ec in EKG_CLASSES)
    and p['score'] >= CONF_THRESHOLD
]
high_conf = [p for p in all_preds_list if p['score'] >= 0.5]

# ── report ─────────────────────────────────────────────────────────────────
print(f"\n=== T-DEED DETECTIONS (conf >= {CONF_THRESHOLD}) ===")
print(f"Total predictions : {len(all_preds_list)}")
print(f"EKG-relevant      : {len(ekg_detections)}")
for p in sorted(ekg_detections, key=lambda x: x['frame']):
    t = p['frame'] / 25.0
    team = p.get('team', '-')
    print(f"  frame:{p['frame']:<5} t={t:5.1f}s  {p['label']:<20} score:{p['score']:.3f}  team:{team}")

print(f"\n=== ALL HIGH CONF (>= 0.5) ===")
for p in sorted(high_conf, key=lambda x: x['frame']):
    t = p['frame'] / 25.0
    team = p.get('team', '-')
    print(f"  frame:{p['frame']:<5} t={t:5.1f}s  {p['label']:<20} score:{p['score']:.3f}  team:{team}")

out_file = f'{OUT_DIR}/detections.json'
json.dump({'ekg_detections': ekg_detections, 'high_conf': high_conf,
           'all_preds': all_preds_list},
          open(out_file, 'w'), indent=2)
print(f"\nSaved to {out_file}")
