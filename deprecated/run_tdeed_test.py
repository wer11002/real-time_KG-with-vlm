"""
run_tdeed_test.py — T-DEED inference on the full match

Must be run from the T-DEED root directory so relative module imports resolve:
    cd /work/s2616011/models/T-DEED
    python /work/s2616011/real-time_KG-with-vlm/src/tdeed_integration/run_tdeed_test.py
"""

import os, sys, re, json, torch, numpy as np
sys.path.insert(0, os.getcwd())

import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.frame import ActionSpotVideoDataset
from model.model import TDEEDModel
from util.io import load_text
from util.eval import process_frame_predictions, soft_non_maximum_supression

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_WORK_DIR     = os.path.abspath(os.path.join(_PIPELINE_DIR, '..'))  # parent of project

FRAME_DIR  = os.path.join(_WORK_DIR, 'tdeed_full_frames')
VIDEO_NAME = 'full_match'
CHECKPOINT = 'checkpoints/checkpoint_best.pt'
OUT_DIR    = os.path.join(_WORK_DIR, 'tdeed_full_out')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"FRAME_DIR : {FRAME_DIR}")
print(f"OUT_DIR   : {OUT_DIR}")

# ── helpers ────────────────────────────────────────────────────────────────
def frame_to_gametime(frame_num, fps=25):
    secs = int(frame_num / fps)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ── class list ─────────────────────────────────────────────────────────────
classes = {}
for i, x in enumerate(load_text('data/soccernetball/class.txt')):
    classes[x] = i + 1

print(f"Classes ({len(classes)}): {list(classes.keys())}")

# ── model ──────────────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location='cpu')

if isinstance(ckpt, dict) and 'args' in ckpt:
    args       = ckpt['args']
    args.batch_size = 4
    state_dict = ckpt.get('state_dict', ckpt)
    print(f"Args from checkpoint: {vars(args)}")
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
            self.batch_size        = 4
            self.n_layers          = n_sgp_mixer if n_sgp_mixer > 0 else 2
            self.radi_displacement = radi_displacement
            self.sgp_ks            = sgp_ks
            self.sgp_r             = 1
            self.event_team        = True
        def __getattr__(self, name):
            print(f"  WARNING: Args missing '{name}' — defaulting to None")
            return None

    args = Args()

model = TDEEDModel(args=args)

try:
    model.load(state_dict)
except RuntimeError as e:
    print(f"Strict load failed — remapping old _fc1/_fc2 head keys to current _fc_out")
    print(f"  ({str(e)[:200]})")
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
    for k in model_sd:
        if k not in compatible and 'convkw' in k and k in remapped:
            ckpt_w  = remapped[k].float()
            target  = model_sd[k].shape[2]
            resized = F.interpolate(ckpt_w, size=target, mode='linear', align_corners=False)
            compatible[k] = resized.to(model_sd[k].dtype)
            print(f"  Interpolated {k}: kernel {ckpt_w.shape[2]} → {target}")
    missing = [k for k in model_sd if k not in compatible]
    extra   = [k for k in remapped  if k not in compatible]
    model._model.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)}/{len(model_sd)} tensors; "
          f"{len(missing)} random-initialised, {len(extra)} checkpoint-only skipped")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model._model.eval()
model._model = model._model.to(device)
print(f"Model on {device}")

# ── dataset / dataloader ───────────────────────────────────────────────────
video_dir  = os.path.join(FRAME_DIR, VIDEO_NAME)
num_frames = len([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
print(f"Found {num_frames} frames ({num_frames/25/60:.1f} min)")

LABEL_FILE = '/tmp/tdeed_full_labels.json'
json.dump([{'video': VIDEO_NAME, 'num_frames': num_frames, 'fps': 25, 'events': []}],
          open(LABEL_FILE, 'w'))

dataset = ActionSpotVideoDataset(
    classes, LABEL_FILE, FRAME_DIR,
    modality='rgb', clip_len=args.clip_len,
    overlap_len=0, dataset='soccernetball')
loader  = DataLoader(dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=4, pin_memory=(device == 'cuda'))

print(f"Clips to process: {len(dataset)}")

# ── inference ──────────────────────────────────────────────────────────────
n_cols = len(classes) + 1
pred_dict = {}
for video, video_len, _ in dataset.videos:
    pred_dict[video] = (
        np.zeros((video_len, n_cols), np.float32),
        np.zeros(video_len, np.int32))

print("Running inference...")
for i, batch in enumerate(loader):
    if i % 50 == 0:
        pct = 100 * i * args.batch_size / len(dataset)
        print(f"  {pct:5.1f}%  batch {i}/{len(loader)}")

    videos = batch['video']
    starts = batch['start'].numpy()
    frames = batch['frame'].to(device)

    with torch.no_grad():
        out, _ = model._model(frames)

    cls_logits = out['im_feat'] if isinstance(out, dict) else out
    cls_probs  = F.softmax(cls_logits, dim=-1).cpu().numpy()

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
            chunk = cls_probs[j, cs:ce]
            vid_scores[s:e]  += chunk
            vid_support[s:e] += (chunk.sum(axis=1) != 0).astype(np.int32)

print("Post-processing...")
pred_events = process_frame_predictions(dataset, classes, pred_dict)
results     = soft_non_maximum_supression(pred_events, window=4)
all_preds   = [e for r in results for e in r.get('events', [])]

# ── filter ─────────────────────────────────────────────────────────────────
EKG_CLASSES    = {'SHOT', 'GOAL', 'FREE KICK', 'FREE_KICK'}
CONF_THRESHOLD = 0.10

ekg = [p for p in all_preds
       if any(ec in p['label'].upper() for ec in EKG_CLASSES)
       and p['score'] >= CONF_THRESHOLD]
high = [p for p in all_preds if p['score'] >= 0.40]

# ── report ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FULL MATCH — T-DEED DETECTIONS")
print(f"{'='*60}")
print(f"Total predictions : {len(all_preds)}")
print(f"EKG-relevant (>={CONF_THRESHOLD}) : {len(ekg)}")

print(f"\n--- EKG events (SHOT / GOAL / FREE KICK) ---")
for p in sorted(ekg, key=lambda x: x['frame']):
    t = frame_to_gametime(p['frame'])
    team = p.get('team', '-')
    print(f"  {t}  {p['label']:<20} score:{p['score']:.3f}  team:{team}")

print(f"\n--- All high confidence (>= 0.40) ---")
for p in sorted(high, key=lambda x: x['frame']):
    t = frame_to_gametime(p['frame'])
    team = p.get('team', '-')
    print(f"  {t}  {p['label']:<20} score:{p['score']:.3f}  team:{team}")

out_file = f'{OUT_DIR}/detections.json'
json.dump({'ekg': ekg, 'high_conf': high, 'all_preds': all_preds},
          open(out_file, 'w'), indent=2)
print(f"\nSaved to {out_file}")
