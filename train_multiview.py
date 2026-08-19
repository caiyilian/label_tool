import os, json, glob, sys, time, pickle, random
import cv2
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTPUT_DIR, 'cached_multiview.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
NUM_KP = len(KP_NAMES)

# ============================
# 1. Load data with both views
# ============================
def load_all_data():
    samples = []
    ann_files = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    for fpath in ann_files:
        rel = fpath.replace(BASE, '').lstrip('/')
        cls_name = rel.split('/')[0]
        with open(fpath) as f:
            data = json.load(f)

        jpg = os.path.basename(fpath).replace('.json', '.jpg')
        front_path = os.path.join(os.path.dirname(fpath).replace('annotations', 'forward'), jpg)
        back_path = os.path.join(os.path.dirname(fpath).replace('annotations', 'backward'), jpg.replace('front_', 'back_'))

        front_img = cv2.imread(front_path)
        back_img = cv2.imread(back_path)
        if front_img is None or back_img is None:
            continue
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
        h, w = front_img.shape[:2]

        for ann in data.get('annotations', []):
            bbox = ann['bbox']
            back_bbox = ann.get('back_bbox')
            if back_bbox is None:
                back_bbox = bbox
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            front_crop = cv2.resize(front_img[y1:y2, x1:x2], (224, 224))
            ch, cw = front_crop.shape[:2]

            bx1, by1, bx2, by2 = map(int, back_bbox)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 <= bx1 or by2 <= by1:
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            back_crop = cv2.resize(back_img[by1:by2, bx1:bx2], (224, 224))

            kp_front = ann.get('pose_keypoints') or []
            kp_front_target = np.zeros((NUM_KP, 3), dtype=np.float32)
            for kp in kp_front:
                if kp is None:
                    continue
                idx = KP_NAMES.index(kp['name']) if kp['name'] in KP_NAMES else -1
                if idx < 0:
                    continue
                vis = 1.0 if kp.get('visible') else 0.0
                if vis:
                    kx = np.clip((kp['x'] - x1) / max(cw, 1), 0, 1)
                    ky = np.clip((kp['y'] - y1) / max(ch, 1), 0, 1)
                else:
                    kx, ky = 0.0, 0.0
                kp_front_target[idx] = [kx, ky, vis]

            kp_back = ann.get('pose_keypoints_back') or []
            kp_back_target = np.zeros((NUM_KP, 3), dtype=np.float32)
            for kp in kp_back:
                if kp is None:
                    continue
                idx = KP_NAMES.index(kp['name']) if kp['name'] in KP_NAMES else -1
                if idx < 0:
                    continue
                vis = 1.0 if kp.get('visible') else 0.0
                if vis:
                    kx = np.clip((kp['x'] - bx1) / max(bx2 - bx1, 1), 0, 1)
                    ky = np.clip((kp['y'] - by1) / max(by2 - by1, 1), 0, 1)
                else:
                    kx, ky = 0.0, 0.0
                kp_back_target[idx] = [kx, ky, vis]

            has_back = int(any(kp.get('visible') for kp in kp_back if kp))

            samples.append({
                'class': cls_name,
                'front_img': front_crop,
                'back_img': back_crop,
                'kp_front': kp_front_target,
                'kp_back': kp_back_target,
                'has_back_kp': has_back,
                'status': ann['status'],
                'posture': ann['posture'],
                'bad_reason': ann.get('bad_reason', []),
            })
    return samples

if os.path.exists(CACHE_FILE):
    print('Loading cached data...')
    all_samples = pickle.load(open(CACHE_FILE, 'rb'))
else:
    print('Loading and preprocessing data...')
    all_samples = load_all_data()
    pickle.dump(all_samples, open(CACHE_FILE, 'wb'))
print(f'Total samples: {len(all_samples)}')

# Count back KP coverage
n_back = sum(s['has_back_kp'] for s in all_samples)
print(f'Samples with back keypoints: {n_back}/{len(all_samples)} ({n_back/len(all_samples)*100:.1f}%)')

status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}

# ============================
# 2. Split by class
# ============================
classes = sorted(set(s['class'] for s in all_samples))
random.shuffle(classes)
n = len(classes)
train_classes = classes[:int(n * 0.7)]
val_classes = classes[int(n * 0.7):int(n * 0.85)]
test_classes = classes[int(n * 0.85):]
print(f'Train ({len(train_classes)}): {train_classes}')
print(f'Val ({len(val_classes)}): {val_classes}')
print(f'Test ({len(test_classes)}): {test_classes}')

train_samples = [s for s in all_samples if s['class'] in train_classes]
val_samples = [s for s in all_samples if s['class'] in val_classes]
test_samples = [s for s in all_samples if s['class'] in test_classes]
print(f'Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

# Check back KP distribution in splits
for name, split in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
    n_bk = sum(s['has_back_kp'] for s in split)
    print(f'  {name}: {n_bk}/{len(split)} have back KP ({n_bk/max(len(split),1)*100:.1f}%)')

# ============================
# 3. Prepare tensors
# ============================
samples_mean = np.array([0.485, 0.456, 0.406])
samples_std = np.array([0.229, 0.224, 0.225])

def prepare_data(samples, augment=False):
    front_imgs, back_imgs = [], []
    kp_fronts, kp_backs = [], []
    has_backs = []
    status_labels, posture_labels, br_labels = [], [], []
    for s in samples:
        fim = s['front_img'].astype(np.float32) / 255.0
        bim = s['back_img'].astype(np.float32) / 255.0
        if augment:
            if random.random() < 0.3:
                fim = np.fliplr(fim).copy()
                bim = np.fliplr(bim).copy()
        fim = (fim - samples_mean) / samples_std
        bim = (bim - samples_mean) / samples_std
        front_imgs.append(torch.from_numpy(fim.transpose(2, 0, 1)).float())
        back_imgs.append(torch.from_numpy(bim.transpose(2, 0, 1)).float())
        kp_fronts.append(torch.from_numpy(s['kp_front'].flatten()).float())
        kp_backs.append(torch.from_numpy(s['kp_back'].flatten()).float())
        has_backs.append(torch.tensor(s['has_back_kp']).float())
        status_labels.append(status_map[s['status']])
        posture_labels.append(posture_map[s['posture']])
        br = [0] * 4
        for r in s['bad_reason']:
            if r in bad_reason_map:
                br[bad_reason_map[r]] = 1
        br_labels.append(br)
    return (torch.stack(front_imgs), torch.stack(back_imgs),
            torch.stack(kp_fronts), torch.stack(kp_backs),
            torch.stack(has_backs),
            torch.tensor(status_labels), torch.tensor(posture_labels),
            torch.tensor(br_labels, dtype=torch.float32))

print('Preparing tensors...')
train_data = prepare_data(train_samples, augment=True)
val_data = prepare_data(val_samples, augment=False)
test_data = prepare_data(test_samples, augment=False)

# ============================
# 4. Two-Stream Multi-View Model
# ============================
class TwoStreamModel(nn.Module):
    def __init__(self, num_kp=8, num_status=4, num_posture=2, num_br=4):
        super().__init__()
        backbone = models.efficientnet_b0(weights='DEFAULT')
        self.stream = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        self.kp_front_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_kp * 3))
        self.kp_back_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_kp * 3))
        self.status_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_status))
        self.posture_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feat_dim, 64),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_posture))
        self.br_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feat_dim, 64),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_br))

    def forward(self, front_x, back_x):
        f_f = self.avgpool(self.stream(front_x)).flatten(1)
        f_b = self.avgpool(self.stream(back_x)).flatten(1)
        fused = f_f + f_b
        return (self.kp_front_head(f_f), self.kp_back_head(f_b),
                self.status_head(fused), self.posture_head(fused),
                self.br_head(fused))

model = TwoStreamModel().to(DEVICE)

# ============================
# 5. Loss functions
# ============================
def kp_loss(pred, target, eps=1e-6):
    B, D = pred.shape
    target = target.view(B, NUM_KP, 3)
    pred = pred.view(B, NUM_KP, 3)
    pred_xy = torch.sigmoid(pred[:, :, :2])
    pred_vis = torch.sigmoid(pred[:, :, 2:3])
    true_xy = target[:, :, :2]
    true_vis = target[:, :, 2:3]
    vis_mask = (true_vis > 0.5).float()
    xy_loss = (vis_mask * (pred_xy - true_xy) ** 2).sum() / (vis_mask.sum() + eps)
    vis_loss = nn.functional.binary_cross_entropy(pred_vis, true_vis)
    return xy_loss + 0.5 * vis_loss

def back_kp_loss(pred, target, has_back_mask):
    B, D = pred.shape
    target = target.view(B, NUM_KP, 3)
    pred = pred.view(B, NUM_KP, 3)
    pred_xy = torch.sigmoid(pred[:, :, :2])
    pred_vis = torch.sigmoid(pred[:, :, 2:3])
    true_xy = target[:, :, :2]
    true_vis = target[:, :, 2:3]
    has_back_mask = has_back_mask.view(B, 1, 1)
    vis_mask = (true_vis > 0.5).float() * has_back_mask
    xy_loss = (vis_mask * (pred_xy - true_xy) ** 2).sum() / (vis_mask.sum() + 1e-6)
    vis_loss = (has_back_mask * nn.functional.binary_cross_entropy(pred_vis, true_vis, reduction='none')).sum()
    vis_loss = vis_loss / (has_back_mask.sum() + 1e-6)
    return xy_loss + 0.5 * vis_loss

criterion_kp_front = kp_loss
criterion_kp_back = back_kp_loss
status_weights = torch.tensor([1.0, 1.0, 3.0, 3.0], device=DEVICE)
criterion_status = nn.CrossEntropyLoss(weight=status_weights)
criterion_posture = nn.CrossEntropyLoss()
criterion_br = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ============================
# 6. Training
# ============================
def batch_iter(data, batch_size=64, shuffle=True):
    fi, bi, kpf, kpb, hb, s, p, br = data
    n = len(fi)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, n, batch_size):
        idx = indices[i:i+batch_size]
        yield (fi[idx].to(DEVICE), bi[idx].to(DEVICE),
               kpf[idx].to(DEVICE), kpb[idx].to(DEVICE),
               hb[idx].to(DEVICE), s[idx].to(DEVICE),
               p[idx].to(DEVICE), br[idx].to(DEVICE))

def evaluate(data):
    model.eval()
    fi, bi, kpf, kpb, hb, s, p, br = data
    fi, bi, kpf, kpb, hb, s, p, br = (fi.to(DEVICE), bi.to(DEVICE),
        kpf.to(DEVICE), kpb.to(DEVICE), hb.to(DEVICE),
        s.to(DEVICE), p.to(DEVICE), br.to(DEVICE))
    with torch.no_grad():
        kpf_o, kpb_o, s_o, p_o, br_o = model(fi, bi)
        loss = (criterion_kp_front(kpf_o, kpf) +
                criterion_kp_back(kpb_o, kpb, hb) +
                criterion_status(s_o, s) +
                criterion_posture(p_o, p) +
                criterion_br(br_o, br))
    return (loss.item(), kpf_o, kpb_o, s_o, p_o, br_o, s, p, br, kpf, kpb, hb)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [],
           'val_kpf_err': [], 'val_kpb_err': [],
           'val_status_acc': [], 'val_posture_acc': []}

for epoch in range(30):
    t0 = time.time()
    model.train()
    total_loss = 0
    n_batches = 0
    for fi, bi, kpf, kpb, hb, s, p, br in batch_iter(train_data, batch_size=64):
        optimizer.zero_grad()
        kpf_o, kpb_o, s_o, p_o, br_o = model(fi, bi)
        loss = (criterion_kp_front(kpf_o, kpf) +
                criterion_kp_back(kpb_o, kpb, hb) +
                criterion_status(s_o, s) +
                criterion_posture(p_o, p) +
                criterion_br(br_o, br))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    vloss, vkpf_o, vkpb_o, vs_o, vp_o, vbr_o, vs, vp, vbr, vkpf, vkpb, vhb = evaluate(val_data)
    vs_acc = accuracy_score(vs.cpu().numpy(), vs_o.argmax(1).cpu().numpy())
    vp_acc = accuracy_score(vp.cpu().numpy(), vp_o.argmax(1).cpu().numpy())

    kpf_t = vkpf.view(-1, NUM_KP, 3)
    kpf_p = torch.sigmoid(vkpf_o).view(-1, NUM_KP, 3)
    vmask = (kpf_t[:, :, 2] > 0.5).float()
    kpf_err = ((kpf_p[:, :, :2] - kpf_t[:, :, :2]).abs() * vmask.unsqueeze(-1)).sum() / (vmask.sum() + 1e-6)

    kpb_t = vkpb.view(-1, NUM_KP, 3)
    kpb_p = torch.sigmoid(vkpb_o).view(-1, NUM_KP, 3)
    bmask = (kpb_t[:, :, 2] > 0.5).float() * vhb.view(-1, 1, 1)
    kpb_err = ((kpb_p[:, :, :2] - kpb_t[:, :, :2]).abs() * bmask.unsqueeze(-1)).sum() / (bmask.sum() + 1e-6)

    history['train_loss'].append(total_loss / n_batches)
    history['val_loss'].append(vloss)
    history['val_kpf_err'].append(kpf_err.item())
    history['val_kpb_err'].append(kpb_err.item())
    history['val_status_acc'].append(vs_acc)
    history['val_posture_acc'].append(vp_acc)

    if vloss < best_val_loss:
        best_val_loss = vloss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_multiview.pth'))

    elapsed = time.time() - t0
    print(f'Epoch {epoch+1:2d} ({elapsed:.0f}s): loss={total_loss/n_batches:.4f}/{vloss:.4f} '
          f'kp_f={kpf_err.item():.4f} kp_b={kpb_err.item():.4f} '
          f'sta={vs_acc:.4f} pos={vp_acc:.4f}')
    scheduler.step(vloss)

# ============================
# 7. Test Evaluation
# ============================
print('\n===== TEST EVALUATION =====')
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_multiview.pth')))
model.eval()

tloss, tkpf_o, tkpb_o, ts_o, tp_o, tbr_o, ts, tp, tbr, tkpf, tkpb, thb = evaluate(test_data)

ts_p = ts_o.argmax(1).cpu().numpy()
ts_t = ts.cpu().numpy()
tp_p = tp_o.argmax(1).cpu().numpy()
tp_t = tp.cpu().numpy()
tbr_p = (torch.sigmoid(tbr_o) > 0.5).cpu().numpy()
tbr_t = tbr.cpu().numpy()
thb_np = thb.cpu().numpy()

status_rev = {v: k for k, v in status_map.items()}
posture_rev = {v: k for k, v in posture_map.items()}
br_rev = {v: k for k, v in bad_reason_map.items()}

# Front keypoint eval
kpf_t = tkpf.view(-1, NUM_KP, 3).cpu().numpy()
kpf_p = torch.sigmoid(tkpf_o).view(-1, NUM_KP, 3).cpu().numpy()

print(f'\n--- Front View Keypoints ---')
total_xy, total_n = 0, 0
for i in range(NUM_KP):
    mask = kpf_t[:, i, 2] > 0.5
    vis_t = kpf_t[:, i, 2]
    vis_p = (kpf_p[:, i, 2] > 0.5).astype(float)
    vacc = accuracy_score(vis_t, vis_p)
    xy_err = np.abs(kpf_p[mask, i, :2] - kpf_t[mask, i, :2]).mean() if mask.sum() > 0 else 0
    print(f'  {KP_NAMES[i]:15s}: xy_err={xy_err:.4f}, vis_acc={vacc:.4f} ({int(mask.sum())}/{len(mask)})')
    total_xy += xy_err * mask.sum()
    total_n += mask.sum()
print(f'  {"AVERAGE":15s}: xy_err={total_xy/max(total_n,1):.4f}')

# Back keypoint eval
kpb_t = tkpb.view(-1, NUM_KP, 3).cpu().numpy()
kpb_p = torch.sigmoid(tkpb_o).view(-1, NUM_KP, 3).cpu().numpy()
has_back_mask = thb_np > 0.5

print(f'\n--- Back View Keypoints ---')
if has_back_mask.sum() > 0:
    total_xy, total_n = 0, 0
    for i in range(NUM_KP):
        mask = (kpb_t[:, i, 2] > 0.5) & has_back_mask
        vis_t = kpb_t[:, i, 2]
        vis_p = (kpb_p[:, i, 2] > 0.5).astype(float)
        vacc = accuracy_score(vis_t[has_back_mask], vis_p[has_back_mask])
        xy_err = np.abs(kpb_p[mask, i, :2] - kpb_t[mask, i, :2]).mean() if mask.sum() > 0 else 0
        print(f'  {KP_NAMES[i]:15s}: xy_err={xy_err:.4f}, vis_acc={vacc:.4f} ({int(mask.sum())}/{int(has_back_mask.sum())})')
        total_xy += xy_err * mask.sum()
        total_n += mask.sum()
    print(f'  {"AVERAGE":15s}: xy_err={total_xy/max(total_n,1):.4f}')
else:
    print('  (no back keypoints in test set)')

print(f'\n--- Status Classification ---')
print(f'Accuracy: {accuracy_score(ts_t, ts_p):.4f}')
print(classification_report(ts_t, ts_p, target_names=[status_rev[i] for i in range(4)], zero_division=0))

print(f'\n--- Posture Classification ---')
print(f'Accuracy: {accuracy_score(tp_t, tp_p):.4f}')
print(classification_report(tp_t, tp_p, target_names=[posture_rev[i] for i in range(2)], zero_division=0))

print(f'\n--- Bad Reason (Multi-label) ---')
for i in range(4):
    gt = tbr_t[:, i]
    pred = tbr_p[:, i]
    acc = accuracy_score(gt, pred)
    p, r, f, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
    print(f'  {br_rev[i]:15s}: acc={acc:.4f}, precision={p:.4f}, recall={r:.4f}, f1={f:.4f}')

# ============================
# 8. Visualizations
# ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].set_title('Total Loss')

axes[0, 1].plot(history['val_kpf_err'], label='Front KP')
axes[0, 1].plot(history['val_kpb_err'], label='Back KP')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('XY Error')
axes[0, 1].legend()
axes[0, 1].set_title('Keypoint Error (Val)')

axes[0, 2].plot(history['val_status_acc'], label='Status')
axes[0, 2].plot(history['val_posture_acc'], label='Posture')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].legend()
axes[0, 2].set_title('Classification (Val)')

for ax, y_true, y_pred, labels, title in [
    (axes[1, 0], ts_t, ts_p, [status_rev[i] for i in range(4)], 'Status CM'),
    (axes[1, 1], tp_t, tp_p, [posture_rev[i] for i in range(2)], 'Posture CM'),
]:
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

ax = axes[1, 2]
x = range(NUM_KP)
xf = [np.abs(kpf_p[mask, i, :2] - kpf_t[mask, i, :2]).mean() if (mask := (kpf_t[:, i, 2] > 0.5)).sum() > 0 else 0 for i in range(NUM_KP)]
xb = [np.abs(kpb_p[mask, i, :2] - kpb_t[mask, i, :2]).mean() if (mask := (kpb_t[:, i, 2] > 0.5) & has_back_mask).sum() > 0 else 0 for i in range(NUM_KP)]
ax.bar([xi - 0.15 for xi in x], xf, width=0.3, alpha=0.7, label='Front')
ax.bar([xi + 0.15 for xi in x], xb, width=0.3, alpha=0.7, label='Back')
ax.set_xticks(list(x))
ax.set_xticklabels(KP_NAMES, rotation=45, ha='right')
ax.set_ylabel('XY Error')
ax.set_title('Keypoint Error by View')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'multiview_results.png'), dpi=150)
print(f'\nResults saved')

# ============================
# 9. Sample keypoint viz
# ============================
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()
vis_idx = np.where((kpf_t[:, :, 2].sum(axis=1) > 0))[0]
if len(vis_idx) > 0:
    chosen = np.random.choice(vis_idx, min(8, len(vis_idx)), replace=False)
    for ax_i, idx in enumerate(chosen):
        ft = test_data[0][idx].cpu().numpy().transpose(1, 2, 0)
        ft = np.clip(ft * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[ax_i].imshow(ft)
        for k in range(NUM_KP):
            if kpf_t[idx, k, 2] > 0.5:
                axes[ax_i].scatter(kpf_t[idx, k, 0] * 224, kpf_t[idx, k, 1] * 224, c='g', s=30, marker='o')
                axes[ax_i].scatter(kpf_p[idx, k, 0] * 224, kpf_p[idx, k, 1] * 224, c='r', s=20, marker='x')
        axes[ax_i].axis('off')
        axes[ax_i].set_title('Front - green=GT red=pred')
    for ax_i in range(len(chosen), 16):
        axes[ax_i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'multiview_samples.png'), dpi=150)
print(f'Sample visualizations saved')

print(f'\n===== All outputs in {OUTPUT_DIR} =====')