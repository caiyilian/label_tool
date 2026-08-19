import os, json, glob, sys, time, pickle, random
import cv2
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, r2_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTPUT_DIR, 'cached_multitask.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
NUM_KP = len(KP_NAMES)  # 8

# ============================
# 1. Load and cache data
# ============================
def load_all_data():
    samples = []
    ann_files = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    for fpath in ann_files:
        rel = fpath.replace(BASE, '').lstrip('/')
        cls_name = rel.split('/')[0]
        with open(fpath) as f:
            data = json.load(f)
        forward_path = os.path.join(
            os.path.dirname(fpath).replace('annotations', 'forward'),
            os.path.basename(fpath).replace('.json', '.jpg'))
        img = cv2.imread(forward_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for ann in data.get('annotations', []):
            bbox = ann['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            crop_h, crop_w = crop.shape[:2]

            kp_front = ann.get('pose_keypoints') or []
            kp_target = np.zeros((NUM_KP, 3), dtype=np.float32)
            for kp in kp_front:
                if kp is None:
                    continue
                idx = KP_NAMES.index(kp['name']) if kp['name'] in KP_NAMES else -1
                if idx < 0:
                    continue
                visible = 1.0 if kp.get('visible') else 0.0
                if visible:
                    kx = (kp['x'] - x1) / max(crop_w, 1)
                    ky = (kp['y'] - y1) / max(crop_h, 1)
                    kx = np.clip(kx, 0, 1)
                    ky = np.clip(ky, 0, 1)
                else:
                    kx, ky = 0.0, 0.0
                kp_target[idx] = [kx, ky, visible]

            samples.append({
                'class': cls_name,
                'image': crop,
                'status': ann['status'],
                'posture': ann['posture'],
                'bad_reason': ann.get('bad_reason', []),
                'keypoints': kp_target,
                'student_id': ann['student_id'],
                'frame_id': data.get('frame_id', ''),
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

# ============================
# 3. Prepare tensors
# ============================
samples_mean = np.array([0.485, 0.456, 0.406])
samples_std = np.array([0.229, 0.224, 0.225])

def prepare_data(samples, augment=False):
    images, kp_targets, status_labels, posture_labels, br_labels = [], [], [], [], []
    for s in samples:
        img = s['image'].astype(np.float32) / 255.0
        if augment and random.random() < 0.3:
            img = np.fliplr(img).copy()
        img = (img - samples_mean) / samples_std
        images.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
        kp_targets.append(torch.from_numpy(s['keypoints'].flatten()).float())
        status_labels.append(status_map[s['status']])
        posture_labels.append(posture_map[s['posture']])
        br = [0] * 4
        for r in s['bad_reason']:
            if r in bad_reason_map:
                br[bad_reason_map[r]] = 1
        br_labels.append(br)
    return (torch.stack(images),
            torch.stack(kp_targets),
            torch.tensor(status_labels),
            torch.tensor(posture_labels),
            torch.tensor(br_labels, dtype=torch.float32))

print('Preparing tensors...')
train_data = prepare_data(train_samples, augment=True)
val_data = prepare_data(val_samples, augment=False)
test_data = prepare_data(test_samples, augment=False)

# ============================
# 4. Multi-task Model
# ============================
class MultiTaskModel(nn.Module):
    def __init__(self, num_kp=8, num_status=4, num_posture=2, num_br=4):
        super().__init__()
        backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        self.kp_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_kp * 3))
        self.status_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_status))
        self.posture_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_posture))
        self.br_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_br))

    def forward(self, x):
        f = self.backbone(x)
        f = self.avgpool(f).flatten(1)
        return (self.kp_head(f), self.status_head(f),
                self.posture_head(f), self.br_head(f))

model = MultiTaskModel().to(DEVICE)

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

criterion_kp = kp_loss
status_weights = torch.tensor([1.0, 1.0, 3.0, 3.0], device=DEVICE)
criterion_status = nn.CrossEntropyLoss(weight=status_weights)
criterion_posture = nn.CrossEntropyLoss()
criterion_br = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# ============================
# 6. Training
# ============================
def batch_iter(data, batch_size=64, shuffle=True):
    images, kp, s, p, br = data
    n = len(images)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, n, batch_size):
        idx = indices[i:i+batch_size]
        yield (images[idx].to(DEVICE), kp[idx].to(DEVICE),
               s[idx].to(DEVICE), p[idx].to(DEVICE), br[idx].to(DEVICE))

def evaluate(data):
    model.eval()
    images, kp, s, p, br = data
    images, kp, s, p, br = (images.to(DEVICE), kp.to(DEVICE),
                             s.to(DEVICE), p.to(DEVICE), br.to(DEVICE))
    with torch.no_grad():
        kp_out, s_out, p_out, br_out = model(images)
        loss = (criterion_kp(kp_out, kp) + criterion_status(s_out, s) +
                criterion_posture(p_out, p) + criterion_br(br_out, br))
    return (loss.item(), kp_out, s_out, p_out, br_out, s, p, br, kp)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_kp_xy_err': [],
           'val_status_acc': [], 'val_posture_acc': []}

for epoch in range(30):
    t0 = time.time()
    model.train()
    total_loss = 0
    n_batches = 0
    for images, kp, s, p, br in batch_iter(train_data, batch_size=64):
        optimizer.zero_grad()
        kp_out, s_out, p_out, br_out = model(images)
        loss = (criterion_kp(kp_out, kp) + criterion_status(s_out, s) +
                criterion_posture(p_out, p) + criterion_br(br_out, br))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    val_loss, val_kp_out, val_s_out, val_p_out, val_br_out, val_s, val_p, val_br, val_kp = evaluate(val_data)
    val_s_acc = accuracy_score(val_s.cpu().numpy(), val_s_out.argmax(1).cpu().numpy())
    val_p_acc = accuracy_score(val_p.cpu().numpy(), val_p_out.argmax(1).cpu().numpy())

    kp_true = val_kp.view(-1, NUM_KP, 3)
    kp_pred = torch.sigmoid(val_kp_out).view(-1, NUM_KP, 3)
    vis_mask = (kp_true[:, :, 2] > 0.5).float()
    xy_err = ((kp_pred[:, :, :2] - kp_true[:, :, :2]).abs() * vis_mask.unsqueeze(-1)).sum() / (vis_mask.sum() + 1e-6)

    history['train_loss'].append(total_loss / n_batches)
    history['val_loss'].append(val_loss)
    history['val_kp_xy_err'].append(xy_err.item())
    history['val_status_acc'].append(val_s_acc)
    history['val_posture_acc'].append(val_p_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_multitask.pth'))

    elapsed = time.time() - t0
    print(f'Epoch {epoch+1:2d} ({elapsed:.0f}s): train_loss={total_loss/n_batches:.4f}, '
          f'val_loss={val_loss:.4f}, kp_xy_err={xy_err.item():.4f}, '
          f'status_acc={val_s_acc:.4f}, posture_acc={val_p_acc:.4f}')
    scheduler.step()

# ============================
# 7. Test Evaluation
# ============================
print('\n===== TEST EVALUATION =====')
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_multitask.pth')))
model.eval()

test_loss, test_kp_out, test_s_out, test_p_out, test_br_out, test_s, test_p, test_br, test_kp = evaluate(test_data)
test_s_pred = test_s_out.argmax(1).cpu().numpy()
test_s_true = test_s.cpu().numpy()
test_p_pred = test_p_out.argmax(1).cpu().numpy()
test_p_true = test_p.cpu().numpy()
test_br_pred = (torch.sigmoid(test_br_out) > 0.5).cpu().numpy()
test_br_true = test_br.cpu().numpy()

status_rev = {v: k for k, v in status_map.items()}
posture_rev = {v: k for k, v in posture_map.items()}
br_rev = {v: k for k, v in bad_reason_map.items()}

# Keypoint evaluation
kp_true_all = test_kp.view(-1, NUM_KP, 3).cpu().numpy()
kp_pred_all = torch.sigmoid(test_kp_out).view(-1, NUM_KP, 3).cpu().numpy()

print(f'\n--- Keypoint Estimation ---')
total_xy_err = 0
total_vis = 0
kp_vis_acc = 0
kp_vis_total = 0
for i in range(NUM_KP):
    vis_true = kp_true_all[:, i, 2]
    vis_pred = (kp_pred_all[:, i, 2] > 0.5).astype(float)
    vis_acc = accuracy_score(vis_true, vis_pred)
    mask = vis_true > 0.5
    xy_err = np.abs(kp_pred_all[mask, i, :2] - kp_true_all[mask, i, :2]).mean() if mask.sum() > 0 else 0
    print(f'  {KP_NAMES[i]:15s}: xy_err={xy_err:.4f}, vis_acc={vis_acc:.4f} '
          f'(visible: {int(mask.sum())}/{len(mask)})')
    total_xy_err += xy_err * mask.sum()
    total_vis += mask.sum()
    kp_vis_acc += vis_acc * len(vis_true)
    kp_vis_total += len(vis_true)
print(f'  {"AVERAGE":15s}: xy_err={total_xy_err/max(total_vis,1):.4f}, '
      f'vis_acc={kp_vis_acc/max(kp_vis_total,1):.4f}')

print(f'\n--- Status Classification ---')
print(f'Accuracy: {accuracy_score(test_s_true, test_s_pred):.4f}')
print(classification_report(test_s_true, test_s_pred,
                            target_names=[status_rev[i] for i in range(4)], zero_division=0))

print(f'\n--- Posture Classification ---')
print(f'Accuracy: {accuracy_score(test_p_true, test_p_pred):.4f}')
print(classification_report(test_p_true, test_p_pred,
                            target_names=[posture_rev[i] for i in range(2)], zero_division=0))

print(f'\n--- Bad Reason (Multi-label) ---')
for i in range(4):
    gt = test_br_true[:, i]
    pred = test_br_pred[:, i]
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

axes[0, 1].plot(history['val_kp_xy_err'])
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('XY Error')
axes[0, 1].set_title('Keypoint XY Error (Val)')

axes[0, 2].plot(history['val_status_acc'], label='Status')
axes[0, 2].plot(history['val_posture_acc'], label='Posture')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].legend()
axes[0, 2].set_title('Classification Accuracy (Val)')

for ax, y_true, y_pred, labels, title in [
    (axes[1, 0], test_s_true, test_s_pred, [status_rev[i] for i in range(4)], 'Status CM'),
    (axes[1, 1], test_p_true, test_p_pred, [posture_rev[i] for i in range(2)], 'Posture CM'),
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

# Per-keypoint bar chart
ax = axes[1, 2]
names = []
xerrs = []
vaccs = []
for i in range(NUM_KP):
    mask = kp_true_all[:, i, 2] > 0.5
    if mask.sum() > 0:
        err = np.abs(kp_pred_all[mask, i, :2] - kp_true_all[mask, i, :2]).mean()
    else:
        err = 0
    names.append(KP_NAMES[i])
    xerrs.append(err)
    vis_true = kp_true_all[:, i, 2]
    vis_pred = (kp_pred_all[:, i, 2] > 0.5).astype(float)
    vaccs.append(accuracy_score(vis_true, vis_pred))
x = range(len(names))
ax.bar(x, xerrs, alpha=0.7, label='XY Error')
ax2 = ax.twinx()
ax2.plot(x, vaccs, 'ro-', label='Vis Acc')
ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('XY Error')
ax2.set_ylabel('Visibility Acc')
ax.set_title('Keypoint Performance')
from matplotlib.lines import Line2D
ax.legend([Line2D([0],[0],color='blue',lw=4), Line2D([0],[0],color='red',lw=2)],
          ['XY Error', 'Vis Acc'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'multitask_results.png'), dpi=150)
print(f'\nResults saved to {OUTPUT_DIR}/multitask_results.png')

# ============================
# 9. Data distribution
# ============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in [
    (axes[0], 'status', 'Status Distribution'),
    (axes[1], 'posture', 'Posture Distribution'),
    (axes[2], 'bad_reason', 'Bad Reason Distribution'),
]:
    counter = Counter()
    for s in all_samples:
        if key == 'status':
            counter[s['status']] += 1
        elif key == 'posture':
            counter[s['posture']] += 1
        elif key == 'bad_reason':
            for r in s['bad_reason']:
                counter[r] += 1
    labels, values = zip(*counter.items())
    ax.bar(labels, values)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'data_distribution.png'), dpi=150)
print(f'Data distribution saved')

# ============================
# 10. Sample keypoint visualization
# ============================
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()
vis_idx = np.where((kp_true_all[:, :, 2].sum(axis=1) > 0))[0]
if len(vis_idx) > 0:
    chosen = np.random.choice(vis_idx, min(16, len(vis_idx)), replace=False)
    for ax_i, idx in enumerate(chosen):
        img_tensor = test_data[0][idx].cpu()
        img = img_tensor.numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[ax_i].imshow(img)
        for k in range(NUM_KP):
            if kp_true_all[idx, k, 2] > 0.5:
                tx, ty = kp_true_all[idx, k, :2]
                px, py = kp_pred_all[idx, k, :2]
                axes[ax_i].scatter(tx * 224, ty * 224, c='g', s=30, marker='o')
                axes[ax_i].scatter(px * 224, py * 224, c='r', s=20, marker='x')
        axes[ax_i].axis('off')
        axes[ax_i].set_title(f'True=green, Pred=red' if ax_i == 0 else '')
    for ax_i in range(len(chosen), 16):
        axes[ax_i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'keypoint_samples.png'), dpi=150)
print(f'Keypoint samples saved')

print(f'\n===== All outputs in {OUTPUT_DIR} =====')