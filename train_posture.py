import os, json, glob, sys, time
import cv2
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTPUT_DIR, 'cached_data.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================
# 1. Load and cache all data
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
            samples.append({
                'class': cls_name,
                'image': crop,
                'status': ann['status'],
                'posture': ann['posture'],
                'bad_reason': ann.get('bad_reason', []),
                'student_id': ann['student_id'],
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
print(f'Train: {train_classes}')
print(f'Val: {val_classes}')
print(f'Test: {test_classes}')

train_samples = [s for s in all_samples if s['class'] in train_classes]
val_samples = [s for s in all_samples if s['class'] in val_classes]
test_samples = [s for s in all_samples if s['class'] in test_classes]
print(f'Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

# ============================
# 3. In-memory Dataset
# ============================
samples_mean = np.array([0.485, 0.456, 0.406])
samples_std = np.array([0.229, 0.224, 0.225])

def prepare_data(samples, augment=False):
    images, status_labels, posture_labels, br_labels = [], [], [], []
    for s in samples:
        img = s['image'].astype(np.float32) / 255.0
        if augment and random.random() < 0.3:
            img = np.fliplr(img).copy()
        img = (img - samples_mean) / samples_std
        images.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
        status_labels.append(status_map[s['status']])
        posture_labels.append(posture_map[s['posture']])
        br = [0] * 4
        for r in s['bad_reason']:
            if r in bad_reason_map:
                br[bad_reason_map[r]] = 1
        br_labels.append(br)
    return (torch.stack(images),
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
    def __init__(self, num_status=4, num_posture=2, num_bad_reason=4):
        super().__init__()
        backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280
        self.status_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_status))
        self.posture_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_posture))
        self.bad_reason_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_bad_reason))

    def forward(self, x):
        f = self.backbone(x)
        f = self.avgpool(f).flatten(1)
        return (self.status_head(f), self.posture_head(f), self.bad_reason_head(f))

model = MultiTaskModel().to(DEVICE)

status_weights = torch.tensor([1.0, 1.0, 3.0, 3.0], device=DEVICE)
criterion_status = nn.CrossEntropyLoss(weight=status_weights)
criterion_posture = nn.CrossEntropyLoss()
criterion_bad_reason = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# ============================
# 5. Training
# ============================
def batch_iter(data, batch_size=64, shuffle=True):
    images, s_labels, p_labels, br_labels = data
    n = len(images)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]
        yield (images[batch_idx].to(DEVICE),
               s_labels[batch_idx].to(DEVICE),
               p_labels[batch_idx].to(DEVICE),
               br_labels[batch_idx].to(DEVICE))

def evaluate(data):
    model.eval()
    images, s_labels, p_labels, br_labels = data
    images, s_labels, p_labels, br_labels = (
        images.to(DEVICE), s_labels.to(DEVICE), p_labels.to(DEVICE), br_labels.to(DEVICE))
    with torch.no_grad():
        s_out, p_out, br_out = model(images)
        loss_s = criterion_status(s_out, s_labels)
        loss_p = criterion_posture(p_out, p_labels)
        loss_br = criterion_bad_reason(br_out, br_labels)
        loss = loss_s + loss_p + loss_br
    return (loss.item(),
            s_out.argmax(1).cpu().numpy(), s_labels.cpu().numpy(),
            p_out.argmax(1).cpu().numpy(), p_labels.cpu().numpy())

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_status_acc': [], 'val_posture_acc': []}

for epoch in range(30):
    t0 = time.time()
    model.train()
    total_loss = 0
    n_batches = 0
    for images, s_labels, p_labels, br_labels in batch_iter(train_data, batch_size=64):
        optimizer.zero_grad()
        s_out, p_out, br_out = model(images)
        loss = (criterion_status(s_out, s_labels) +
                criterion_posture(p_out, p_labels) +
                criterion_bad_reason(br_out, br_labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    val_loss, val_s_pred, val_s_true, val_p_pred, val_p_true = evaluate(val_data)
    status_acc = accuracy_score(val_s_true, val_s_pred)
    posture_acc = accuracy_score(val_p_true, val_p_pred)

    history['train_loss'].append(total_loss / n_batches)
    history['val_loss'].append(val_loss)
    history['val_status_acc'].append(status_acc)
    history['val_posture_acc'].append(posture_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))

    elapsed = time.time() - t0
    print(f'Epoch {epoch+1:2d} ({elapsed:.0f}s): train_loss={total_loss/n_batches:.4f}, '
          f'val_loss={val_loss:.4f}, status_acc={status_acc:.4f}, posture_acc={posture_acc:.4f}')
    scheduler.step()

# ============================
# 6. Test Evaluation
# ============================
print('\n===== TEST EVALUATION =====')
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

test_loss, test_s_pred, test_s_true, test_p_pred, test_p_true = evaluate(test_data)
test_br_pred = []
test_br_true = []
with torch.no_grad():
    for images, _, _, br_labels in [test_data]:
        images = images.to(DEVICE)
        _, _, br_out = model(images)
        test_br_pred = (torch.sigmoid(br_out) > 0.5).cpu().numpy()
        test_br_true = br_labels.numpy()

status_rev = {v: k for k, v in status_map.items()}
posture_rev = {v: k for k, v in posture_map.items()}
br_rev = {v: k for k, v in bad_reason_map.items()}

print(f'\nTest loss: {test_loss:.4f}')
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

# Per-class confusion analysis
print(f'\n--- Per-Class Breakdown ---')
for cls_name in test_classes:
    cls_idx = [i for i, s in enumerate(test_samples) if s['class'] == cls_name]
    if not cls_idx:
        continue
    cls_true = test_s_true[cls_idx]
    cls_pred = test_s_pred[cls_idx]
    cls_p_true = test_p_true[cls_idx]
    cls_p_pred = test_p_pred[cls_idx]
    print(f'  {cls_name}: status_acc={accuracy_score(cls_true, cls_pred):.4f}, '
          f'posture_acc={accuracy_score(cls_p_true, cls_p_pred):.4f} '
          f'({len(cls_idx)} samples)')

# ============================
# 7. Confusion Matrix
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, labels, title in [
    (axes[0], test_s_true, test_s_pred, [status_rev[i] for i in range(4)], 'Status'),
    (axes[1], test_p_true, test_p_pred, [posture_rev[i] for i in range(2)], 'Posture'),
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
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150)
print(f'\nConfusion matrices saved')

# ============================
# 8. Training curves
# ============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Loss')

axes[1].plot(history['val_status_acc'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Status Validation Accuracy')

axes[2].plot(history['val_posture_acc'])
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Posture Validation Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
print(f'Training curves saved')

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

print(f'\n===== All outputs in {OUTPUT_DIR} =====')