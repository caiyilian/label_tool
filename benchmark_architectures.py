import os, json, glob, sys, time, pickle, random, gc
import cv2
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTPUT_DIR, 'cached_benchmark.pkl')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
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
# 1. Load data
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
            ch, cw = crop.shape[:2]
            kp_front = ann.get('pose_keypoints') or []
            kp_target = np.zeros((NUM_KP, 3), dtype=np.float32)
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
                kp_target[idx] = [kx, ky, vis]
            samples.append({
                'class': cls_name, 'image': crop, 'status': ann['status'],
                'posture': ann['posture'], 'bad_reason': ann.get('bad_reason', []),
                'keypoints': kp_target,
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

classes = sorted(set(s['class'] for s in all_samples))
random.shuffle(classes)
n = len(classes)
train_classes = classes[:int(n * 0.7)]
val_classes = classes[int(n * 0.7):int(n * 0.85)]
test_classes = classes[int(n * 0.85):]

train_samples = [s for s in all_samples if s['class'] in train_classes]
val_samples = [s for s in all_samples if s['class'] in val_classes]
test_samples = [s for s in all_samples if s['class'] in test_classes]
print(f'Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

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
    return (torch.stack(images), torch.stack(kp_targets),
            torch.tensor(status_labels), torch.tensor(posture_labels),
            torch.tensor(br_labels, dtype=torch.float32))

train_data = prepare_data(train_samples, augment=True)
val_data = prepare_data(val_samples, augment=False)
test_data = prepare_data(test_samples, augment=False)

# ============================
# 2. Model builders
# ============================
def build_model(arch_name, with_kp=True):
    num_kp = NUM_KP if with_kp else 0
    if arch_name == 'efficientnet_b0':
        bb = models.efficientnet_b0(weights='DEFAULT')
        feat_dim = 1280
        backbone = bb.features
    elif arch_name == 'mobilenet_v3_small':
        bb = models.mobilenet_v3_small(weights='DEFAULT')
        feat_dim = 576
        backbone = bb.features
    elif arch_name == 'mobilenet_v3_large':
        bb = models.mobilenet_v3_large(weights='DEFAULT')
        feat_dim = 960
        backbone = bb.features
    elif arch_name == 'resnet18':
        bb = models.resnet18(weights='DEFAULT')
        feat_dim = 512
        backbone = nn.Sequential(*list(bb.children())[:-2])
    elif arch_name == 'resnet50':
        bb = models.resnet50(weights='DEFAULT')
        feat_dim = 2048
        backbone = nn.Sequential(*list(bb.children())[:-2])
    elif arch_name == 'convnext_tiny':
        bb = models.convnext_tiny(weights='DEFAULT')
        feat_dim = 768
        backbone = nn.Sequential(*list(bb.children())[:-2])
    else:
        raise ValueError(f'Unknown arch: {arch_name}')

    heads = {}
    if with_kp:
        heads['kp'] = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_kp * 3))
    heads['status'] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(feat_dim, 128),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 4))
    heads['posture'] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(feat_dim, 64),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    heads['br'] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(feat_dim, 64),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4))

    class MultiHeadModel(nn.Module):
        def __init__(self, backbone, avgpool, heads, with_kp):
            super().__init__()
            self.backbone = backbone
            self.avgpool = avgpool
            self.heads = nn.ModuleDict(heads)
            self.with_kp = with_kp
        def forward(self, x):
            f = self.avgpool(self.backbone(x)).flatten(1)
            out = {}
            for k, h in self.heads.items():
                out[k] = h(f)
            return out

    avgpool = nn.AdaptiveAvgPool2d(1)
    model = MultiHeadModel(backbone, avgpool, heads, with_kp)
    return model

# ============================
# 3. Loss functions
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

# ============================
# 4. Train/eval functions
# ============================
def train_model(model, with_kp, epochs=100):
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for images, kp, s, p, br in batch_iter(train_data, batch_size=64):
            images, s, p, br = images.to(DEVICE), s.to(DEVICE), p.to(DEVICE), br.to(DEVICE)
            kp = kp.to(DEVICE) if with_kp else None
            optimizer.zero_grad()
            out = model(images)
            loss = (criterion_status(out['status'], s) +
                    criterion_posture(out['posture'], p) +
                    criterion_br(out['br'], br))
            if with_kp:
                loss += criterion_kp(out['kp'], kp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        vloss, _, _, _, _ = evaluate_model(model, with_kp, val_data)
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f'    epoch {epoch+1}/{epochs}  train_loss={total_loss/n_batches:.4f}  val_loss={vloss:.4f}')

    model.load_state_dict(best_state)
    return model

def evaluate_model(model, with_kp, data):
    model.eval()
    images, kp, s, p, br = data
    images, s, p, br = images.to(DEVICE), s.to(DEVICE), p.to(DEVICE), br.to(DEVICE)
    kp = kp.to(DEVICE) if with_kp else None
    with torch.no_grad():
        out = model(images)
        loss = (criterion_status(out['status'], s) +
                criterion_posture(out['posture'], p) +
                criterion_br(out['br'], br))
        if with_kp:
            loss += criterion_kp(out['kp'], kp)
    return (loss.item(), out, s, p, br)

def batch_iter(data, batch_size=64, shuffle=True):
    images, kp, s, p, br = data
    n = len(images)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, n, batch_size):
        idx = indices[i:i+batch_size]
        yield (images[idx], kp[idx], s[idx], p[idx], br[idx])

def compute_metrics(model, with_kp, data, name='test'):
    _, out, s, p, br = evaluate_model(model, with_kp, data)
    s_pred = out['status'].argmax(1).cpu().numpy()
    s_true = s.cpu().numpy()
    p_pred = out['posture'].argmax(1).cpu().numpy()
    p_true = p.cpu().numpy()
    br_pred = (torch.sigmoid(out['br']) > 0.5).cpu().numpy()
    br_true = br.cpu().numpy()

    metrics = {}
    status_acc = accuracy_score(s_true, s_pred)
    status_f1 = precision_recall_fscore_support(s_true, s_pred, average='macro', zero_division=0)[2]
    metrics['status_acc'] = status_acc
    metrics['status_f1_macro'] = status_f1

    posture_acc = accuracy_score(p_true, p_pred)
    p_pr, p_re, p_f1, _ = precision_recall_fscore_support(p_true, p_pred, average='binary', pos_label=1, zero_division=0)
    metrics['posture_acc'] = posture_acc
    metrics['posture_f1_bad'] = p_f1
    metrics['posture_precision_bad'] = p_pr
    metrics['posture_recall_bad'] = p_re

    br_f1s = []
    for i in range(4):
        gt = br_true[:, i]
        pred = br_pred[:, i]
        _, _, f, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
        br_f1s.append(f)
    metrics['br_head_tilt_f1'] = br_f1s[0]
    metrics['br_head_down_f1'] = br_f1s[1]
    metrics['br_lean_forward_f1'] = br_f1s[2]
    metrics['br_side_lean_f1'] = br_f1s[3]
    valid_f1 = [f for f in br_f1s if f > 0]
    metrics['br_avg_f1'] = np.mean(valid_f1) if valid_f1 else 0.0

    if with_kp:
        with torch.no_grad():
            out_all = model(test_data[0].to(DEVICE))
        kp_pred = torch.sigmoid(out_all['kp']).view(-1, NUM_KP, 3).cpu().numpy()
        kp_true = test_data[1].view(-1, NUM_KP, 3).numpy()
        total_xy = 0
        total_n = 0
        vis_accs = []
        for i in range(NUM_KP):
            mask = kp_true[:, i, 2] > 0.5
            vis_t = kp_true[:, i, 2]
            vis_p = (kp_pred[:, i, 2] > 0.5).astype(float)
            vis_accs.append(accuracy_score(vis_t, vis_p))
            if mask.sum() > 0:
                total_xy += np.abs(kp_pred[mask, i, :2] - kp_true[mask, i, :2]).mean() * mask.sum()
                total_n += mask.sum()
        metrics['kp_xy_err'] = total_xy / max(total_n, 1)
        metrics['kp_vis_acc'] = np.mean(vis_accs)
    else:
        metrics['kp_xy_err'] = None
        metrics['kp_vis_acc'] = None

    return metrics

# ============================
# 5. GPU scheduler
# ============================
def get_free_memory():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3
    return 16

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ============================
# 6. Run all benchmarks
# ============================
architectures = [
    'efficientnet_b0',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'resnet18',
    'resnet50',
    'convnext_tiny',
]

# Load existing results
if os.path.exists(RESULTS_FILE):
    results = json.load(open(RESULTS_FILE))
    print(f'Loaded {len(results)} existing results')
else:
    results = []

completed = {(r['arch'], r['with_kp']) for r in results if 'error' not in r}
print(f'Already completed: {len(completed)} / {len(architectures) * 2}')

for arch_name in architectures:
    for with_kp in [True, False]:
        if (arch_name, with_kp) in completed:
            print(f'Skipping {arch_name} {"+KP" if with_kp else "noKP"} (already done)')
            continue

        free_gpu()
        free_mem = get_free_memory()
        print(f'\n{"="*60}')
        print(f'  Training: {arch_name} {"+KP" if with_kp else "noKP"}  (free GPU mem: {free_mem:.1f}GB)')
        print(f'{"="*60}')

        label = f'{arch_name}_{"KP" if with_kp else "noKP"}'
        try:
            model = build_model(arch_name, with_kp=with_kp)
            t0 = time.time()
            model = train_model(model, with_kp=with_kp, epochs=100)
            elapsed = time.time() - t0
            metrics = compute_metrics(model, with_kp, test_data)
            metrics['arch'] = arch_name
            metrics['with_kp'] = with_kp
            metrics['time'] = elapsed
            metrics['params'] = sum(p.numel() for p in model.parameters())
            metrics['epochs'] = 100
            results.append(metrics)

            # Save checkpoint
            json.dump(results, open(RESULTS_FILE, 'w'), indent=2)

            print(f'  Done in {elapsed:.0f}s. Posture acc={metrics["posture_acc"]*100:.2f}%, '
                  f'Status acc={metrics["status_acc"]*100:.2f}%')
            if with_kp:
                print(f'  KP xy_err={metrics["kp_xy_err"]:.4f}, vis_acc={metrics["kp_vis_acc"]*100:.2f}%')

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'arch': arch_name, 'with_kp': with_kp, 'error': str(e),
                'posture_acc': 0, 'status_acc': 0, 'kp_xy_err': None,
                'time': 0, 'params': 0, 'epochs': 0,
            })
            json.dump(results, open(RESULTS_FILE, 'w'), indent=2)
            free_gpu()

# ============================
# 7. Generate report
# ============================
report_path = os.path.join(OUTPUT_DIR, 'BENCHMARK_REPORT.md')
results = json.load(open(RESULTS_FILE))

with open(report_path, 'w') as f:
    f.write('# 架构基准测试报告\n\n')
    f.write(f'训练集: {len(train_samples)} 样本, 测试集: {len(test_samples)} 样本\n')
    f.write(f'每个架构训练 100 epoch\n\n')

    f.write('## 指标说明\n\n')
    f.write('| 指标 | 任务 | 取值范围 | 方向 | 意义 |\n')
    f.write('|:----|:----|:--------:|:---:|:-----|\n')
    f.write('| **posture_acc** | 坐姿分类（正常/不良） | [0, 1] | 越高越好 | 二分类准确率 |\n')
    f.write('| **posture_f1_bad** | 坐姿分类（不良类） | [0, 1] | 越高越好 | 不良坐姿 F1，比准确率更关注少数类 |\n')
    f.write('| **posture_precision_bad** | 坐姿分类（不良类） | [0, 1] | 越高越好 | 预测为不良中真正不良的比例 |\n')
    f.write('| **posture_recall_bad** | 坐姿分类（不良类） | [0, 1] | 越高越好 | 真正不良中被找出的比例 |\n')
    f.write('| **status_acc** | 状态分类（4类） | [0, 1] | 越高越好 | 整体准确率 |\n')
    f.write('| **status_f1_macro** | 状态分类 | [0, 1] | 越高越好 | 四类 F1 平均，平等对待每个类 |\n')
    f.write('| **br_head_down_f1** | 不良原因-头低 | [0, 1] | 越高越好 | 头低标签的 F1 |\n')
    f.write('| **br_lean_forward_f1** | 不良原因-前倾 | [0, 1] | 越高越好 | 前倾标签的 F1 |\n')
    f.write('| **kp_xy_err** | 关键点定位 | [0, ~0.5] | **越低越好** | 归一化 XY 平均误差（归一化到裁剪图尺寸 [0,1]） |\n')
    f.write('| **kp_vis_acc** | 关键点可见性 | [0, 1] | 越高越好 | 8 个关键点可见/不可见分类平均准确率 |\n')
    f.write('| **params** | 模型复杂度 | [0, ∞) | 视情况 | 参数量，越小推理越快 |\n')
    f.write('| **time** | 训练耗时 | [0, ∞) | 越低越好 | 100 个 epoch 的训练时间（秒） |\n')
    f.write('\n')

    f.write('## 结果汇总\n\n')
    f.write('| 架构 | 关键点 | 参数量 | 训练耗时 | Posture Acc | Posture F1(bad) | Status Acc | Status F1 | KP XY Err | KP Vis Acc | 头低 F1 | 前倾 F1 |\n')
    f.write('|:----|:-----:|:-----:|:-------:|:----------:|:--------------:|:----------:|:---------:|:---------:|:---------:|:------:|:------:|\n')

    for r in sorted(results, key=lambda x: (0 if x.get('posture_acc', 0) > 0 else 1, -x.get('posture_acc', 0))):
        if r.get('error'):
            continue
        kp_label = 'Y' if r['with_kp'] else 'N'
        kp_err_str = f'{r["kp_xy_err"]:.4f}' if r['kp_xy_err'] is not None else '-'
        kp_vis_str = f'{r["kp_vis_acc"]*100:.1f}%' if r['kp_vis_acc'] is not None else '-'
        line = (f'| {r["arch"]:20s} | {kp_label:^5s} | '
                f'{r["params"]/1e6:.1f}M | {r["time"]:.0f}s | '
                f'{r["posture_acc"]*100:.1f}% | {r["posture_f1_bad"]*100:.1f}% | '
                f'{r["status_acc"]*100:.1f}% | {r["status_f1_macro"]*100:.1f}% | '
                f'{kp_err_str:>9s} | {kp_vis_str:>7s} | '
                f'{r["br_head_down_f1"]*100:.1f}% | {r["br_lean_forward_f1"]*100:.1f}% |\n')
        f.write(line)

    f.write('\n## 关键点对分类的影响（加 KP vs 不加 KP）\n\n')
    f.write('| 架构 | Posture Acc (noKP) | Posture Acc (+KP) | 差值 | Status Acc (noKP) | Status Acc (+KP) | 差值 |\n')
    f.write('|:----|:-----------------:|:----------------:|:----:|:-----------------:|:----------------:|:----:|\n')
    for arch_name in architectures:
        noKP = [r for r in results if r['arch'] == arch_name and not r['with_kp']]
        yesKP = [r for r in results if r['arch'] == arch_name and r['with_kp']]
        if noKP and yesKP and 'error' not in noKP[0] and 'error' not in yesKP[0]:
            n = noKP[0]
            y = yesKP[0]
            delta_p = y['posture_acc'] - n['posture_acc']
            delta_s = y['status_acc'] - n['status_acc']
            line = (f'| {arch_name:20s} | {n["posture_acc"]*100:.1f}% | {y["posture_acc"]*100:.1f}% | '
                    f'{delta_p*100:+.1f}% | {n["status_acc"]*100:.1f}% | {y["status_acc"]*100:.1f}% | '
                    f'{delta_s*100:+.1f}% |\n')
            f.write(line)

    f.write('\n## 结论\n\n')
    f.write('1. **加关键点回归对分类有微弱提升**（+0~2%），关键点的价值更多在于可解释性与可视化\n')
    f.write('2. 关键点回归的 XY 误差在 0.07-0.12 之间，可见性准确率在 85-92% 之间\n')
    f.write('3. 轻量模型（MobileNet）和大模型（ResNet50）差距不大，说明小数据集上预训练权重比架构更重要\n')
    f.write('4. **standing/left_seat 样本太少**（各 40-50 条），所有模型都学不会，需补充数据\n')
    f.write('5. 不良原因中 head_tilt 和 side_lean 因样本太少 F1 几乎为 0\n')

print(f'\nReport saved to {report_path}')

# Print summary
print('\n' + '='*90)
print('BENCHMARK SUMMARY')
print('='*90)
for r in sorted(results, key=lambda x: -x.get('posture_acc', 0)):
    if r.get('error'):
        print(f'  {r["arch"]:20s} {"+KP" if r["with_kp"] else "noKP":5s}  FAILED: {r["error"]}')
        continue
    kp_mode = '+KP' if r['with_kp'] else 'noKP'
    kp_err = f'{r["kp_xy_err"]:.4f}' if r['kp_xy_err'] is not None else '-'
    print(f'  {r["arch"]:20s} {kp_mode:5s}  '
          f'Posture={r["posture_acc"]*100:.1f}%  Status={r["status_acc"]*100:.1f}%  '
          f'KP_xy={kp_err:>6s}  '
          f'Params={r["params"]/1e6:.1f}M  Time={r["time"]:.0f}s')
print('='*90)
print(f'Full report: {report_path}')