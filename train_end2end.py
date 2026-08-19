import os, json, glob, random, time, copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import Pose
from ultralytics import YOLO
from PIL import Image

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}
KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
NUM_KP = 8

# ========== 1. 数据 ==========
def load_dataset():
    anns = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    samples = []
    for f in anns:
        cls = f.replace(BASE, '').lstrip('/').split('/')[0]
        with open(f) as fh: data = json.load(fh)
        jpg = os.path.basename(f).replace('.json', '.jpg')
        fwd = os.path.join(os.path.dirname(f).replace('annotations','forward'), jpg)
        img = cv2.imread(fwd)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labels = []
        for ann in data['annotations']:
            kpf = ann.get('pose_keypoints') or []
            kp = np.zeros((NUM_KP, 3), dtype=np.float32)
            for k in kpf:
                if k is None or k['name'] not in KP_NAMES: continue
                idx = KP_NAMES.index(k['name'])
                vis = 1.0 if k.get('visible') else 0.0
                kp[idx] = [k['x'], k['y'], vis] if vis else [0,0,0]
            br = [0]*4
            for r in ann.get('bad_reason', []):
                if r in bad_reason_map: br[bad_reason_map[r]] = 1
            labels.append({'bbox': ann['bbox'], 'kp': kp,
                'status': status_map[ann['status']], 'posture': posture_map[ann['posture']], 'br': br})
        samples.append({'cls': cls, 'img': img, 'labels': labels, 'h': h, 'w': w})
    return samples

all_samples = load_dataset()
classes = sorted(set(s['cls'] for s in all_samples))
random.seed(42); random.shuffle(classes)
train_cls = set(classes[:int(len(classes)*0.7)])
val_cls = set(classes[int(len(classes)*0.7):int(len(classes)*0.85)])
test_cls = set(classes[int(len(classes)*0.85):])
train_data = [s for s in all_samples if s['cls'] in train_cls]
val_data = [s for s in all_samples if s['cls'] in val_cls]
test_data = [s for s in all_samples if s['cls'] in test_cls]
print(f'Train: {len(train_data)} imgs, Val: {len(val_data)} imgs, Test: {len(test_data)} imgs')

# ========== 2. 自定义多任务 head ==========
# 替换 YOLOv8n-pose 的 head 为多任务 head
# 在 Pose head 的基础上加一个 cv5 分类分支
# 加载模型，直接在原 Pose head 上加 cv5 分支
print('Loading YOLOv8n-pose...')
m = YOLO('yolov8n-pose.pt')
head = m.model.model[-1]
# 获取实际特征图通道数（从 cv2 第一个卷积的输入通道获取）
ch = [head.cv2[i][0].conv.in_channels for i in range(3)]
print(f'Feature map channels: {ch}')

# 在原 head 上添加 cv5 分类分支
n_extra = 10
c5 = max(ch[0] // 4, n_extra)
head.cv5 = nn.ModuleList(
    nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, n_extra, 1)) for x in ch)

# 猴补丁 forward 方法
orig_forward = head.forward
def patched_forward(self, x):
    for i in range(self.nl):
        x[i] = torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i]), self.cv5[i](x[i])], 1)
    return x
head.forward = patched_forward.__get__(head, type(head))

head.f = -1
head.i = 22
m.model.model[-1] = head
model = m.model.to(DEVICE)
print(f'Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# 只训练 cv5 和新加的 head 部分
for name, p in model.named_parameters():
    if 'cv5' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable/1e6:.2f}M (only cv5)')

# ========== 3. 数据集 ==========
class E2EDataset(torch.utils.data.Dataset):
    def __init__(self, samples, imgsz=640, augment=False):
        self.samples = samples
        self.imgsz = imgsz
        self.augment = augment
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = s['img']
        h, w = s['h'], s['w']
        r = self.imgsz / max(h, w)
        new_h, new_w = int(h * r), int(w * r)
        img = cv2.resize(img, (new_w, new_h))
        dh, dw = self.imgsz - new_h, self.imgsz - new_w
        top, left = dh // 2, dw // 2
        img = cv2.copyMakeBorder(img, top, self.imgsz - new_h - top, left, self.imgsz - new_w - left,
                                 cv2.BORDER_CONSTANT, value=(114,114,114))
        img = np.transpose(img.astype(np.float32)/255.0, (2, 0, 1))
        labels = []
        for l in s['labels']:
            x1, y1, x2, y2 = l['bbox']
            x1 = x1 * r + left; y1 = y1 * r + top
            x2 = x2 * r + left; y2 = y2 * r + top
            kp = l['kp'].copy()
            for i in range(NUM_KP):
                if kp[i, 2] > 0.5:
                    kp[i, 0] = kp[i, 0] * r + left
                    kp[i, 1] = kp[i, 1] * r + top
            labels.append({'bbox': [x1, y1, x2, y2], 'kp': kp,
                           'status': l['status'], 'posture': l['posture'], 'br': l['br']})
        return torch.from_numpy(img).float(), labels

def collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels

BS = 8
train_loader = torch.utils.data.DataLoader(E2EDataset(train_data), BS, True, num_workers=4, collate_fn=collate)
val_loader = torch.utils.data.DataLoader(E2EDataset(val_data), BS, False, num_workers=4, collate_fn=collate)
test_loader = torch.utils.data.DataLoader(E2EDataset(test_data), BS, False, num_workers=4, collate_fn=collate)

# ========== 4. 训练 ==========
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

imgsz = 640
EPOCHS = 30
print('Training...')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_b = 0
    t0 = time.time()
    for imgs, batch_labels in train_loader:
        imgs = imgs.to(DEVICE)
        # 前向
        feats = []
        x = imgs
        y = []
        for i, m in enumerate(model.model):
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in model.save else None)
            if m.i == 15:  # P3
                feats.append(x)
            elif m.i == 18:  # P4
                feats.append(x)
            elif m.i == 21:  # P5
                feats.append(x)
        # head 输出
        head_out = model.model[-1](feats)  # 3 个尺度的特征图 (B, ch, H, W)
        
        # 解码 + loss
        loss = torch.tensor(0.0, device=DEVICE)
        n_pos = 0
        stride = [8, 16, 32]
        for si, (out, feat, st) in enumerate(zip(head_out, feats, stride)):
            B, C, H, W = out.shape
            # 每个特征图位置的输出: 4*16(reg) + 1(obj) + 24(kp) + 10(extra) = 99
            # 展平到 (B, 99, H*W) -> (B, H*W, 99)
            out = out.view(B, C, -1).permute(0, 2, 1)
            for bi in range(B):
                labels = batch_labels[bi]
                for gi, lb in enumerate(labels):
                    x1, y1, x2, y2 = lb['bbox']
                    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                    gx = int(cx / st); gy = int(cy / st)
                    if 0 <= gx < W and 0 <= gy < H:
                        idx = gy * W + gx
                        # box loss (L1)
                        pred_box = torch.sigmoid(out[bi, idx, :4]) * imgsz
                        gt_box = torch.tensor([x1, y1, x2, y2], device=DEVICE)
                        loss += F.l1_loss(pred_box, gt_box)
                        # kp loss (MSE on visible)
                        pred_kp = torch.sigmoid(out[bi, idx, 4*16+1:4*16+1+24]).view(NUM_KP, 3)
                        gt_kp = torch.tensor(lb['kp'], device=DEVICE)
                        vis = (gt_kp[:, 2] > 0.5).float()
                        if vis.sum() > 0:
                            loss += 0.1 * (vis.view(-1,1) * (pred_kp[:, :2] - gt_kp[:, :2])**2).sum() / vis.sum()
                        # extra loss
                        pe = out[bi, idx, 4*16+1+24:]
                        loss += F.cross_entropy(pe[:4].unsqueeze(0), torch.tensor([lb['status']], device=DEVICE))
                        loss += F.binary_cross_entropy_with_logits(pe[4:6], torch.tensor(
                            [lb['posture'], lb['posture']], device=DEVICE))
                        loss += F.binary_cross_entropy_with_logits(pe[6:10], torch.tensor(
                            lb['br'], device=DEVICE, dtype=torch.float))
                        n_pos += 1
        if n_pos > 0:
            loss = loss / n_pos
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
    scheduler.step()
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{EPOCHS} ({time.time()-t0:.0f}s): loss={total_loss/max(n_b,1):.4f}')

# ========== 5. 测试评估 ==========
print('\n===== TEST =====')
model.eval()
all_preds, all_gts = [], []
with torch.no_grad():
    for imgs, batch_labels in test_loader:
        imgs = imgs.to(DEVICE)
        x = imgs
        y = []
        feats = []
        for i, m in enumerate(model.model):
            if m.f != -1:
                x = [x if j == -1 else y[j] for j in m.f] if isinstance(m.f, list) else y[m.f]
            x = m(x)
            y.append(x if m.i in model.save else None)
            if m.i in [15, 18, 21]:
                feats.append(x)
        head_out = model.model[-1](feats)
        stride = [8, 16, 32]
        all_box_preds = {bi: [] for bi in range(imgs.shape[0])}
        for si, (out, st) in enumerate(zip(head_out, stride)):
            B, C, H, W = out.shape
            out = out.view(B, C, -1).permute(0, 2, 1)
            for bi in range(B):
                for gy in range(H):
                    for gx in range(W):
                        idx = gy * W + gx
                        obj = torch.sigmoid(out[bi, idx, 4*16])
                        if obj > 0.3:
                            box = torch.sigmoid(out[bi, idx, :4]) * imgsz
                            pe = out[bi, idx, 4*16+1+24:]
                            all_box_preds[bi].append({
                                'box': box.cpu().numpy(), 'obj': obj.item(),
                                'status': pe[:4].argmax().item(),
                                'posture': (torch.sigmoid(pe[4]) > 0.5).item(),
                                'br': (torch.sigmoid(pe[6:10]) > 0.5).cpu().numpy().astype(int),
                                'cx': (box[0]+box[2])/2, 'cy': (box[1]+box[3])/2,
                            })
        for bi in range(imgs.shape[0]):
            labels = batch_labels[bi]
            for gi, lb in enumerate(labels):
                gt_cx = (lb['bbox'][0] + lb['bbox'][2]) / 2
                gt_cy = (lb['bbox'][1] + lb['bbox'][3]) / 2
                best = None; best_d = float('inf')
                for p in all_box_preds[bi]:
                    d = (p['cx']-gt_cx)**2 + (p['cy']-gt_cy)**2
                    if d < best_d: best_d = d; best = p
                if best and best_d < 10000:
                    all_preds.append(best)
                    all_gts.append(lb)
                else:
                    all_preds.append({'status': 0, 'posture': 0, 'br': np.zeros(4, dtype=int)})
                    all_gts.append(lb)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
if all_preds:
    s_t = np.array([g['status'] for g in all_gts])
    s_p = np.array([p['status'] for p in all_preds])
    p_t = np.array([g['posture'] for g in all_gts])
    p_p = np.array([p['posture'] for p in all_preds])
    br_t = np.array([g['br'] for g in all_gts])
    br_p = np.array([p['br'] for p in all_preds])
    print(f'Status Acc={accuracy_score(s_t,s_p)*100:.1f}% MacroF1={precision_recall_fscore_support(s_t,s_p,average="macro",zero_division=0)[2]*100:.1f}%')
    print(f'Posture Acc={accuracy_score(p_t,p_p)*100:.1f}% F1(bad)={precision_recall_fscore_support(p_t,p_p,average="binary",pos_label=1,zero_division=0)[2]*100:.1f}%')
    for i,nm in enumerate(['head_tilt','head_down','lean_forward','side_lean']):
        f1=precision_recall_fscore_support(br_t[:,i],br_p[:,i],average='binary',zero_division=0)[2]
        print(f'  BR {nm:12s} F1={f1*100:.1f}%')
else:
    print('No predictions')

print('\n===== DONE =====')
print('Model: 端到端多任务检测 (YOLOv8n backbone + 自定义分类头)')
print('输入: 前视角整图(640x640) → 输出: 每学生 bbox+8kp+status+posture+br')
print('30 epoch, 无优化, 指标够不够直接反映数据量够不够')