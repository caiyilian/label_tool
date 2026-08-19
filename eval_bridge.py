import os, json, sys, glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
YOLO_DATA = '/public/cyl/label_tool/yolo_data'
CHECKPOINT = '/public/cyl/label_tool/training_output/best_multitask.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}
status_rev = {v: k for k, v in status_map.items()}
posture_rev = {v: k for k, v in posture_map.items()}

# ---------- 多任务分类器（与 train_multitask.py 一致） ----------
class MultiTaskModel(torch.nn.Module):
    def __init__(self, num_kp=8, num_status=4, num_posture=2, num_br=4):
        super().__init__()
        bb = models.efficientnet_b0(weights='DEFAULT')
        self.backbone = bb.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        feat = 1280
        self.kp_head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, num_kp*3))
        self.status_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, num_status))
        self.posture_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, num_posture))
        self.br_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, num_br))
    def forward(self, x):
        f = self.avgpool(self.backbone(x)).flatten(1)
        return self.kp_head(f), self.status_head(f), self.posture_head(f), self.br_head(f)

from torchvision import transforms as TF
from PIL import Image
tf = TF.Compose([
    TF.Resize((224, 224)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- 加载 GT 测试样本 ----------
def load_test_samples(view='front'):
    meta = json.load(open(os.path.join(YOLO_DATA, 'meta.json')))
    test_ids = [l.strip() for l in open(os.path.join(YOLO_DATA, 'test_front.txt'))]
    test_ids = {os.path.basename(p).replace('.jpg', '') for p in open(os.path.join(YOLO_DATA, 'test_front.txt'))}
    samples = []
    for i_str in test_ids:
        i = int(i_str)
        if meta[str(i)]['view'] != view:
            continue
        img_path = os.path.join(YOLO_DATA, 'images', f'{i:06d}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for ann in meta[str(i)]['annotations']:
            x1, y1, x2, y2 = [int(v) for v in ann['bbox']]
            samples.append({
                'image': img, 'bbox': ann['bbox'],
                'status': ann['status'], 'posture': ann['posture'], 'br': ann['br'],
                'kp': ann['kps'], 'cls': meta[str(i)]['class'],
            })
    return samples

def crop_classify(samples, use_pred_bbox=False, detector=None, det_conf=0.3):
    model = MultiTaskModel().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 按图分组
    from collections import defaultdict
    imgs = defaultdict(list)
    for s in samples:
        imgs[id(s['image'])].append(s)

    s_pred, s_true, p_pred, p_true, br_pred, br_true = [], [], [], [], [], []
    with torch.no_grad():
        for img_id, group in imgs.items():
            img = group[0]['image']  # 同一图
            if use_pred_bbox:
                # 检测
                det = detector(img)
                boxes = det[0].boxes.xyxy.cpu().numpy() if len(det[0].boxes) else np.zeros((0, 4))
            for s in group:
                if use_pred_bbox:
                    # 用预测框里最接近 GT 的（简单起见用 IoU 最高的预测框）
                    if len(boxes):
                        gt = np.array(s['bbox'])
                        ious = []
                        for b in boxes:
                            x1, y1, x2, y2 = b
                            gx1, gy1, gx2, gy2 = gt
                            inter = max(0, min(x2, gx2) - max(x1, gx1)) * max(0, min(y2, gy2) - max(y1, gy1))
                            union = (x2-x1)*(y2-y1) + (gx2-gx1)*(gy2-gy1) - inter
                            ious.append(inter / max(union, 1))
                        box = boxes[int(np.argmax(ious))]
                    else:
                        box = np.array(s['bbox'])
                    x1, y1, x2, y2 = [int(v) for v in box]
                else:
                    x1, y1, x2, y2 = [int(v) for v in s['bbox']]
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                t = tf(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
                kp_o, s_o, p_o, br_o = model(t)
                s_pred.append(s_o.argmax(1).item())
                s_true.append(s['status'])
                p_pred.append(p_o.argmax(1).item())
                p_true.append(s['posture'])
                br_pred.append((torch.sigmoid(br_o)[0] > 0.5).cpu().numpy())
                br_true.append(np.array(s['br']))
    return (np.array(s_pred), np.array(s_true), np.array(p_pred), np.array(p_true),
            np.array(br_pred), np.array(br_true))

def report(s_pred, s_true, p_pred, p_true, br_pred, br_true, label):
    print(f'\n===== {label} =====')
    sa = accuracy_score(s_true, s_pred)
    sm = precision_recall_fscore_support(s_true, s_pred, average='macro', zero_division=0)[2]
    print(f'Status Acc={sa*100:.1f}%  MacroF1={sm*100:.1f}%')
    pa = accuracy_score(p_true, p_pred)
    p_f1 = precision_recall_fscore_support(p_true, p_pred, average='binary', pos_label=1, zero_division=0)[2]
    print(f'Posture Acc={pa*100:.1f}%  F1(bad)={p_f1*100:.1f}%')
    for i, nm in enumerate(['head_tilt', 'head_down', 'lean_forward', 'side_lean']):
        gt, pr = br_true[:, i], br_pred[:, i]
        f1 = precision_recall_fscore_support(gt, pr, average='binary', zero_division=0)[2]
        print(f'  BR {nm:12s} F1={f1*100:.1f}%')
    return sa, sm, pa, p_f1

if __name__ == '__main__':
    print('Loading test samples (front view)...')
    samples = load_test_samples('front')
    print(f'{len(samples)} samples')

    print('\n>>> GT bbox 裁剪分类（理想上界）')
    s_pred, s_true, p_pred, p_true, br_pred, br_true = crop_classify(samples, use_pred_bbox=False)
    report(s_pred, s_true, p_pred, p_true, br_pred, br_true, 'GT bbox crop')

    print('\n\nLoading detector (pose_front best) ...')
    det_model = YOLO('/public/cyl/label_tool/runs_pose/pose_front/weights/best.pt')

    print('\n>>> 预测 bbox 裁剪分类（端到端真实管线）')
    s_pred, s_true, p_pred, p_true, br_pred, br_true = crop_classify(samples, use_pred_bbox=True, detector=det_model)
    report(s_pred, s_true, p_pred, p_true, br_pred, br_true, 'Pred bbox crop (end-to-end)')