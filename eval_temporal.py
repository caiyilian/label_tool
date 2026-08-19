import os, json, glob, re
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms as TF
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
YOLO_DATA = '/public/cyl/label_tool/yolo_data'
CHECKPOINT = '/public/cyl/label_tool/training_output/best_multitask.pth'
POSE_CKPT = '/public/cyl/label_tool/runs_pose/pose_front/weights/best.pt'

status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}

class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.efficientnet_b0(weights='DEFAULT')
        self.backbone = bb.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        feat = 1280
        self.kp_head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 24))
        self.status_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 4))
        self.posture_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 2))
        self.br_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 4))
    def forward(self, x):
        f = self.avgpool(self.backbone(x)).flatten(1)
        return self.kp_head(f), self.status_head(f), self.posture_head(f), self.br_head(f)

tf = TF.Compose([TF.Resize((224, 224)), TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def load_classifier():
    model = MultiTaskModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model

def classify_crops(model, crops):
    """crops: list of numpy RGB arrays"""
    outs = []
    with torch.no_grad():
        for c in crops:
            t = tf(Image.fromarray(c)).unsqueeze(0).to(DEVICE)
            _, s, p, br = model(t)
            outs.append((s.softmax(1).cpu().numpy()[0],
                         p.softmax(1).cpu().numpy()[0],
                         torch.sigmoid(br).cpu().numpy()[0]))
    return outs

def run_temporal_ablation():
    print('Loading meta...')
    meta = json.load(open(os.path.join(YOLO_DATA, 'meta.json')))
    test_ids = [int(l.strip().split('/')[-1].replace('.jpg', ''))
                for l in open(os.path.join(YOLO_DATA, 'test_front.txt'))]

    # 构建测试班级的帧序列
    test_meta = {i: meta[str(i)] for i in test_ids if meta[str(i)]['view'] == 'front'}
    by_class = {}
    for i, m in test_meta.items():
        fr = int(re.search(r'(\d+)\.jpg', m['frame']).group(1))
        by_class.setdefault(m['class'], []).append((fr, i))
    for c in by_class:
        by_class[c].sort()

    print(f'Test classes: {list(by_class.keys())}')

    det = YOLO(POSE_CKPT)
    clf = load_classifier()

    # GT 真值：frame -> student_bbox -> label
    gt_by_class = {}
    for c, items in by_class.items():
        frames = {}
        for fr, i in items:
            labels = {}  # bbox(cx) -> label tuple. 用中心x近似关联
            for ann in meta[str(i)]['annotations']:
                x1, y1, x2, y2 = ann['bbox']
                cx = round((x1 + x2) / 2 / 1920, 3)
                labels[cx] = (ann['status'], ann['posture'], ann['br'])
            frames[fr] = labels
        gt_by_class[c] = frames

    all_single_pred = {'status': [], 'posture': [], 'br': []}
    all_single_true = {'status': [], 'posture': [], 'br': []}
    all_temporal_pred = {'status': [], 'posture': [], 'br': []}
    all_temporal_true = {'status': [], 'posture': [], 'br': []}
    temporal_pairs = 0

    for c, items in by_class.items():
        frames = gt_by_class[c]
        # 检测所有帧的所有学生
        det_results = {}  # fr -> list of (box, crop)
        for fr_v, i in items:
            img_path = os.path.join(YOLO_DATA, 'images', f'{i:06d}.jpg')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = det(img)
            arr = []
            for b in r[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                crop = img[y1:y2, x1:x2]
                if crop.size:
                    cx = round((b[0] + b[2]) / 2 / 1920, 3)
                    arr.append((cx, crop))
            det_results[fr_v] = arr

        # 帧序列（同一班级的帧）
        frs = [fr for fr, _ in items]
        for fidx, fr in enumerate(frs):
            if fr not in det_results:
                continue
            # 单帧：每个检测框直接分类
            bboxes_single = det_results[fr]
            crops_s = [c for _, c in bboxes_single]
            single_outs = classify_crops(clf, crops_s) if crops_s else []

            # 时序：用相邻帧（在测试班级中前后的帧序号, diff<=3）做投票
            # 先收集 temporal 邻居
            temporal_outs = []
            for cxb, crop in bboxes_single:
                vote_s = [np.zeros(4)]; vote_p = [np.zeros(2)]; vote_br = [np.zeros(4)]
                # 当前帧自己的预测在单帧里已经算了, 这里重新算本帧
                cur = classify_crops(clf, [crop])[0]
                vote_s = [cur[0]]; vote_p = [cur[1]]; vote_br = [cur[2]]
                # 找邻居帧
                for delta in [-1, 1]:
                    nf = fr + delta
                    if nf in det_results:
                        arr_n = det_results[nf]
                        # IOU 关联：找与该框最近的
                        if arr_n:
                            cand_crops = [cr for _, cr in arr_n]
                            cand_outs = classify_crops(clf, cand_crops)
                            vote_s.append(cand_outs[0][0])
                            vote_p.append(cand_outs[0][1])
                            vote_br.append(cand_outs[0][2])
                temporal_outs.append(
                    (np.mean(vote_s, axis=0), np.mean(vote_p, axis=0), np.mean(vote_br, axis=0))
                )
            temporal_pairs += len(bboxes_single)

            # 与 GT 关联（用中心x匹配）
            if fr in frames:
                gt_labels = frames[fr]
                for (cxb, crop), single_out, temp_out in zip(bboxes_single, single_outs, temporal_outs):
                    # 找最近 GT
                    best_cx = min(gt_labels.keys(), key=lambda g: abs(g - cxb))
                    st, po, brl = gt_labels[best_cx]
                    all_single_true['status'].append(st)
                    all_single_true['posture'].append(po)
                    all_single_true['br'].append(np.array(brl))
                    all_single_pred['status'].append(int(np.argmax(single_out[0])))
                    all_single_pred['posture'].append(int(np.argmax(single_out[1])))
                    all_single_pred['br'].append((single_out[2] > 0.5).astype(int))
                    all_temporal_true['status'].append(st)
                    all_temporal_true['posture'].append(po)
                    all_temporal_true['br'].append(np.array(brl))
                    all_temporal_pred['status'].append(int(np.argmax(temp_out[0])))
                    all_temporal_pred['posture'].append(int(np.argmax(temp_out[1])))
                    all_temporal_pred['br'].append((temp_out[2] > 0.5).astype(int))

    print(f'\nTotal matched detections: {len(all_single_pred["status"])}')

    def fmt(pred_dict, true_dict, name):
        print(f'\n===== {name} =====')
        sa = accuracy_score(true_dict['status'], pred_dict['status'])
        sm = precision_recall_fscore_support(true_dict['status'], pred_dict['status'], average='macro', zero_division=0)[2]
        print(f'Status Acc={sa*100:.1f}%  MacroF1={sm*100:.1f}%')
        pa = accuracy_score(true_dict['posture'], pred_dict['posture'])
        pf1 = precision_recall_fscore_support(true_dict['posture'], pred_dict['posture'], average='binary', pos_label=1, zero_division=0)[2]
        print(f'Posture Acc={pa*100:.1f}%  F1(bad)={pf1*100:.1f}%')
        br_pred = np.array(pred_dict['br']); br_true = np.array(true_dict['br'])
        for i, nm in enumerate(['head_tilt', 'head_down', 'lean_forward', 'side_lean']):
            f1 = precision_recall_fscore_support(br_true[:, i], br_pred[:, i], average='binary', zero_division=0)[2]
            print(f'  BR {nm:12s} F1={f1*100:.1f}%')
        return sa, pa

    fmt(all_single_pred, all_single_true, '单帧（无时序）')
    fmt(all_temporal_pred, all_temporal_true, '时序投票（±1帧邻域平均）')

if __name__ == '__main__':
    run_temporal_ablation()