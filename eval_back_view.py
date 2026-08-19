import os, json, glob
import cv2
import numpy as np
import torch
from torchvision import models, transforms as TF
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
YOLO_DATA = '/public/cyl/label_tool/yolo_data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}

tf = TF.Compose([TF.Resize((224, 224)), TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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

class TwoStreamModel(torch.nn.Module):
    """train_multiview.py 中的双流模型"""
    def __init__(self, num_kp=8, num_status=4, num_posture=2, num_br=4):
        super().__init__()
        bb = models.efficientnet_b0(weights='DEFAULT')
        self.stream = bb.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        feat = 1280
        self.kp_front_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 24))
        self.kp_back_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 24))
        self.status_head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(feat, 128),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(128, num_status))
        self.posture_head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(feat, 64),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, num_posture))
        self.br_head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(feat, 64),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, num_br))
    def forward(self, front_x, back_x):
        f_f = self.avgpool(self.stream(front_x)).flatten(1)
        f_b = self.avgpool(self.stream(back_x)).flatten(1)
        fused = f_f + f_b
        return (self.kp_front_head(f_f), self.kp_back_head(f_b),
                self.status_head(fused), self.posture_head(fused), self.br_head(fused))

def load_back_samples():
    meta = json.load(open(os.path.join(YOLO_DATA, 'meta.json')))
    test_ids = set(int(l.strip().split('/')[-1].replace('.jpg', ''))
                   for l in open(os.path.join(YOLO_DATA, 'test_front.txt')))
    samples = []
    for i in test_ids:
        m = meta[str(i)]
        if m['view'] != 'front':
            continue
        with open(m['json']) as f:
            raw = json.load(f)
        forward_path = os.path.join(os.path.dirname(m['json']).replace('annotations', 'forward'),
                                    os.path.basename(m['json']).replace('.json', '.jpg'))
        back_path = forward_path.replace('/forward/', '/backward/').replace('.jpg', '')[:-5] + '.jpg'
        back_path = forward_path.replace('forward', 'backward').replace('front_', 'back_', 1)
        fimg = cv2.imread(forward_path); fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
        bimg = cv2.imread(back_path); bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
        for ann in raw['annotations']:
            bbox_b = ann.get('back_bbox')
            if bbox_b is None:
                continue  # 只保留有真实 back_bbox 的样本
            br_vec = np.zeros(4, dtype=int)
            for r in ann.get('bad_reason', []):
                if r in bad_reason_map:
                    br_vec[bad_reason_map[r]] = 1
            samples.append({
                'front': fimg, 'bb_f': ann['bbox'], 'back': bimg, 'bb_b': bbox_b,
                'status': status_map[ann['status']], 'posture': posture_map[ann['posture']],
                'br': br_vec,
                'jpg': os.path.basename(m['json']),
            })
    return samples

def crop(img, bb):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bb]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def run():
    samples = load_back_samples()
    print(f'有真实 back_bbox 的测试样本: {len(samples)}')

    single = MultiTaskModel().to(DEVICE)
    single.load_state_dict(torch.load('/public/cyl/label_tool/training_output/best_multitask.pth', map_location=DEVICE))
    single.eval()

    mv = TwoStreamModel().to(DEVICE)
    mv.load_state_dict(torch.load('/public/cyl/label_tool/training_output/best_multiview.pth', map_location=DEVICE))
    mv.eval()

    accs = {'single': {'status': [], 'posture': [], 'br': []},
            'multiview': {'status': [], 'posture': [], 'br': []}}
    trues = {'status': [], 'posture': [], 'br': []}

    with torch.no_grad():
        for s in samples:
            fc = crop(s['front'], s['bb_f'])
            bc = crop(s['back'], s['bb_b'])
            if fc is None or bc is None:
                continue
            ft = tf(Image.fromarray(fc)).unsqueeze(0).to(DEVICE)
            bt = tf(Image.fromarray(bc)).unsqueeze(0).to(DEVICE)
            _, so, po, bro = single(ft)
            _, _, s2, p2, br2 = mv(ft, bt)
            accs['single']['status'].append(int(so.argmax(1)))
            accs['single']['posture'].append(int(po.argmax(1)))
            accs['single']['br'].append((torch.sigmoid(bro)[0] > 0.5).cpu().numpy())
            accs['multiview']['status'].append(int(s2.argmax(1)))
            accs['multiview']['posture'].append(int(p2.argmax(1)))
            accs['multiview']['br'].append((torch.sigmoid(br2)[0] > 0.5).cpu().numpy())
            trues['status'].append(s['status'])
            trues['posture'].append(s['posture'])
            trues['br'].append(s['br'].astype(int))

    for name, d in accs.items():
        print(f'\n===== {name} (仅看有后视角标注的样本, n={len(d["status"])}) =====')
        sa = accuracy_score(trues['status'], d['status'])
        sm = precision_recall_fscore_support(trues['status'], d['status'], average='macro', zero_division=0)[2]
        pa = accuracy_score(trues['posture'], d['posture'])
        pf1 = precision_recall_fscore_support(trues['posture'], d['posture'], average='binary', pos_label=1, zero_division=0)[2]
        print(f'Status Acc={sa*100:.1f}% MacroF1={sm*100:.1f}%')
        print(f'Posture Acc={pa*100:.1f}% F1(bad)={pf1*100:.1f}%')
        brp = np.array(d['br']); brt = np.array(trues['br'])
        for i, nm in enumerate(['head_tilt', 'head_down', 'lean_forward', 'side_lean']):
            f1 = precision_recall_fscore_support(brt[:, i], brp[:, i], average='binary', zero_division=0)[2]
            print(f'  BR {nm:12s} F1={f1*100:.1f}%')

if __name__ == '__main__':
    run()