import os, json, glob, sys, time, pickle, random, gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms as TF
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUTPUT_DIR = '/public/cyl/label_tool/training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

KP_NAMES = ['head','neck','right_shoulder','right_elbow','right_hand','left_shoulder','left_elbow','left_hand']
NUM_KP = 8
status_map = {'sitting_listening':0,'sitting_reading':1,'standing':2,'left_seat':3}
posture_map = {'normal':0,'bad':1}
bad_reason_map = {'head_tilt':0,'head_down':1,'lean_forward':2,'side_lean':3}
status_rev = {v:k for k,v in status_map.items()}
posture_rev = {v:k for k,v in posture_map.items()}

# ========== 1. 数据加载 ==========
def load_data():
    anns = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    samples = []
    for f in anns:
        cls = f.replace(BASE,'').lstrip('/').split('/')[0]
        with open(f) as fh: data = json.load(fh)
        jpg = os.path.basename(f).replace('.json','.jpg')
        fwd = os.path.join(os.path.dirname(f).replace('annotations','forward'), jpg)
        img = cv2.imread(fwd)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        for ann in data['annotations']:
            bx = ann['bbox']; x1,y1,x2,y2 = map(int,bx)
            x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
            if x2<=x1 or y2<=y1: continue
            crop = cv2.resize(img[y1:y2,x1:x2], (224,224))
            kpf = ann.get('pose_keypoints') or []
            kp = np.zeros((NUM_KP,3),dtype=np.float32)
            for k in kpf:
                if k is None or k['name'] not in KP_NAMES: continue
                idx = KP_NAMES.index(k['name'])
                vis = 1.0 if k.get('visible') else 0.0
                kp[idx] = [np.clip((k['x']-x1)/max(x2-x1,1),0,1), np.clip((k['y']-y1)/max(y2-y1,1),0,1), vis] if vis else [0,0,0]
            br = [0]*4
            for r in ann.get('bad_reason',[]):
                if r in bad_reason_map: br[bad_reason_map[r]] = 1
            samples.append({'class':cls,'image':crop,'status':ann['status'],'posture':ann['posture'],'br':br,'keypoints':kp})
    return samples

all_samples = load_data()
print(f'Total: {len(all_samples)}')

classes = sorted(set(s['class'] for s in all_samples)); random.shuffle(classes)
n=len(classes)
train_cls=set(classes[:int(n*0.7)]); val_cls=set(classes[int(n*0.7):int(n*0.85)]); test_cls=set(classes[int(n*0.85):])
train_s=[s for s in all_samples if s['class'] in train_cls]
val_s=[s for s in all_samples if s['class'] in val_cls]
test_s=[s for s in all_samples if s['class'] in test_cls]
print(f'Train:{len(train_s)} Val:{len(val_s)} Test:{len(test_s)}')

# mean/std
mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])

def prepare(samples, aug=False):
    imgs,kps,sts,pos,brs = [],[],[],[],[]
    for s in samples:
        img = s['image'].astype(np.float32)/255.0
        if aug and random.random()<0.3: img = np.fliplr(img).copy()
        img = (img-mean)/std
        imgs.append(torch.from_numpy(img.transpose(2,0,1)).float())
        kps.append(torch.from_numpy(s['keypoints'].flatten()).float())
        sts.append(status_map[s['status']])
        pos.append(posture_map[s['posture']])
        brs.append(s['br'])
    return (torch.stack(imgs), torch.stack(kps), torch.tensor(sts), torch.tensor(pos), torch.tensor(brs,dtype=torch.float32))

train_data = prepare(train_s, aug=True)
val_data = prepare(val_s, aug=False)
test_data = prepare(test_s, aug=False)

# ========== 2. 6 个架构 ==========
architectures = [
    ('efficientnet_b0', lambda: models.efficientnet_b0(weights='DEFAULT'), 1280),
    ('mobilenet_v3_small', lambda: models.mobilenet_v3_small(weights='DEFAULT'), 576),
    ('mobilenet_v3_large', lambda: models.mobilenet_v3_large(weights='DEFAULT'), 960),
    ('resnet18', lambda: models.resnet18(weights='DEFAULT'), 512),
    ('resnet50', lambda: models.resnet50(weights='DEFAULT'), 2048),
    ('convnext_tiny', lambda: models.convnext_tiny(weights='DEFAULT'), 768),
]

def build_model(bb_fn, feat_dim):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            bb = bb_fn()
            # 提取 backbone 特征
            if hasattr(bb, 'features'):
                self.backbone = bb.features
            elif hasattr(bb, 'conv1'):
                self.backbone = nn.Sequential(*list(bb.children())[:-2])
            else:
                self.backbone = nn.Sequential(*list(bb.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.kp = nn.Sequential(nn.Dropout(0.3),nn.Linear(feat_dim,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,NUM_KP*3))
            self.status = nn.Sequential(nn.Dropout(0.3),nn.Linear(feat_dim,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,4))
            self.posture = nn.Sequential(nn.Dropout(0.3),nn.Linear(feat_dim,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,2))
            self.br = nn.Sequential(nn.Dropout(0.3),nn.Linear(feat_dim,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,4))
        def forward(self,x):
            f = self.avgpool(self.backbone(x)).flatten(1)
            return self.kp(f), self.status(f), self.posture(f), self.br(f)
    return M()

def kp_loss(pred,target):
    B,D=pred.shape
    t=target.view(B,NUM_KP,3); p=pred.view(B,NUM_KP,3)
    pxy=torch.sigmoid(p[:,:,:2]); pvis=torch.sigmoid(p[:,:,2:3]); txy=t[:,:,:2]; tvis=t[:,:,2:3]
    m=(tvis>0.5).float()
    xy = (m*(pxy-txy)**2).sum()/(m.sum()+1e-6)
    vis = nn.functional.binary_cross_entropy(pvis,tvis)
    return xy+0.5*vis

criterion_kp = kp_loss
status_w = torch.tensor([1.0,1.0,3.0,3.0], device=DEVICE)
criterion_st = nn.CrossEntropyLoss(weight=status_w)
criterion_po = nn.CrossEntropyLoss()
criterion_br = nn.BCEWithLogitsLoss()

def batch_iter(data, bs=64, shuffle=True):
    imgs,kps,sts,pos,brs = data
    idxs = list(range(len(imgs)))
    if shuffle: random.shuffle(idxs)
    for i in range(0,len(imgs),bs):
        ii = idxs[i:i+bs]
        yield (imgs[ii].to(DEVICE), kps[ii].to(DEVICE), sts[ii].to(DEVICE), pos[ii].to(DEVICE), brs[ii].to(DEVICE))

def train_classifier(arch_name, bb_fn, feat_dim, epochs=30):
    model = build_model(bb_fn, feat_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        for imgs,kps,sts,pos,brs in batch_iter(train_data):
            opt.zero_grad()
            kp_o,st_o,po_o,br_o = model(imgs)
            loss = criterion_kp(kp_o,kps) + criterion_st(st_o,sts) + criterion_po(po_o,pos) + criterion_br(br_o,brs)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        # val
        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs,kps,sts,pos,brs in batch_iter(val_data, shuffle=False):
                kp_o,st_o,po_o,br_o = model(imgs)
                vloss += (criterion_kp(kp_o,kps)+criterion_st(st_o,sts)+criterion_po(po_o,pos)+criterion_br(br_o,brs)).item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        sch.step()
        if (ep+1)%10==0:
            print(f'  {arch_name} ep{ep+1}: val_loss={vloss:.4f}')
    model.load_state_dict(best_state)
    ckpt = os.path.join(OUTPUT_DIR, f'classifier_{arch_name}.pth')
    torch.save(model.state_dict(), ckpt)
    return model, ckpt

def evaluate_model(model, data):
    model.eval()
    s_pred,s_true,p_pred,p_true,br_pred,br_true = [],[],[],[],[],[]
    with torch.no_grad():
        for imgs,kps,sts,pos,brs in batch_iter(data, shuffle=False):
            _,st_o,po_o,br_o = model(imgs)
            s_pred.extend(st_o.argmax(1).cpu().numpy())
            s_true.extend(sts.cpu().numpy())
            p_pred.extend(po_o.argmax(1).cpu().numpy())
            p_true.extend(pos.cpu().numpy())
            br_pred.extend((torch.sigmoid(br_o)>0.5).cpu().numpy())
            br_true.extend(brs.cpu().numpy())
    return (np.array(s_pred),np.array(s_true),np.array(p_pred),np.array(p_true),np.array(br_pred),np.array(br_true))

# ========== 3. 端到端 Bridge 评估 ==========
def evaluate_bridge(model, ckpt_path, arch_name):
    # 加载检测器
    det = YOLO('/public/cyl/label_tool/runs_pose/pose_front/weights/best.pt')
    # 加载分类器
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    # 获取测试集图片
    meta = json.load(open(os.path.join(OUTPUT_DIR.replace('training_output','yolo_data'),'meta.json')))
    # 直接用 test_s 中的样本
    from collections import defaultdict
    img_groups = defaultdict(list)
    # 重建测试集图片路径
    anns = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    for f in anns:
        cls = f.replace(BASE,'').lstrip('/').split('/')[0]
        if cls not in test_cls: continue
        with open(f) as fh: data = json.load(fh)
        jpg = os.path.basename(f).replace('.json','.jpg')
        fwd = os.path.join(os.path.dirname(f).replace('annotations','forward'), jpg)
        img = cv2.imread(fwd)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        for ann in data['annotations']:
            bx = ann['bbox']
            br = [0]*4
            for r in ann.get('bad_reason',[]):
                if r in bad_reason_map: br[bad_reason_map[r]] = 1
            img_groups[id(img)].append({'img':img,'bbox':bx,'status':status_map[ann['status']],
                'posture':posture_map[ann['posture']],'br':br})

    s_pred,s_true,p_pred,p_true,br_pred,br_true = [],[],[],[],[],[]
    tf = TF.Compose([TF.Resize((224,224)), TF.ToTensor(), TF.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    with torch.no_grad():
        for img_id, group in img_groups.items():
            img = group[0]['img']
            det_r = det(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if len(det_r[0].boxes)==0:
                for s in group:
                    s_pred.append(0); s_true.append(s['status']); p_pred.append(0); p_true.append(s['posture'])
                    br_pred.append(np.zeros(4,dtype=int)); br_true.append(np.array([0,0,0,0]))
                continue
            boxes = det_r[0].boxes.xyxy.cpu().numpy()
            for s in group:
                gt = np.array(s['bbox'])
                best_iou = -1; best_box = boxes[0]
                for b in boxes:
                    inter = max(0,min(b[2],gt[2])-max(b[0],gt[0]))*max(0,min(b[3],gt[3])-max(b[1],gt[1]))
                    union = (b[2]-b[0])*(b[3]-b[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter
                    iou = inter/max(union,1)
                    if iou>best_iou: best_iou=iou; best_box=b
                x1,y1,x2,y2 = [int(v) for v in best_box]
                x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(img.shape[1],x2),min(img.shape[0],y2)
                if x2<=x1 or y2<=y1:
                    s_pred.append(0); p_pred.append(0); br_pred.append(np.zeros(4,dtype=int))
                else:
                    crop = img[y1:y2,x1:x2]
                    t = tf(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
                    _,st_o,po_o,br_o = model(t)
                    s_pred.append(st_o.argmax(1).item())
                    p_pred.append(po_o.argmax(1).item())
                    br_pred.append((torch.sigmoid(br_o)[0]>0.5).cpu().numpy().astype(int))
                s_true.append(s['status']); p_true.append(s['posture']); br_true.append(s['br'])
    return (np.array(s_pred),np.array(s_true),np.array(p_pred),np.array(p_true),np.array(br_pred),np.array(br_true))

# ========== 4. 运行所有 ==========
results = []
for arch_name, bb_fn, feat_dim in architectures:
    print(f'\n{"="*60}\n  [{arch_name}] 训练分类器...\n{"="*60}')
    model, ckpt_path = train_classifier(arch_name, bb_fn, feat_dim, epochs=30)
    
    # GT bbox 评估
    s_pred,s_true,p_pred,p_true,br_pred,br_true = evaluate_model(model, test_data)
    gt_metrics = {
        'status_acc': accuracy_score(s_true,s_pred),
        'status_f1_macro': precision_recall_fscore_support(s_true,s_pred,average='macro',zero_division=0)[2],
        'posture_acc': accuracy_score(p_true,p_pred),
        'posture_f1_bad': precision_recall_fscore_support(p_true,p_pred,average='binary',pos_label=1,zero_division=0)[2],
    }
    for i,nm in enumerate(['head_tilt','head_down','lean_forward','side_lean']):
        gt_metrics[f'br_{nm}_f1'] = precision_recall_fscore_support(br_true[:,i],br_pred[:,i],average='binary',zero_division=0)[2]
    print(f'  GT: Posture={gt_metrics["posture_acc"]*100:.1f}% Status={gt_metrics["status_acc"]*100:.1f}%')
    
    # Pred bbox 评估（端到端真实管线）
    print(f'  Bridge 评估...')
    try:
        s_pred,s_true,p_pred,p_true,br_pred,br_true = evaluate_bridge(model, ckpt_path, arch_name)
        bridge_metrics = {
            'bridge_status_acc': accuracy_score(s_true,s_pred),
            'bridge_status_f1_macro': precision_recall_fscore_support(s_true,s_pred,average='macro',zero_division=0)[2],
            'bridge_posture_acc': accuracy_score(p_true,p_pred),
            'bridge_posture_f1_bad': precision_recall_fscore_support(p_true,p_pred,average='binary',pos_label=1,zero_division=0)[2],
        }
        for i,nm in enumerate(['head_tilt','head_down','lean_forward','side_lean']):
            bridge_metrics[f'bridge_br_{nm}_f1'] = precision_recall_fscore_support(br_true[:,i],br_pred[:,i],average='binary',zero_division=0)[2]
        print(f'  Bridge: Posture={bridge_metrics["bridge_posture_acc"]*100:.1f}% Status={bridge_metrics["bridge_status_acc"]*100:.1f}%')
    except Exception as e:
        print(f'  Bridge FAILED: {e}')
        bridge_metrics = {}
    
    results.append({'arch':arch_name, **gt_metrics, **bridge_metrics})
    json.dump(results, open(os.path.join(OUTPUT_DIR,'end2end_results.json'),'w'), indent=2)
    del model; gc.collect(); torch.cuda.empty_cache()

print('\n\n===== 所有训练完成 =====')
print(json.dumps(results, indent=2))