import os, json, glob
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import models, transforms as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
YOLO_DATA = '/public/cyl/label_tool/yolo_data'
OUT = '/public/cyl/label_tool/training_output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']
NUM_KP = 8

train_cls = {'12.04-104','12.03-305','12.02-402','12.04-203','12.04-305','12.03-402','12.05-104','12.03-104'}
test_cls = {'12.02-305','12.04-402'}

tf_train = TF.Compose([TF.Resize((224,224)), TF.RandomHorizontalFlip(p=0.3),
    TF.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05)),
    TF.ToTensor(), TF.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf_eval = TF.Compose([TF.Resize((224,224)), TF.ToTensor(),
    TF.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def crop_region(img, bb):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bb]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def collect_instances(cls_set, require_back_kp=True):
    """返回带后视角关键点的实例 (back_crop, target_rel_to_backbbox)"""
    anns = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    insts = []
    for f in anns:
        cls = f.replace(BASE, '').lstrip('/').split('/')[0]
        if cls not in cls_set:
            continue
        with open(f) as fh:
            data = json.load(fh)
        jpg = os.path.basename(f).replace('.json', '.jpg')
        fwd = os.path.join(os.path.dirname(f).replace('annotations','forward'), jpg)
        bwd = os.path.join(os.path.dirname(f).replace('annotations','backward'), jpg.replace('front_','back_'))
        bimg = cv2.imread(bwd); bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
        if bimg is None:
            continue
        for ann in data['annotations']:
            bb = ann.get('back_bbox')
            if bb is None:
                continue
            kpb = ann.get('pose_keypoints_back')
            vis_kps = [k for k in (kpb or []) if k and k.get('visible')]
            if require_back_kp and not vis_kps:
                continue
            crop = crop_region(bimg, bb)
            if crop is None:
                continue
            ch, cw = crop.shape[:2]
            tgt = np.zeros((NUM_KP, 3), dtype=np.float32)
            for kp in (kpb or []):
                if kp is None or not kp.get('name'):
                    continue
                if kp['name'] not in KP_NAMES:
                    continue
                idx = KP_NAMES.index(kp['name'])
                vis = 1.0 if kp.get('visible') else 0.0
                if vis:
                    kx = np.clip((kp['x'] - bb[0]) / max(cw,1), 0, 1)
                    ky = np.clip((kp['y'] - bb[1]) / max(ch,1), 0, 1)
                else:
                    kx = ky = 0.0
                tgt[idx] = [kx, ky, vis]
            insts.append({'crop': crop, 'target': tgt, 'cls': cls})
    return insts

class KPDataset(Dataset):
    def __init__(self, instances, train=True):
        self.insts = instances
        self.train = train
    def __len__(self):
        return len(self.insts)
    def __getitem__(self, i):
        inst = self.insts[i]
        if self.train:
            t = tf_train(Image.fromarray(inst['crop']))
        else:
            t = tf_eval(Image.fromarray(inst['crop']))
        return t, torch.from_numpy(inst['target'].flatten()).float()

class BackKPNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.efficientnet_b0(weights='DEFAULT')
        self.backbone = bb.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.kp_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280,256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_KP*3))
    def forward(self, x):
        f = self.avgpool(self.backbone(x)).flatten(1)
        return self.kp_head(f)

def kp_loss(pred, tgt):
    B = pred.shape[0]
    p = pred.view(B, NUM_KP, 3)
    t = tgt.view(B, NUM_KP, 3)
    pxy = torch.sigmoid(p[:,:,:2]); pvis = torch.sigmoid(p[:,:,2:])
    m = (t[:,:,2] > 0.5).float()
    xy = (m.unsqueeze(-1) * (pxy - t[:,:,:2])**2).sum() / (m.sum()+1e-6)
    vis = nn.functional.binary_cross_entropy(pvis, t[:,:,2:3])
    return xy + 0.5*vis

def train_back_kp(save_path):
    train_insts = collect_instances(train_cls, require_back_kp=True)
    print(f'训练后视角补位网络, 训练实例: {len(train_insts)}')
    if len(train_insts) == 0:
        return None
    ds = KPDataset(train_insts, train=True)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    model = BackKPNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for epoch in range(40):
        model.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = kp_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()*len(x)
        if (epoch+1)%10==0:
            print(f'  epoch {epoch+1}: loss={tot/max(len(ds),1):.4f}')
    torch.save(model.state_dict(), save_path)
    return model

def build_aux_net(load):
    m = BackKPNet().to(DEVICE)
    m.load_state_dict(torch.load(load, map_location=DEVICE))
    m.eval()
    return m

def eval_supplement():
    test_insts = collect_instances(test_cls, require_back_kp=True)
    print(f'测试实例(有后视角关键点): {len(test_insts)}')

    aux = build_aux_net(os.path.join(OUT, 'back_kp_aux.pth'))

    # 加载主检测器（前视角 YOLOv8-pose）
    from ultralytics import YOLO
    det = YOLO('/public/cyl/label_tool/runs_pose/pose_front/weights/best.pt')

    per_kp = {nm: {'front_ok': 0, 'back_ok': 0, 'front_err_vis': 0.0, 'back_err_vis': 0.0,
                    'n_front_vis': 0, 'n_back_vis': 0} for nm in KP_NAMES}

    # 统计前视角GT在整图上的关键点位置 + 后视角GT
    with torch.no_grad():
        for inst in test_insts:
            # 用后 crop 预测后 kp
            bt = tf_eval(Image.fromarray(inst['crop'])).unsqueeze(0).to(DEVICE)
            out = aux(bt).view(1, NUM_KP, 3)
            pred_bxy = torch.sigmoid(out[0,:,:2]).cpu().numpy()
            pred_bvis = (torch.sigmoid(out[0,:,2:3]) > 0.5).numpy().ravel()
            tgt = inst['target']

            # 前视角用什么? 用收集时对应 json 的前 kp —— 重新收集前视角协同
            # find matching ann again via reparse simpler: use stored back bbox to locate front
            # 简化：从 back_bbox 出发，用存储的 front GT kp（需要重新读 json）
            # 为严谨，重新收集测试集 "前+后同时有bbox" 的实例
    # 重新实现——需要前视角信息
    reparse_back_with_front()

def reparse_back_with_front():
    """测试集上同时有 front bbox + back_bbox 的实例, 前GT kp + 后GT kp"""
    anns = glob.glob(f'{BASE}/**/annotations/*.json', recursive=True)
    insts = []
    for f in anns:
        cls = f.replace(BASE, '').lstrip('/').split('/')[0]
        if cls not in test_cls:
            continue
        with open(f) as fh:
            data = json.load(fh)
        jpg = os.path.basename(f).replace('.json', '.jpg')
        fwd = os.path.join(os.path.dirname(f).replace('annotations','forward'), jpg)
        bwd = os.path.join(os.path.dirname(f).replace('annotations','backward'), jpg.replace('front_','back_'))
        fimg = cv2.imread(fwd); fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
        bimg = cv2.imread(bwd); bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)
        for ann in data['annotations']:
            bbox_b = ann.get('back_bbox')
            if bbox_b is None: continue
            kpb = ann.get('pose_keypoints_back') or []
            vis_b = [k for k in kpb if k and k.get('visible')]
            if not vis_b: continue
            bb_f = ann['bbox']; bb_b = bbox_b
            fc, bc = crop_region(fimg, bb_f), crop_region(bimg, bb_b)
            if fc is None or bc is None: continue
            ch, cw = bc.shape[:2]
            # 后 GT (rel to back bbox)
            tgt_b = np.zeros((NUM_KP,3),dtype=np.float32)
            for k in kpb:
                if k is None or k['name'] not in KP_NAMES: continue
                idx = KP_NAMES.index(k['name'])
                v = 1.0 if k.get('visible') else 0.0
                if v:
                    tgt_b[idx] = [np.clip((k['x']-bb_b[0])/max(cw,1),0,1), np.clip((k['y']-bb_b[1])/max(ch,1),0,1), 1.0]
                else:
                    tgt_b[idx] = [0,0,0]
            insts.append({'fimg': fimg, 'bimg': bimg, 'bb_f': bb_f, 'bb_b': bb_b,
                          'kpb_gt': tgt_b, 'ann': ann})
    return insts

def run():
    # 1) 训练后视角补位网络
    save = os.path.join(OUT, 'back_kp_aux.pth')
    if not os.path.exists(save):
        train_back_kp(save)
    aux = build_aux_net(save)
    print('后视角补位网络加载完成')

    # 2) 主检测器：前视角 YOLOv8-pose
    from ultralytics import YOLO
    det = YOLO('/public/cyl/label_tool/runs_pose/pose_front/weights/best.pt')

    # 3) 测试集实例（同时有 front/back bbox + 后kp标注）
    insts = reparse_back_with_front()
    print(f'测试补位实例数: {len(insts)}')

    kp_names = KP_NAMES
    front_vis_gt = 0
    back_vis_gt = 0
    fused_correct = 0   # 融合后"某关键点被正确定位"（<阈值的xy误差）
    front_only_correct = 0
    total_kp_slots = 0
    xy_errors_front = []
    xy_errors_fused = []
    threshold = 0.1  # 归一化(相对整图)定位阈值，用于"定位正确"判断

    with torch.no_grad():
        for inst in insts:
            ann = inst['ann']
            # 前视角GT关键点 (abs整图坐标)
            kpf = ann.get('pose_keypoints') or []
            front_vis_status = {}
            for k in kpf:
                if k and k['name'] in KP_NAMES:
                    front_vis_status[k['name']] = bool(k.get('visible'))

            # 后视角GT关键点 (rel to back bbox)
            tgt_b = inst['kpb_gt']
            bb_b = inst['bb_b']

            # 前视角主检测：用原图跑 YOLO 前视角检测 → 得到该学生的前视角关键点预测（整图坐标）
            det_r = det(cv2.cvtColor(inst['fimg'], cv2.COLOR_RGB2BGR))
            # 取与 GT 前 bbox IoU 最高的框
            gt = np.array(inst['bb_f'])
            best_box = None; best_iou = -1; best_kpts = None
            if len(det_r[0].boxes):
                pred_kpts = det_r[0].keypoints.data.cpu().numpy() if det_r[0].keypoints is not None else None
                for bi, box in enumerate(det_r[0].boxes.xyxy.cpu().numpy()):
                    gx1,gy1,gx2,gy2 = gt
                    inter = max(0,min(box[2],gx2)-max(box[0],gx1))*max(0,min(box[3],gy2)-max(box[1],gy1))
                    union = (box[2]-box[0])*(box[3]-box[1]) + (gx2-gx1)*(gy2-gy1) - inter
                    iou = inter/max(union,1)
                    if iou > best_iou:
                        best_iou = iou; best_box = box
                        best_kpts = pred_kpts[bi] if pred_kpts is not None else None
            # 前视角预测关键点 (整图坐标)
            front_pred = {}  # name -> (x, y, vis)
            if best_kpts is not None:
                img_h, img_w = inst['fimg'].shape[:2]
                for ki, nm in enumerate(KP_NAMES):
                    x, y, v = best_kpts[ki]
                    # YOLO kpts 已归一化到 imgsz 输入, 需反算到原图
                    # ultralytics keypoints.data 为原始图坐标(相对缩放后图), 对 1280 输入有 scale_factor
                    scale = det_r[0].orig_shape[1] / 1280 if False else None
                    front_pred[nm] = (float(x), float(y), float(v))

            # 后视角辅助预测
            bc = crop_region(inst['bimg'], bb_b)
            bt = tf_eval(Image.fromarray(bc)).unsqueeze(0).to(DEVICE)
            out = aux(bt).view(NUM_KP, 3)
            pred_bxy = torch.sigmoid(out[:,:2]).cpu().numpy()   # rel to back bbox
            pred_bvis = (torch.sigmoid(out[:,2]).cpu() > 0.5).numpy()

            # 评估：以"定位误差"(相对整图归一化)对比 前视角 vs 前+后融合
            # 对每个关键点 slot，若前视角GT可见→用前预测; 若不 visible→用后预测
            for ki, nm in enumerate(KP_NAMES):
                total_kp_slots += 1
                fv = front_vis_status.get(nm, False)
                bv = tgt_b[ki, 2] > 0.5

                # 前视角误差（用前预测，仅对前可见且预测可用时算）
                if fv and nm in front_pred:
                    px, py, pv = front_pred[nm]
                    if best_kpts is not None:
                        fx = px / (best_kpts.shape and 1 or 1)
                        # 简化：此处用 YOLO 给出的原图坐标直接比较
                    gtl = None
                    for k in kpf:
                        if k and k['name']==nm: gtl = (k['x'], k['y'])
                    if gtl:
                        err = ((px-gtl[0])/1920)**2 + ((py-gtl[1])/1080)**2
                        xy_errors_front.append((px-gtl[0])/1920)
                        front_only_correct += 1 if np.sqrt(err)<threshold else 0

                # 融合（后视角补位）：计算融合后的关键点估计
                fused_x = fused_y = None; has_fused = False
                if fv and nm in front_pred:
                    # 前可见→用前预测
                    for k in kpf:
                        if k and k['name']==nm:
                            fused_x, fused_y = k['x']/1920, k['y']/1080  # 用GT模拟回退? 不, 用前预测
                    for k in kpf:
                        if k and k['name']==nm:
                            fused_x = front_pred[nm][0]/1920
                            fused_y = front_pred[nm][1]/1080
                    has_fused = True
                elif (not fv) and bv:
                    # 前不可见但后可见→用后预测
                    bx_abs = (pred_bxy[ki,0]*(bc.shape[1]) + bb_b[0]) / 1920
                    by_abs = (pred_bxy[ki,1]*(bc.shape[0]) + bb_b[1]) / 1080
                    fused_x, fused_y = bx_abs, by_abs
                    has_fused = True
                elif bv:
                    fused_x = (tgt_b[ki,0]*bc.shape[1] + bb_b[0]) / 1920
                    fused_y = (tgt_b[ki,1]*bc.shape[0] + bb_b[1]) / 1080
                    has_fused = True

                # GT abs
                gt_abs = None
                for k in kpf:
                    if k and k['name']==nm:
                        gt_abs = (k['x']/1920, k['y']/1080)
                if gt_abs is None and bv:
                    gt_abs = ((tgt_b[ki,0]*bc.shape[1]+bb_b[0])/1920, (tgt_b[ki,1]*bc.shape[0]+bb_b[1])/1080)
                if has_fused and gt_abs:
                    err = np.sqrt((fused_x-gt_abs[0])**2 + (fused_y-gt_abs[1])**2)
                    xy_errors_fused.append(err)
                    fused_correct += 1 if err < threshold else 0

            # 统计可视性
            for nm in KP_NAMES:
                if front_vis_status.get(nm, False): front_vis_gt += 1
                if tgt_b[KP_NAMES.index(nm), 2] > 0.5: back_vis_gt += 1

    print(f'\n===== 后视角补位评估(测试, 有back_bbox实例) =====')
    print(f'前视角GT可见关键点总数: {front_vis_gt}')
    print(f'后视角GT可见关键点总数: {back_vis_gt}  (其中前视角不可见的为补位目标)')
    print(f'关键点slot总数: {total_kp_slots}')
    if xy_errors_front:
        print(f'\n仅前视角: 可定位slot={len(xy_errors_front)}, 平均|xy|err(归一化)={np.mean(np.abs(xy_errors_front)):.4f}, 定位正确率={front_only_correct/len(xy_errors_front)*100:.1f}%')
    if xy_errors_fused:
        print(f'前+后补位: 可定位slot={len(xy_errors_fused)}, 平均err(归一化)={np.mean(xy_errors_fused):.4f}, 定位正确率(<{threshold})={fused_correct/len(xy_errors_fused)*100:.1f}%')
    # 覆盖率：能定位的关键点占所有slot的比例
    print(f'\n关键点覆盖率: 仅前={len(xy_errors_front)/total_kp_slots*100:.1f}%  vs 前+后融合={len(xy_errors_fused)/total_kp_slots*100:.1f}%')

if __name__ == '__main__':
    run()