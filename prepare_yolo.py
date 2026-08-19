import os, json, glob, shutil, re
from collections import defaultdict

BASE = '/public/cyl/label_tool/坐姿视频标记-20260722'
OUT = '/public/cyl/label_tool/yolo_data'
IMG_SIZE = (1920, 1080)
KP_NAMES = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand',
            'left_shoulder', 'left_elbow', 'left_hand']

status_map = {'sitting_listening': 0, 'sitting_reading': 1, 'standing': 2, 'left_seat': 3}
posture_map = {'normal': 0, 'bad': 1}
bad_reason_map = {'head_tilt': 0, 'head_down': 1, 'lean_forward': 2, 'side_lean': 3}

# yolo data/example/annotations dirs
for split in ['yolo']:
    os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'labels_full'), exist_ok=True)

idx = 0
meta = {}  # idx -> {class, frame, is_back, json_path, status_dist}

ann_files = sorted(glob.glob(f'{BASE}/**/annotations/*.json', recursive=True))

for fpath in ann_files:
    rel = fpath.replace(BASE, '').lstrip('/')
    cls_name = rel.split('/')[0]
    with open(fpath) as f:
        data = json.load(f)

    jpg = os.path.basename(fpath).replace('.json', '.jpg')
    front_img = os.path.join(os.path.dirname(fpath).replace('annotations', 'forward'), jpg)
    back_img = os.path.join(os.path.dirname(fpath).replace('annotations', 'backward'), jpg.replace('front_', 'back_'))

    for view, img_path in [('front', front_img), ('back', back_img)]:
        if not os.path.exists(img_path):
            print(f'missing {img_path}')
            continue

        labels = []
        labels_full = []
        for ann in data.get('annotations', []):
            sid = ann['student_id']
            if view == 'front':
                bx = ann['bbox']
                kps = ann.get('pose_keypoints') or []
            else:
                bx = ann.get('back_bbox') or ann['bbox']
                kps = ann.get('pose_keypoints_back') or []

            x1, y1, x2, y2 = map(float, bx)
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            cx = (x1 + x2) / 2 / IMG_SIZE[0]
            cy = (y1 + y2) / 2 / IMG_SIZE[1]
            nw = w / IMG_SIZE[0]
            nh = h / IMG_SIZE[1]

            # 关键点归一化（相对整图）
            kp_feat = [0.0] * (len(KP_NAMES) * 3)
            for kp in kps:
                if kp is None:
                    continue
                nm = kp.get('name')
                if nm not in KP_NAMES:
                    continue
                ki = KP_NAMES.index(nm)
                vis = 1 if kp.get('visible') else 0
                if vis:
                    kp_feat[ki*3]   = kp['x'] / IMG_SIZE[0]
                    kp_feat[ki*3+1] = kp['y'] / IMG_SIZE[1]
                    kp_feat[ki*3+2] = vis
                else:
                    kp_feat[ki*3] = 0
                    kp_feat[ki*3+1] = 0
                    kp_feat[ki*3+2] = 0

            # YOLO pose 格式: cls cx cy w h kp...
            labels.append([0, cx, cy, nw, nh] + kp_feat)

            # extended: 附上 status/posture/br 索引（供自定义多任务用）
            st = status_map[ann['status']]
            po = posture_map[ann['posture']]
            br = [0]*4
            for r in ann.get('bad_reason', []):
                if r in bad_reason_map:
                    br[bad_reason_map[r]] = 1
            labels_full.append({'idx': idx, 'sid': sid,
                                 'bbox': [x1, y1, x2, y2],
                                 'status': st, 'posture': po, 'br': br,
                                 'kps': kp_feat,
                                 'cls': cls_name, 'frame': jpg, 'view': view})

        # 复制图片（用硬链接省空间）
        dst_img = os.path.join(OUT, 'images', f'{idx:06d}.jpg')
        if not os.path.exists(dst_img):
            try:
                os.link(img_path, dst_img)
            except OSError:
                shutil.copy(img_path, dst_img)

        with open(os.path.join(OUT, 'labels', f'{idx:06d}.txt'), 'w') as f:
            for line in labels:
                f.write(' '.join(f'{v:.6f}' for v in line) + '\n')

        meta[idx] = {
            'class': cls_name, 'frame': jpg, 'view': view,
            'json': fpath, 'annotations': labels_full,
        }
        idx += 1

with open(os.path.join(OUT, 'meta.json'), 'w') as f:
    json.dump(meta, f)

print(f'Total images: {idx}')
print('Meta saved to', os.path.join(OUT, 'meta.json'))

# 输出 train/val/test 划分（按班级）
classes = sorted(set(m['class'] for m in meta.values()))
import random
random.seed(42)
random.shuffle(classes)
n = len(classes)
tr, va, te = classes[:int(n*0.7)], classes[int(n*0.7):int(n*0.85)], classes[int(n*0.85):]
print(f'Train classes: {tr}')
print(f'Val classes: {va}')
print(f'Test classes: {te}')

def write_split(ids, name):
    with open(os.path.join(OUT, f'{name}.txt'), 'w') as f:
        for i in ids:
            f.write(os.path.join(OUT, 'images', f'{i:06d}.jpg') + '\n')

tr_ids = [i for i, m in meta.items() if m['class'] in tr]
va_ids = [i for i, m in meta.items() if m['class'] in va]
te_ids = [i for i, m in meta.items() if m['class'] in te]

# 前后视角分开
for vid, name in [(tr_ids, 'train'), (va_ids, 'val'), (te_ids, 'test')]:
    front_ids = [i for i in vid if meta[i]['view'] == 'front']
    back_ids = [i for i in vid if meta[i]['view'] == 'back']
    write_split(front_ids, f'{name}_front')
    write_split(back_ids, f'{name}_back')

print('Split files written.')
print(f'Train (front/back): {sum(1 for i in tr_ids if meta[i]["view"]=="front")}/{sum(1 for i in tr_ids if meta[i]["view"]=="back")}')
print(f'Val (front/back): {sum(1 for i in va_ids if meta[i]["view"]=="front")}/{sum(1 for i in va_ids if meta[i]["view"]=="back")}')
print(f'Test (front/back): {sum(1 for i in te_ids if meta[i]["view"]=="front")}/{sum(1 for i in te_ids if meta[i]["view"]=="back")}')

# 写 yaml
with open(os.path.join(OUT, 'pose.yaml'), 'w') as f:
    f.write(f"""path: {OUT}
train: train_front.txt
val: val_front.txt
test: test_front.txt
nc: 1
names: ['student']
kpt_shape: [8, 3]
flip_idx: [0, 1, 4, 5, 6, 2, 3, 7]
""")
print('pose.yaml written')