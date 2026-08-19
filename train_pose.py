import os, sys, json, shutil, time
import numpy as np
from ultralytics import YOLO
import torch

OUT = '/public/cyl/label_tool/yolo_data'
RUNS = '/public/cyl/label_tool/runs_pose'
os.makedirs(RUNS, exist_ok=True)

def make_yaml(mode, out_yaml):
    base = {
        'path': OUT,
        'nc': 1,
        'names': ['student'],
        'kpt_shape': [8, 3],
        'flip_idx': [0, 1, 4, 5, 6, 2, 3, 7],
    }
    if mode == 'front':
        base.update({'train': 'train_front.txt', 'val': 'val_front.txt', 'test': 'test_front.txt'})
    elif mode == 'back':
        base.update({'train': 'train_back.txt', 'val': 'val_back.txt', 'test': 'test_back.txt'})
    elif mode == 'both':
        # 前后图各自作为独立样本合训（同一帧两个视角都作为样本）
        import glob
        with open(os.path.join(OUT, 'train_all.txt'), 'w') as f:
            for line in open(os.path.join(OUT, 'train_front.txt')):
                f.write(line)
            for line in open(os.path.join(OUT, 'train_back.txt')):
                f.write(line)
        with open(os.path.join(OUT, 'val_all.txt'), 'w') as f:
            for line in open(os.path.join(OUT, 'val_front.txt')):
                f.write(line)
            for line in open(os.path.join(OUT, 'val_back.txt')):
                f.write(line)
        with open(os.path.join(OUT, 'test_all.txt'), 'w') as f:
            for line in open(os.path.join(OUT, 'test_front.txt')):
                f.write(line)
            for line in open(os.path.join(OUT, 'test_back.txt')):
                f.write(line)
        base.update({'train': 'train_all.txt', 'val': 'val_all.txt', 'test': 'test_all.txt'})
    with open(out_yaml, 'w') as f:
        yaml_txt = '\n'.join([f'{k}: {v}' for k, v in base.items()])
        f.write(yaml_txt)
    return out_yaml

def run_pose(mode, epochs=100, imgsz=1280, name=None, init='yolov8n-pose.pt'):
    name = name or f'pose_{mode}'
    yaml_path = make_yaml(mode, os.path.join(OUT, f'pose_{mode}.yaml'))
    print(f'\n===== Training pose [{mode}] epochs={epochs} imgsz={imgsz} =====')
    model = YOLO(init)
    t0 = time.time()
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0,
        workers=8,
        patience=50,
        project=RUNS,
        name=name,
        exist_ok=True,
        lr0=1e-3,
        weight_decay=5e-4,
        optimizer='AdamW',
        cos_lr=True,
        cache=False,
        verbose=True,
    )
    elapsed = time.time() - t0
    m = model.val(data=yaml_path, imgsz=imgsz, split='test', verbose=False)
    print(f'Done {name} in {elapsed:.0f}s. mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} '
          f'kp_map50={m.pose.map50:.4f} kp_map50-95={m.pose.map:.4f}')

    summary = {
        'mode': mode, 'epochs': epochs, 'imgsz': imgsz, 'time': elapsed,
        'box_map50': float(m.box.map50), 'box_map50_95': float(m.box.map),
        'kp_map50': float(m.pose.map50), 'kp_map50_95': float(m.pose.map),
        'kp_map50_ap50': float(m.pose.means[0]) if hasattr(m.pose, 'means') else None,
    }
    result_file = os.path.join(RUNS, 'results.json')
    if os.path.exists(result_file):
        res = json.load(open(result_file))
    else:
        res = []
    res.append(summary)
    json.dump(res, open(result_file, 'w'), indent=2)
    print(f'Appended to {result_file}')
    return summary

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'front'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_pose(mode, epochs=epochs)