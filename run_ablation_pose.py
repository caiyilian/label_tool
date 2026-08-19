import os, sys, time, json
from train_pose import run_pose

# 依次训练消融变体
variants = [
    ('back', 100, 1280),
    ('both', 100, 1280),
    ('front', 60, 1280),   # 再用 micro 探索
]

for mode, epochs, imgsz in variants:
    try:
        run_pose(mode, epochs=epochs, imgsz=imgsz)
    except Exception as e:
        print(f"FAILED {mode}: {e}")