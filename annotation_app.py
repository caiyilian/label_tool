import os
import sys
import json
import glob
import threading
import webbrowser
import urllib.request
import queue
import tkinter as tk
from tkinter import filedialog
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

dialog_queue = queue.Queue()
dialog_result = {}
dialog_event = threading.Event()

data_folder = None
front_files = []
back_files = []
annotations_dir = None

def get_frame_list():
    global front_files, back_files
    front_dir = os.path.join(data_folder, "forward")
    back_dir = os.path.join(data_folder, "backward")
    front_files = sorted(glob.glob(os.path.join(front_dir, "front_*.jpg")))
    back_files = sorted(glob.glob(os.path.join(back_dir, "back_*.jpg")))
    return len(front_files)

def get_annotations_dir():
    global annotations_dir
    annotations_dir = os.path.join(data_folder, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    return annotations_dir

@app.route('/')
def index():
    return render_template('annotation.html')

@app.route('/api/select_folder', methods=['POST'])
def select_folder():
    global data_folder
    dialog_event.clear()
    dialog_queue.put({'type': 'folder'})
    dialog_event.wait()
    folder = dialog_result.get('path', '')
    if not folder:
        return jsonify({"error": "canceled"})

    forward_dir = os.path.join(folder, "forward")
    backward_dir = os.path.join(folder, "backward")

    if not os.path.isdir(forward_dir):
        return jsonify({"error": f"所选文件夹中没有 forward 子目录\n当前路径: {folder}"})
    if not os.path.isdir(backward_dir):
        return jsonify({"error": f"所选文件夹中没有 backward 子目录\n当前路径: {folder}"})

    front_imgs = sorted(glob.glob(os.path.join(forward_dir, "front_*.jpg")))
    back_imgs = sorted(glob.glob(os.path.join(backward_dir, "back_*.jpg")))

    if len(front_imgs) == 0:
        return jsonify({"error": "forward 目录中没有找到 front_*.jpg 图片"})
    if len(back_imgs) == 0:
        return jsonify({"error": "backward 目录中没有找到 back_*.jpg 图片"})
    if len(front_imgs) != len(back_imgs):
        return jsonify({"error": f"前后视角图片数量不一致！\nforward: {len(front_imgs)} 张\nbackward: {len(back_imgs)} 张"})

    data_folder = folder
    get_frame_list()
    get_annotations_dir()

    return jsonify({
        "path": folder,
        "total_frames": len(front_files),
        "forward_dir": forward_dir,
        "backward_dir": backward_dir
    })

@app.route('/api/frame_image/<int:index>/<view>')
def frame_image(index, view):
    from flask import send_file
    if view == 'front':
        files = front_files
    elif view == 'back':
        files = back_files
    else:
        return jsonify({"error": "invalid view"}), 400

    if index < 0 or index >= len(files):
        return jsonify({"error": "index out of range"}), 400

    return send_file(files[index])

@app.route('/api/frame_info/<int:index>')
def frame_info(index):
    if index < 0 or index >= len(front_files):
        return jsonify({"error": "index out of range"}), 400

    front_name = os.path.basename(front_files[index])
    back_name = os.path.basename(back_files[index])

    return jsonify({
        "index": index,
        "total": len(front_files),
        "front_filename": front_name,
        "back_filename": back_name
    })

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    index = data.get('index')
    annotation = data.get('annotation')

    if index is None:
        return jsonify({"error": "missing index"}), 400

    global annotations_dir
    if not os.path.isdir(annotations_dir):
        os.makedirs(annotations_dir, exist_ok=True)

    front_name = os.path.basename(front_files[index])
    json_name = front_name.replace('.jpg', '.json')
    json_path = os.path.join(annotations_dir, json_name)

    if annotation is None or (isinstance(annotation, dict) and not annotation.get('annotations')):
        if os.path.exists(json_path):
            os.remove(json_path)
        return jsonify({"success": True, "deleted": True})

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "path": json_path})

@app.route('/api/load_annotation/<int:index>')
def load_annotation(index):
    if index < 0 or index >= len(front_files):
        return jsonify({"error": "index out of range"}), 400

    front_name = os.path.basename(front_files[index])
    json_name = front_name.replace('.jpg', '.json')
    json_path = os.path.join(annotations_dir, json_name)

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        return jsonify({"annotation": annotation, "exists": True})
    else:
        return jsonify({"annotation": None, "exists": False})

@app.route('/api/batch_status')
def batch_status():
    statuses = []
    for i in range(len(front_files)):
        front_name = os.path.basename(front_files[i])
        json_name = front_name.replace('.jpg', '.json')
        json_path = os.path.join(annotations_dir, json_name)
        statuses.append(os.path.exists(json_path))
    return jsonify({"statuses": statuses})

def run_flask():
    app.run(port=5001, debug=False, use_reloader=False)

def open_browser():
    webbrowser.open('http://127.0.0.1:5001')

def on_closing(root):
    os._exit(0)

def process_dialogs(root):
    try:
        req = dialog_queue.get_nowait()
        if req['type'] == 'focus':
            if root.state() == 'iconic':
                root.deiconify()
            root.attributes('-topmost', True)
            root.attributes('-topmost', False)
            root.lift()
            root.focus_force()
            open_browser()
        else:
            top = tk.Toplevel(root)
            top.attributes('-topmost', True)
            top.withdraw()
            if req['type'] == 'folder':
                d = filedialog.askdirectory(title="选择数据文件夹（需包含 forward 和 backward 子目录）", parent=top)
                dialog_result['path'] = d
            top.destroy()
            dialog_event.set()
    except queue.Empty:
        pass
    root.after(100, lambda: process_dialogs(root))

if __name__ == '__main__':
    try:
        req = urllib.request.Request('http://127.0.0.1:5001/api/batch_status', method='GET')
        with urllib.request.urlopen(req, timeout=1) as response:
            if response.status == 200:
                print("检测到程序已在运行，已唤起已有窗口。")
                os._exit(0)
    except Exception:
        pass

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    root = tk.Tk()
    root.title("坐姿标注工具 - 控制台")
    root.geometry("380x180")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))

    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'{w}x{h}+{x}+{y}')

    tk.Label(root, text="服务正在运行中", font=("微软雅黑", 14, "bold"), fg="green").pack(pady=(20, 5))
    tk.Label(root, text="请保持此窗口打开，关闭窗口即停止服务", font=("微软雅黑", 10), fg="#666666").pack(pady=(0, 20))

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    btn_open = tk.Button(btn_frame, text="打开网页界面", command=open_browser, width=15, font=("微软雅黑", 10), cursor="hand2")
    btn_open.pack(side=tk.LEFT, padx=10)

    btn_exit = tk.Button(btn_frame, text="关闭并退出", command=lambda: on_closing(root), width=15, font=("微软雅黑", 10), cursor="hand2")
    btn_exit.pack(side=tk.LEFT, padx=10)

    root.after(100, lambda: process_dialogs(root))
    root.after(1000, open_browser)

    root.mainloop()
