import os
import sys

if getattr(sys, 'frozen', False):
    app_root = sys._MEIPASS
else:
    app_root = os.path.dirname(os.path.abspath(__file__))

tcl_lib_path = os.path.join(app_root, "tcl_tk_libs", "tcl")
tk_lib_path = os.path.join(app_root, "tcl_tk_libs", "tk")
if os.path.exists(tcl_lib_path) and os.path.exists(tk_lib_path):
    os.environ["TCL_LIBRARY"] = tcl_lib_path
    os.environ["TK_LIBRARY"] = tk_lib_path

import tkinter as tk
from tkinter import filedialog
import cv2
import time
import threading
import subprocess
import re
import glob
import base64
import webbrowser
import logging
import queue
import urllib.request
import json
from flask import Flask, render_template, request, jsonify, Blueprint, send_file

log_flask = logging.getLogger('werkzeug')
log_flask.disabled = True

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

dialog_queue = queue.Queue()
dialog_result = {}
dialog_event = threading.Event()

extract_bp = Blueprint('extract', __name__, url_prefix='/extract')
annotate_bp = Blueprint('annotate', __name__, url_prefix='/annotate')

@app.route('/')
def launcher():
    return render_template('launcher.html')

# ========== Extract Tool ==========

extract_state = {
    "is_processing": False,
    "stop_requested": False,
    "current_process": None,
    "progress1": {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"},
    "progress2": {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"},
    "logs": ["系统初始化完毕", "等待配置视频源..."]
}

def extract_log(msg):
    time_str = time.strftime("%H:%M:%S")
    formatted = f"[{time_str}] {msg}"
    print(formatted)
    extract_state["logs"].append(formatted)
    if len(extract_state["logs"]) > 200:
        extract_state["logs"].pop(0)

@extract_bp.route('/')
def extract_index():
    return render_template('index.html')

@extract_bp.route('/api/select_video', methods=['POST'])
def extract_select_video():
    dialog_event.clear()
    dialog_queue.put({'type': 'video'})
    dialog_event.wait()
    filepath = dialog_result.get('filepath', '')
    if not filepath:
        return jsonify({"error": "canceled"})
    basename = os.path.basename(filepath)
    for char in basename:
        if '\u4e00' <= char <= '\u9fff':
            return jsonify({"error": f"视频文件名不能包含中文，请重命名后重试！\n当前文件名：{basename}"})
    cwd = os.getcwd()
    first_frame_b64 = ""
    width, height, fps, total_frames = 0, 0, 25.0, 0
    duration = 0
    try:
        os.chdir(os.path.dirname(filepath))
        cap = cv2.VideoCapture(basename)
        ret, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0:
            duration = total_frames / fps
        cap.release()
        if ret:
            display_w = min(1280, width)
            scale = display_w / width
            display_h = int(height * scale)
            resized = cv2.resize(frame, (display_w, display_h))
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            first_frame_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        os.chdir(cwd)
    return jsonify({"path": filepath, "name": basename, "width": width, "height": height, "fps": fps, "duration": duration, "total_frames": total_frames, "first_frame": first_frame_b64})

@extract_bp.route('/api/select_dir', methods=['POST'])
def extract_select_dir():
    dialog_event.clear()
    dialog_queue.put({'type': 'dir'})
    dialog_event.wait()
    d = dialog_result.get('path', '')
    if d:
        return jsonify({"path": d})
    return jsonify({"error": "canceled"})

def format_time(seconds):
    if seconds < 0: return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def process_videos_thread(data):
    try:
        v1 = data['v1']
        v2 = data['v2']
        out = data['out']
        t1_sec = data['t1_sec']
        t2_sec = data['t2_sec']
        res_limit = data.get('res_limit')
        sample_t1 = data.get('sample_t1', 5)
        sample_t2 = data.get('sample_t2', 1)
        sample_t3 = data.get('sample_t3', 10)
        forward_dir = os.path.join(out, "forward")
        backward_dir = os.path.join(out, "backward")
        os.makedirs(forward_dir, exist_ok=True)
        os.makedirs(backward_dir, exist_ok=True)
        diff = t1_sec - t2_sec
        skip_sec1 = max(0, -diff)
        skip_sec2 = max(0, diff)
        extract_log(f"配置读取: V1={t1_sec}s, V2={t2_sec}s")
        extract_log(f"解析指令: V1跳过 {skip_sec1}s, V2跳过 {skip_sec2}s")
        extract_log(f"采样策略: 每 {sample_t1} 分钟采样 {sample_t2} 分钟，每 {sample_t3} 秒取 1 帧")

        def extract(video_path, output_folder, skip_seconds, prefix, prog_key):
            extract_state[prog_key]["status"] = "处理中"
            cap = cv2.VideoCapture(video_path)
            total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 25.0
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            file_size_bytes = os.path.getsize(video_path)
            cv2_duration = total_frames_in_video / fps if fps else 0
            cv2_bitrate = (file_size_bytes * 8) / cv2_duration if cv2_duration > 0 else 0
            is_corrupted = cv2_bitrate < 500000
            if is_corrupted:
                estimated_bitrate = width * height * fps * 0.022
                if estimated_bitrate <= 0: estimated_bitrate = 2000000
                estimated_real_duration = (file_size_bytes * 8) / estimated_bitrate
                total_frames_in_video = estimated_real_duration * fps
            fps_round = round(fps)
            skip_frames = int(skip_seconds * fps_round + fps_round // 2)
            window_period_frames = int(sample_t1 * 60 * fps_round)
            window_duration_frames = int(sample_t2 * 60 * fps_round)
            frame_interval = int(sample_t3 * fps_round)
            total_duration_sec = (total_frames_in_video - skip_frames) / fps_round
            num_windows = int(total_duration_sec / (sample_t1 * 60))
            frames_per_window = int(sample_t2 * 60 / sample_t3)
            expected_total_images = num_windows * frames_per_window
            if expected_total_images <= 0: expected_total_images = 1
            extract_state[prog_key]["total"] = expected_total_images
            if is_corrupted:
                extract_log(f"{prefix} 检测到时间戳异常。智能预估真实时长 {estimated_real_duration/60:.1f}m，提取 {expected_total_images}帧")
            out_w, out_h = int(width), int(height)
            if res_limit:
                max_w, max_h = res_limit
                if out_w > max_w or out_h > max_h:
                    scale_factor = min(max_w / out_w, max_h / out_h)
                    out_w = int(out_w * scale_factor)
                    out_h = int(out_h * scale_factor)
                    out_h = out_h if out_h % 2 == 0 else out_h - 1
                    out_w = out_w if out_w % 2 == 0 else out_w - 1
            start_time = time.time()
            out_pattern = os.path.join(output_folder, f"{prefix}_%06d.jpg")
            filter_str = f"select='gte(n,{skip_frames})*lt(mod((n-{skip_frames}),{window_period_frames}),{window_duration_frames})*not(mod((n-{skip_frames}),{frame_interval}))'"
            if out_w != int(width) or out_h != int(height):
                filter_str += f",scale={out_w}:{out_h}"
            if getattr(sys, 'frozen', False):
                app_dir = os.path.dirname(sys.executable)
            else:
                app_dir = os.path.dirname(os.path.abspath(__file__))
            ffmpeg_path = os.path.join(app_dir, "ffmpeg.exe")
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = os.path.join(app_dir, "ffmpeg")
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = "ffmpeg"
            cmd = [ffmpeg_path, "-y", "-i", video_path, "-vf", filter_str, "-vsync", "0", "-q:v", "2", out_pattern]
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='ignore', creationflags=creationflags)
            extract_state["current_process"] = process
            frame_regex = re.compile(r"frame=\s*(\d+)")
            saved_count = 0
            for line in process.stderr:
                if extract_state["stop_requested"]:
                    process.terminate()
                    break
                match = frame_regex.search(line)
                if match:
                    saved_count = int(match.group(1))
                    elapsed = time.time() - start_time
                    if saved_count > 0:
                        eta_seconds = (elapsed / saved_count) * (expected_total_images - saved_count)
                        eta_str = format_time(eta_seconds)
                    else:
                        eta_str = "计算中..."
                    percent = min(100, int((saved_count / expected_total_images) * 100))
                    extract_state[prog_key]["current"] = saved_count
                    extract_state[prog_key]["percent"] = percent
                    extract_state[prog_key]["eta"] = eta_str
            process.wait()
            if extract_state["stop_requested"]:
                extract_log(f"🚫 {prefix} 提取已被强行终止")
                extract_state[prog_key]["status"] = "已终止"
                return saved_count
            if process.returncode != 0:
                extract_log(f"❌ {prefix} ffmpeg 异常退出 (code: {process.returncode})，提取未完成。")
                extract_state[prog_key]["status"] = "异常退出"
                extract_state[prog_key]["eta"] = "失败"
                raise Exception(f"{prefix} ffmpeg 处理崩溃或被强杀")
            extract_state[prog_key]["current"] = expected_total_images
            extract_state[prog_key]["percent"] = 100
            extract_state[prog_key]["eta"] = "完成"
            extract_state[prog_key]["status"] = "完成"
            extract_log(f"✅ {prefix} 提取完成，共落盘 {saved_count} 帧")
            return saved_count

        count1 = extract(v1, forward_dir, skip_sec1, "front", "progress1")
        if not extract_state["stop_requested"]:
            extract_state["progress2"]["status"] = "准备中"
            count2 = extract(v2, backward_dir, skip_sec2, "back", "progress2")
        else:
            count2 = 0
        if not extract_state["stop_requested"]:
            min_count = min(count1, count2)
            extract_log(f"对齐校验: 以较短长度 ({min_count}) 为基准裁剪冗余尾帧")
            def trim_tails(folder, prefix, keep_count):
                files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.jpg")))
                for f in files[keep_count:]:
                    os.remove(f)
            trim_tails(forward_dir, "front", min_count)
            trim_tails(backward_dir, "back", min_count)
            extract_log("🎉 任务圆满结束！双视角帧数已严格等量对齐")
        else:
            extract_log("🚫 任务流程已终止，文件可能不完整")
    except Exception as e:
        extract_log(f"❌ 发生异常: {str(e)}")
    finally:
        extract_state["is_processing"] = False
        extract_state["stop_requested"] = False
        extract_state["current_process"] = None

@extract_bp.route('/api/start', methods=['POST'])
def extract_start():
    data = request.json
    if extract_state["is_processing"]:
        return jsonify({"error": "已经在运行中"})
    extract_state["is_processing"] = True
    extract_state["stop_requested"] = False
    extract_state["progress1"] = {"percent": 0, "current": 0, "total": 0, "eta": "计算中...", "status": "初始化"}
    extract_state["progress2"] = {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"}
    extract_state["logs"] = []
    extract_log("🚀 引擎点火，任务开始执行")
    threading.Thread(target=process_videos_thread, args=(data,), daemon=True).start()
    return jsonify({"success": True})

@extract_bp.route('/api/stop', methods=['POST'])
def extract_stop():
    if extract_state["is_processing"]:
        extract_state["stop_requested"] = True
        extract_log("⚠️ 接收到用户终止指令，正在下发中断信号...")
        if extract_state["current_process"]:
            try:
                extract_state["current_process"].terminate()
            except:
                pass
    return jsonify({"success": True})

@extract_bp.route('/api/status', methods=['GET'])
def extract_status():
    return jsonify({"is_processing": extract_state["is_processing"], "progress1": extract_state["progress1"], "progress2": extract_state["progress2"], "logs": extract_state["logs"]})

# ========== Annotate Tool ==========

annotate_data_folder = None
annotate_front_files = []
annotate_back_files = []
annotate_annotations_dir = None

def annotate_get_frame_list():
    global annotate_front_files, annotate_back_files
    front_dir = os.path.join(annotate_data_folder, "forward")
    back_dir = os.path.join(annotate_data_folder, "backward")
    front_all = sorted(glob.glob(os.path.join(front_dir, "front_*.jpg")))
    back_all = sorted(glob.glob(os.path.join(back_dir, "back_*.jpg")))
    front_map = {}
    for f in front_all:
        name = os.path.basename(f)
        num = name.replace("front_", "").replace(".jpg", "")
        front_map[num] = f
    back_map = {}
    for f in back_all:
        name = os.path.basename(f)
        num = name.replace("back_", "").replace(".jpg", "")
        back_map[num] = f
    common_nums = sorted(set(front_map.keys()) & set(back_map.keys()))
    annotate_front_files = [front_map[n] for n in common_nums]
    annotate_back_files = [back_map[n] for n in common_nums]
    return len(annotate_front_files)

def annotate_get_annotations_dir():
    global annotate_annotations_dir
    annotate_annotations_dir = os.path.join(annotate_data_folder, "annotations")
    os.makedirs(annotate_annotations_dir, exist_ok=True)
    return annotate_annotations_dir

@annotate_bp.route('/')
def annotate_index():
    return render_template('annotation.html')

@annotate_bp.route('/templates/<path:filename>')
def annotate_templates(filename):
    if getattr(sys, 'frozen', False):
        templates_dir = os.path.join(sys._MEIPASS, 'templates')
    else:
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    return send_file(os.path.join(templates_dir, filename))

@annotate_bp.route('/api/select_folder', methods=['POST'])
def annotate_select_folder():
    global annotate_data_folder
    dialog_event.clear()
    dialog_queue.put({'type': 'folder'})
    dialog_event.wait()
    folder = dialog_result.get('path', '')
    if not folder:
        return jsonify({"error": "canceled"})
    forward_dir = os.path.join(folder, "forward")
    backward_dir = os.path.join(folder, "backward")
    has_forward = os.path.isdir(forward_dir)
    has_backward = os.path.isdir(backward_dir)
    if not has_forward and not has_backward:
        return jsonify({"error": f"缺少 forward 和 backward 子目录\n当前路径: {folder}"})
    if not has_forward:
        return jsonify({"error": f"缺少 forward 子目录（存放前摄像头图片）\n当前路径: {folder}"})
    if not has_backward:
        return jsonify({"error": f"缺少 backward 子目录（存放后摄像头图片）\n当前路径: {folder}"})
    front_imgs = sorted(glob.glob(os.path.join(forward_dir, "front_*.jpg")))
    back_imgs = sorted(glob.glob(os.path.join(backward_dir, "back_*.jpg")))
    if len(front_imgs) == 0:
        return jsonify({"error": "forward 目录中没有找到 front_*.jpg 图片"})
    if len(back_imgs) == 0:
        return jsonify({"error": "backward 目录中没有找到 back_*.jpg 图片"})
    annotate_data_folder = folder
    matched_count = annotate_get_frame_list()
    annotate_get_annotations_dir()
    if matched_count == 0:
        return jsonify({"error": f"前后视角没有匹配的图片！\nforward: {len(front_imgs)} 张\nbackward: {len(back_imgs)} 张\n请检查文件名编号是否一致"})
    warning = ""
    if len(front_imgs) != matched_count or len(back_imgs) != matched_count:
        warning = f"注意：forward {len(front_imgs)} 张，backward {len(back_imgs)} 张，匹配成功 {matched_count} 张"
    return jsonify({"path": folder, "total_frames": matched_count, "forward_dir": forward_dir, "backward_dir": backward_dir, "warning": warning})

@annotate_bp.route('/api/frame_image/<int:index>/<view>')
def annotate_frame_image(index, view):
    if view == 'front':
        files = annotate_front_files
    elif view == 'back':
        files = annotate_back_files
    else:
        return jsonify({"error": "invalid view"}), 400
    if index < 0 or index >= len(files):
        return jsonify({"error": "index out of range"}), 400
    return send_file(files[index])

@annotate_bp.route('/api/frame_info/<int:index>')
def annotate_frame_info(index):
    if index < 0 or index >= len(annotate_front_files):
        return jsonify({"error": "index out of range"}), 400
    return jsonify({"index": index, "total": len(annotate_front_files), "front_filename": os.path.basename(annotate_front_files[index]), "back_filename": os.path.basename(annotate_back_files[index])})

@annotate_bp.route('/api/save_annotation', methods=['POST'])
def annotate_save():
    data = request.json
    index = data.get('index')
    annotation = data.get('annotation')
    if index is None:
        return jsonify({"error": "missing index"}), 400
    global annotate_annotations_dir
    if not os.path.isdir(annotate_annotations_dir):
        os.makedirs(annotate_annotations_dir, exist_ok=True)
    front_name = os.path.basename(annotate_front_files[index])
    json_name = front_name.replace('.jpg', '.json')
    json_path = os.path.join(annotate_annotations_dir, json_name)
    if annotation is None or (isinstance(annotation, dict) and not annotation.get('annotations')):
        if os.path.exists(json_path):
            os.remove(json_path)
        return jsonify({"success": True, "deleted": True})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "path": json_path})

@annotate_bp.route('/api/load_annotation/<int:index>')
def annotate_load(index):
    if index < 0 or index >= len(annotate_front_files):
        return jsonify({"error": "index out of range"}), 400
    front_name = os.path.basename(annotate_front_files[index])
    json_name = front_name.replace('.jpg', '.json')
    json_path = os.path.join(annotate_annotations_dir, json_name)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        return jsonify({"annotation": annotation, "exists": True})
    else:
        return jsonify({"annotation": None, "exists": False})

@annotate_bp.route('/api/batch_status')
def annotate_batch_status():
    statuses = []
    for i in range(len(annotate_front_files)):
        front_name = os.path.basename(annotate_front_files[i])
        json_name = front_name.replace('.jpg', '.json')
        json_path = os.path.join(annotate_annotations_dir, json_name)
        statuses.append(os.path.exists(json_path))
    return jsonify({"statuses": statuses})

app.register_blueprint(extract_bp)
app.register_blueprint(annotate_bp)

def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

def open_browser():
    webbrowser.open('http://127.0.0.1:5000')

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
            if req['type'] == 'video':
                filepath = filedialog.askopenfilename(title="选择视频", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")], parent=top)
                dialog_result['filepath'] = filepath
            elif req['type'] == 'dir':
                d = filedialog.askdirectory(title="选择输出目录", parent=top)
                dialog_result['path'] = d
            elif req['type'] == 'folder':
                d = filedialog.askdirectory(title="选择数据文件夹（需包含 forward 和 backward 子目录）", parent=top)
                dialog_result['path'] = d
            top.destroy()
            dialog_event.set()
    except queue.Empty:
        pass
    root.after(100, lambda: process_dialogs(root))

if __name__ == '__main__':
    try:
        req = urllib.request.Request('http://127.0.0.1:5000/', method='GET')
        with urllib.request.urlopen(req, timeout=1) as response:
            if response.status == 200:
                print("检测到程序已在运行，已唤起已有窗口。")
                os._exit(0)
    except Exception:
        pass

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    root = tk.Tk()
    root.title("课堂行为分析工具 - 控制台")
    root.geometry("380x180")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))

    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'{w}x{h}+{x}+{y}')

    tk.Label(root, text="课堂行为分析工具", font=("微软雅黑", 14, "bold"), fg="green").pack(pady=(20, 5))
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
