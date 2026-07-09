import os
import sys

# 解决 Tcl/Tk 版本冲突问题：强制指定使用我们自己打包的正确库
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
from flask import Flask, render_template, request, jsonify

# 关闭 flask 的默认请求日志输出
log_flask = logging.getLogger('werkzeug')
log_flask.disabled = True

# 适配 PyInstaller 打包后的 templates 目录路径
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

# 全局队列和事件，用于跨线程调度 Tkinter 对话框
dialog_queue = queue.Queue()
dialog_result = {}
dialog_event = threading.Event()

state = {
    "is_processing": False,
    "stop_requested": False,
    "current_process": None,
    "progress1": {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"},
    "progress2": {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"},
    "logs": ["系统初始化完毕", "等待配置视频源..."]
}

def log(msg):
    time_str = time.strftime("%H:%M:%S")
    formatted = f"[{time_str}] {msg}"
    print(formatted)
    state["logs"].append(formatted)
    if len(state["logs"]) > 200:
        state["logs"].pop(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/select_video', methods=['POST'])
def select_video():
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
            # 缩小图片以提升传输速度（前端只用于看时间码）
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

    return jsonify({
        "path": filepath,
        "name": basename,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
        "first_frame": first_frame_b64
    })

@app.route('/api/select_dir', methods=['POST'])
def select_dir():
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

        forward_dir = os.path.join(out, "forward")
        backward_dir = os.path.join(out, "backward")
        os.makedirs(forward_dir, exist_ok=True)
        os.makedirs(backward_dir, exist_ok=True)
        
        diff = t1_sec - t2_sec
        skip_sec1 = max(0, -diff)
        skip_sec2 = max(0, diff)
        
        log(f"配置读取: V1={t1_sec}s, V2={t2_sec}s")
        log(f"解析指令: V1跳过 {skip_sec1}s, V2跳过 {skip_sec2}s")

        def extract(video_path, output_folder, skip_seconds, prefix, prog_key):
            state[prog_key]["status"] = "处理中"
            
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
            
            expected_total_images = int((total_frames_in_video - skip_frames) / fps_round)
            if expected_total_images <= 0: expected_total_images = 1
            
            state[prog_key]["total"] = expected_total_images
            
            if is_corrupted:
                log(f"{prefix} 检测到时间戳异常。智能预估真实时长 {estimated_real_duration/60:.1f}m，提取 {expected_total_images}帧")
            
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
            filter_str = f"select='gte(n,{skip_frames})*not(mod(n-{skip_frames},{fps_round}))'"
            
            if out_w != int(width) or out_h != int(height):
                filter_str += f",scale={out_w}:{out_h}"
                
            # 动态获取可执行文件所在目录（兼容 PyInstaller 打包环境）
            if getattr(sys, 'frozen', False):
                app_dir = os.path.dirname(sys.executable)
            else:
                app_dir = os.path.dirname(os.path.abspath(__file__))
            
            ffmpeg_path = os.path.join(app_dir, "ffmpeg.exe")
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = "ffmpeg" # 降级为系统环境变量
                
            cmd = [
                ffmpeg_path, "-y", "-i", video_path, "-vf", filter_str,
                "-vsync", "0", "-q:v", "2", out_pattern
            ]
            
            # 配置 subprocess 以隐藏 ffmpeg 的 CMD 黑框 (仅在 Windows 下生效)
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
            
            process = subprocess.Popen(
                cmd, stderr=subprocess.PIPE, universal_newlines=True,
                encoding='utf-8', errors='ignore',
                creationflags=creationflags
            )
            state["current_process"] = process
            
            frame_regex = re.compile(r"frame=\s*(\d+)")
            saved_count = 0
            
            for line in process.stderr:
                if state["stop_requested"]:
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
                    state[prog_key]["current"] = saved_count
                    state[prog_key]["percent"] = percent
                    state[prog_key]["eta"] = eta_str
                    
            process.wait()
            if state["stop_requested"]:
                log(f"🚫 {prefix} 提取已被强行终止")
                state[prog_key]["status"] = "已终止"
                return saved_count
                
            # 检查 ffmpeg 是否是正常退出（防止进程崩溃或被外部强杀导致假满进度）
            if process.returncode != 0:
                log(f"❌ {prefix} ffmpeg 异常退出 (code: {process.returncode})，提取未完成。")
                state[prog_key]["status"] = "异常退出"
                state[prog_key]["eta"] = "失败"
                raise Exception(f"{prefix} ffmpeg 处理崩溃或被强杀")
                
            state[prog_key]["current"] = expected_total_images
            state[prog_key]["percent"] = 100
            state[prog_key]["eta"] = "完成"
            state[prog_key]["status"] = "完成"
            log(f"✅ {prefix} 提取完成，共落盘 {saved_count} 帧")
            return saved_count

        count1 = extract(v1, forward_dir, skip_sec1, "front", "progress1")
        if not state["stop_requested"]:
            state["progress2"]["status"] = "准备中"
            count2 = extract(v2, backward_dir, skip_sec2, "back", "progress2")
        else:
            count2 = 0
            
        if not state["stop_requested"]:
            min_count = min(count1, count2)
            log(f"对齐校验: 以较短长度 ({min_count}) 为基准裁剪冗余尾帧")
            def trim_tails(folder, prefix, keep_count):
                files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.jpg")))
                for f in files[keep_count:]:
                    os.remove(f)
            trim_tails(forward_dir, "front", min_count)
            trim_tails(backward_dir, "back", min_count)
            log("🎉 任务圆满结束！双视角帧数已严格等量对齐")
        else:
            log("🚫 任务流程已终止，文件可能不完整")
            
    except Exception as e:
        log(f"❌ 发生异常: {str(e)}")
    finally:
        state["is_processing"] = False
        state["stop_requested"] = False
        state["current_process"] = None

@app.route('/api/start', methods=['POST'])
def start_mission():
    data = request.json
    if state["is_processing"]:
        return jsonify({"error": "已经在运行中"})
        
    state["is_processing"] = True
    state["stop_requested"] = False
    state["progress1"] = {"percent": 0, "current": 0, "total": 0, "eta": "计算中...", "status": "初始化"}
    state["progress2"] = {"percent": 0, "current": 0, "total": 0, "eta": "--:--", "status": "等待"}
    state["logs"] = []
    log("🚀 引擎点火，任务开始执行")
    
    threading.Thread(target=process_videos_thread, args=(data,), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/stop', methods=['POST'])
def stop_mission():
    if state["is_processing"]:
        state["stop_requested"] = True
        log("⚠️ 接收到用户终止指令，正在下发中断信号...")
        if state["current_process"]:
            try:
                state["current_process"].terminate()
            except:
                pass
    return jsonify({"success": True})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "is_processing": state["is_processing"],
        "progress1": state["progress1"],
        "progress2": state["progress2"],
        "logs": state["logs"]
    })

@app.route('/api/focus', methods=['POST'])
def focus_api():
    dialog_queue.put({'type': 'focus'})
    return jsonify({"app": "multi-sync", "status": "focused"})

def run_flask():
    # use_reloader=False 避免在多线程/打包环境下启动两次
    app.run(port=5000, debug=False, use_reloader=False)

def open_browser():
    webbrowser.open('http://127.0.0.1:5000')

def on_closing(root):
    # 强制退出整个进程
    os._exit(0)

def process_dialogs(root):
    try:
        req = dialog_queue.get_nowait()
        
        if req['type'] == 'focus':
            # 恢复窗口并强制置顶显示
            if root.state() == 'iconic':
                root.deiconify()
            root.attributes('-topmost', True)
            root.attributes('-topmost', False)
            root.lift()
            root.focus_force()
            open_browser()
        else:
            # 创建一个临时的顶级窗口用于挂载对话框，强制其置顶
            top = tk.Toplevel(root)
            top.attributes('-topmost', True)
            top.withdraw()  # 隐藏这个临时窗口
            
            if req['type'] == 'video':
                filepath = filedialog.askopenfilename(
                    title="选择视频",
                    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
                    parent=top
                )
                dialog_result['filepath'] = filepath
            elif req['type'] == 'dir':
                d = filedialog.askdirectory(title="选择输出目录", parent=top)
                dialog_result['path'] = d
                
            top.destroy()
            dialog_event.set()
    except queue.Empty:
        pass
    root.after(100, lambda: process_dialogs(root))

if __name__ == '__main__':
    # 0. 单实例检测：如果已经有实例在运行，则通知它置顶并退出当前实例
    try:
        req = urllib.request.Request('http://127.0.0.1:5000/api/focus', method='POST')
        with urllib.request.urlopen(req, timeout=1) as response:
            res_body = response.read().decode('utf-8')
            if 'multi-sync' in res_body:
                print("检测到程序已在运行，已唤起已有窗口。")
                os._exit(0)
    except Exception:
        pass

    # 1. 在后台线程中启动 Flask 服务
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 2. 在主线程中创建 Tkinter 控制台 GUI
    root = tk.Tk()
    root.title("多视角视频同步抽帧 - 控制台")
    root.geometry("380x180")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))

    # 居中显示窗口
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # 界面元素
    tk.Label(root, text="✅ 服务正在运行中", font=("微软雅黑", 14, "bold"), fg="green").pack(pady=(20, 5))
    tk.Label(root, text="请保持此窗口打开，关闭窗口即停止服务", font=("微软雅黑", 10), fg="#666666").pack(pady=(0, 20))

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    btn_open = tk.Button(btn_frame, text="🌐 打开网页界面", command=open_browser, width=15, font=("微软雅黑", 10), cursor="hand2")
    btn_open.pack(side=tk.LEFT, padx=10)

    btn_exit = tk.Button(btn_frame, text="⏹ 关闭并退出", command=lambda: on_closing(root), width=15, font=("微软雅黑", 10), cursor="hand2")
    btn_exit.pack(side=tk.LEFT, padx=10)

    # 启动轮询，处理来自 Flask 线程的对话框请求
    root.after(100, lambda: process_dialogs(root))

    # 启动时自动打开一次网页
    root.after(1000, open_browser)

    # 运行主循环
    root.mainloop()
