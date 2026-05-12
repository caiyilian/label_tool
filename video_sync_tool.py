import os
import cv2
import numpy as np
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import glob
import subprocess
import re

class ZoomPanCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.orig_img = None
        self.img_tk = None
        self.scale = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.img_x = 0
        self.img_y = 0

        # 绑定事件
        self.bind("<MouseWheel>", self.on_zoom_windows)
        self.bind("<Button-4>", self.on_zoom_linux_up)
        self.bind("<Button-5>", self.on_zoom_linux_down)
        self.bind("<ButtonPress-1>", self.on_pan_start)
        self.bind("<B1-Motion>", self.on_pan_drag)
        self.bind("<Configure>", self.on_resize)

    def set_image(self, pil_image):
        self.orig_img = pil_image
        self.scale = 1.0
        self.img_x = 0
        self.img_y = 0
        self.update_view()

    def _zoom(self, factor, mx, my):
        if not self.orig_img: return
        
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return

        img_w, img_h = self.orig_img.size
        fit_scale = min(canvas_w / img_w, canvas_h / img_h)
        
        old_w = img_w * fit_scale * self.scale
        old_h = img_h * fit_scale * self.scale
        
        # 记录鼠标在当前图像上的相对位置比例
        rx = (mx - self.img_x) / old_w if old_w > 0 else 0
        ry = (my - self.img_y) / old_h if old_h > 0 else 0
        
        self.scale *= factor
        # 限制缩放比例 (1倍 到 20倍)
        self.scale = max(1.0, min(self.scale, 20.0))
        
        new_w = img_w * fit_scale * self.scale
        new_h = img_h * fit_scale * self.scale
        
        # 调整图像的 x, y，确保鼠标所在位置相对不变，实现以鼠标为中心的缩放
        self.img_x = mx - (rx * new_w)
        self.img_y = my - (ry * new_h)
        
        self.update_view()

    def on_zoom_windows(self, event):
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom(factor, event.x, event.y)

    def on_zoom_linux_up(self, event):
        self._zoom(1.2, event.x, event.y)

    def on_zoom_linux_down(self, event):
        self._zoom(1 / 1.2, event.x, event.y)

    def on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_drag(self, event):
        if not self.orig_img or self.scale <= 1.0: return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.img_x += dx
        self.img_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_view()

    def on_resize(self, event):
        self.update_view()

    def update_view(self):
        if not self.orig_img: return
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return

        img_w, img_h = self.orig_img.size
        fit_scale = min(canvas_w / img_w, canvas_h / img_h)
        S = fit_scale * self.scale

        current_w = img_w * S
        current_h = img_h * S

        # 边界约束，确保图片不能被拖拽出可视范围外
        if current_w <= canvas_w:
            self.img_x = (canvas_w - current_w) / 2
        else:
            self.img_x = max(min(self.img_x, 0), canvas_w - current_w)

        if current_h <= canvas_h:
            self.img_y = (canvas_h - current_h) / 2
        else:
            self.img_y = max(min(self.img_y, 0), canvas_h - current_h)

        # 优化：只截取可见区域进行缩放，防止放大倍数过大时内存崩溃或卡顿
        crop_x0 = -self.img_x / S
        crop_y0 = -self.img_y / S
        crop_x1 = (canvas_w - self.img_x) / S
        crop_y1 = (canvas_h - self.img_y) / S

        actual_crop_x0 = int(max(0, crop_x0))
        actual_crop_y0 = int(max(0, crop_y0))
        actual_crop_x1 = int(min(img_w, crop_x1))
        actual_crop_y1 = int(min(img_h, crop_y1))

        if actual_crop_x1 <= actual_crop_x0 or actual_crop_y1 <= actual_crop_y0:
            return

        cropped = self.orig_img.crop((actual_crop_x0, actual_crop_y0, actual_crop_x1, actual_crop_y1))
        
        display_w = int((actual_crop_x1 - actual_crop_x0) * S)
        display_h = int((actual_crop_y1 - actual_crop_y0) * S)
        
        if display_w <= 0 or display_h <= 0:
            return
            
        resized = cropped.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized)
        
        self.delete("all")
        draw_x = max(0, self.img_x)
        draw_y = max(0, self.img_y)
        self.create_image(draw_x, draw_y, anchor=tk.NW, image=self.img_tk)


class VideoSyncTool:
    def __init__(self, root):
        self.root = root
        self.root.title("多视角视频时间同步与抽帧工具 (支持鼠标滚轮缩放拖拽)")
        self.root.geometry("900x700")
        
        self.video1_path = tk.StringVar()
        self.video2_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        self.h1_var = tk.StringVar(value="15")
        self.m1_var = tk.StringVar(value="00")
        self.s1_var = tk.StringVar(value="00")
        self.h2_var = tk.StringVar(value="15")
        self.m2_var = tk.StringVar(value="00")
        self.s2_var = tk.StringVar(value="00")
        self.hint_var = tk.StringVar(value="")
        
        self.is_processing = False
        self.stop_requested = False
        self.current_process = None
        
        self.setup_ui()
        self.update_ui_states()
        
    def create_time_input(self, parent, row, col, h_var, m_var, s_var):
        f = ttk.Frame(parent)
        f.grid(row=row, column=col, padx=5, pady=5, sticky=tk.W)
        hours = [f"{i:02d}" for i in range(24)]
        mins = [f"{i:02d}" for i in range(60)]
        ttk.Spinbox(f, values=hours, textvariable=h_var, width=3, state="readonly", wrap=True).pack(side=tk.LEFT)
        ttk.Label(f, text=":").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(f, values=mins, textvariable=m_var, width=3, state="readonly", wrap=True).pack(side=tk.LEFT)
        ttk.Label(f, text=":").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(f, values=mins, textvariable=s_var, width=3, state="readonly", wrap=True).pack(side=tk.LEFT)

    def setup_ui(self):
        # 提示区域
        frame_hint = ttk.Frame(self.root, padding=(10, 5))
        frame_hint.pack(fill=tk.X)
        hint_label = tk.Label(frame_hint, textvariable=self.hint_var, 
                              fg="blue", font=("", 10, "bold"), anchor="w", justify=tk.LEFT)
        hint_label.pack(fill=tk.X)

        # 顶部：选择视频区域
        frame_top = ttk.LabelFrame(self.root, text="步骤 1：选择视频并对齐时间", padding=10)
        frame_top.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_v1 = tk.Button(frame_top, text="选择视频 1 (前视角)", command=lambda: self.load_video(1), cursor="hand2")
        self.btn_v1.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame_top, textvariable=self.video1_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame_top, text="第一帧时间 (时:分:秒):").grid(row=0, column=2, padx=5, pady=5)
        self.create_time_input(frame_top, 0, 3, self.h1_var, self.m1_var, self.s1_var)
        
        self.btn_v2 = tk.Button(frame_top, text="选择视频 2 (后视角)", command=lambda: self.load_video(2), cursor="hand2")
        self.btn_v2.grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(frame_top, textvariable=self.video2_path, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame_top, text="第一帧时间 (时:分:秒):").grid(row=1, column=2, padx=5, pady=5)
        self.create_time_input(frame_top, 1, 3, self.h2_var, self.m2_var, self.s2_var)
        
        # 中间：图像预览区域 (替换为自定义画布)
        frame_mid = ttk.Frame(self.root, padding=10)
        frame_mid.pack(fill=tk.BOTH, expand=True)
        
        self.img1_canvas = ZoomPanCanvas(frame_mid, bg="gray")
        self.img1_canvas.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH)
        
        self.img2_canvas = ZoomPanCanvas(frame_mid, bg="gray")
        self.img2_canvas.pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.BOTH)
        
        # 底部：输出与控制区域
        frame_bottom = ttk.LabelFrame(self.root, text="步骤 2 & 3：选择输出与执行", padding=10)
        frame_bottom.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_select_out = tk.Button(frame_bottom, text="选择输出目录", command=self.select_output_dir, state=tk.DISABLED, cursor="hand2")
        self.btn_select_out.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame_bottom, textvariable=self.output_dir, width=60).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_bottom, text="输出分辨率限制:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.resolutions = {
            "原分辨率 (Original)": None,
            "1080p (最大 1920x1080)": (1920, 1080),
            "720p (最大 1280x720)": (1280, 720),
            "480p (最大 854x480)": (854, 480)
        }
        self.resolution_var = tk.StringVar(value="原分辨率 (Original)")
        self.res_cb = ttk.Combobox(frame_bottom, textvariable=self.resolution_var, values=["请先加载视频"], state="disabled", width=30)
        self.res_cb.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 按钮容器
        frame_btns = ttk.Frame(frame_bottom)
        frame_btns.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.start_btn = tk.Button(frame_btns, text="▶ 开始同步抽帧", command=self.start_processing, state=tk.DISABLED, cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(frame_btns, text="⏹ 停止", command=self.stop_processing, state=tk.DISABLED, cursor="hand2")
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # 视频分辨率记录
        self.video_resolutions = {1: None, 2: None}
        
        # 进度条区域
        frame_progress = ttk.Frame(frame_bottom)
        frame_progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(frame_progress, text="视频 1 进度:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.progress1 = ttk.Progressbar(frame_progress, orient=tk.HORIZONTAL, mode='determinate')
        self.progress1.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.eta_label1 = ttk.Label(frame_progress, text="剩余时间: --:--", width=15)
        self.eta_label1.grid(row=0, column=2, padx=5, sticky=tk.E)
        frame_progress.columnconfigure(1, weight=1)
        
        ttk.Label(frame_progress, text="视频 2 进度:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.progress2 = ttk.Progressbar(frame_progress, orient=tk.HORIZONTAL, mode='determinate')
        self.progress2.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.eta_label2 = ttk.Label(frame_progress, text="剩余时间: --:--", width=15)
        self.eta_label2.grid(row=1, column=2, padx=5, sticky=tk.E)
        
        # 日志输出区域
        self.log_text = tk.Text(self.root, height=10, state=tk.NORMAL)
        self.log_text.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        self.log("工具已启动。支持鼠标放在画面上滚轮缩放，左键按住拖拽。")
        
    def check_chinese_filename(self, filepath):
        basename = os.path.basename(filepath)
        for char in basename:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
        
    def set_btn_accent(self, btn, is_accent):
        if is_accent:
            btn.config(bg="#4CAF50", fg="white", font=("", 9, "bold"), relief=tk.RAISED)
        else:
            btn.config(bg="#f0f0f0", fg="black", font=("", 9, ""), relief=tk.GROOVE)

    def update_ui_states(self):
        v1_loaded = bool(self.video1_path.get())
        v2_loaded = bool(self.video2_path.get())
        out_loaded = bool(self.output_dir.get())

        # 提示显示
        if v1_loaded or v2_loaded:
            self.hint_var.set("💡 提示：将鼠标放在下方视频画面上，滚动滚轮可放大图片，按住左键可拖拽。放大后能更清晰地看准监控时间！")
        else:
            self.hint_var.set("")

        # 步骤 1 按钮
        self.set_btn_accent(self.btn_v1, not v1_loaded)
        self.set_btn_accent(self.btn_v2, not v2_loaded)

        # 步骤 2 按钮
        if v1_loaded and v2_loaded:
            self.btn_select_out.config(state=tk.NORMAL)
            self.set_btn_accent(self.btn_select_out, not out_loaded)
        else:
            self.btn_select_out.config(state=tk.DISABLED)
            self.set_btn_accent(self.btn_select_out, False)

        # 步骤 3 按钮
        if v1_loaded and v2_loaded and out_loaded:
            self.start_btn.config(state=tk.NORMAL)
            self.set_btn_accent(self.start_btn, True)
        else:
            self.start_btn.config(state=tk.DISABLED)
            self.set_btn_accent(self.start_btn, False)

    def load_video(self, video_num):
        filepath = filedialog.askopenfilename(title=f"选择视频 {video_num}", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not filepath:
            return
            
        if self.check_chinese_filename(filepath):
            messagebox.showerror("错误", f"视频文件名不能包含中文，请重命名后重试！\n当前文件名：{os.path.basename(filepath)}")
            return
            
        if video_num == 1:
            self.video1_path.set(filepath)
            self.show_first_frame(filepath, 1)
        else:
            self.video2_path.set(filepath)
            self.show_first_frame(filepath, 2)
            
        self.update_ui_states()
            
    def show_first_frame(self, filepath, video_num):
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(filepath))
            cap = cv2.VideoCapture(os.path.basename(filepath))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame) # 保持原图尺寸
                
                # 记录视频分辨率
                height, width, _ = frame.shape
                self.video_resolutions[video_num] = (width, height)
                self.update_resolution_options()
                
                if video_num == 1:
                    self.img1_canvas.set_image(img)
                else:
                    self.img2_canvas.set_image(img)
            else:
                self.log(f"无法读取视频第一帧: {filepath}")
        finally:
            os.chdir(cwd)
            
    def update_resolution_options(self):
        v1_res = self.video_resolutions[1]
        v2_res = self.video_resolutions[2]
        
        # 只有当至少加载了一个视频时，才更新选项
        if not v1_res and not v2_res:
            return
            
        # 找出当前已加载视频中的最小宽度（作为木桶效应的短板）
        min_width = float('inf')
        if v1_res: min_width = min(min_width, v1_res[0])
        if v2_res: min_width = min(min_width, v2_res[0])
        
        available_options = ["原分辨率 (Original)"]
        
        # 根据最小宽度，动态过滤出不超过原视频大小的选项
        if min_width > 1920:
            available_options.append("1080p (最大 1920x1080)")
        if min_width > 1280:
            available_options.append("720p (最大 1280x720)")
        if min_width > 854:
            available_options.append("480p (最大 854x480)")
            
        self.res_cb.config(values=available_options, state="readonly")
        
        # 如果当前选中的值不再列表中，重置为"原分辨率"
        current_val = self.resolution_var.get()
        if current_val not in available_options:
            self.resolution_var.set("原分辨率 (Original)")
            
    def select_output_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_dir.set(d)
            self.update_ui_states()
            
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def stop_processing(self):
        if self.is_processing:
            self.stop_requested = True
            self.log("⚠️ 用户请求停止，正在终止进程...")
            self.stop_btn.config(state=tk.DISABLED, text="停止中...")
            if self.current_process:
                try:
                    self.current_process.terminate()
                except Exception as e:
                    self.log(f"终止进程失败: {e}")

    def start_processing(self):
        v1 = self.video1_path.get()
        v2 = self.video2_path.get()
        out = self.output_dir.get()
        
        if not v1 or not v2 or not out:
            messagebox.showwarning("警告", "请确保已选择两个视频和输出目录！")
            return
            
        try:
            t1_sec = int(self.h1_var.get()) * 3600 + int(self.m1_var.get()) * 60 + int(self.s1_var.get())
            t2_sec = int(self.h2_var.get()) * 3600 + int(self.m2_var.get()) * 60 + int(self.s2_var.get())
        except ValueError:
            messagebox.showwarning("警告", "时间输入有误，请确保时分秒框内均为有效数字！")
            return
            
        res_key = self.resolution_var.get()
        res_target = self.resolutions.get(res_key)
            
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL, text="⏹ 停止")
        self.btn_select_out.config(state=tk.DISABLED)
        self.res_cb.config(state=tk.DISABLED)
        
        self.is_processing = True
        self.stop_requested = False
        
        self.progress1['value'] = 0
        self.progress2['value'] = 0
        self.eta_label1.config(text="剩余时间: 计算中...")
        self.eta_label2.config(text="剩余时间: --:--")
        threading.Thread(target=self.process_videos, args=(v1, v2, out, t1_sec, t2_sec, res_target), daemon=True).start()
        
    def update_progress(self, progress_bar, eta_label, value, maximum, eta_str):
        progress_bar['maximum'] = maximum
        progress_bar['value'] = value
        if eta_label:
            eta_label.config(text=f"剩余时间: {eta_str}")

    def process_videos(self, v1, v2, out, t1_sec, t2_sec, res_target=None):
        try:
            forward_dir = os.path.join(out, "forward")
            backward_dir = os.path.join(out, "backward")
            
            os.makedirs(forward_dir, exist_ok=True)
            os.makedirs(backward_dir, exist_ok=True)
            
            diff = t1_sec - t2_sec
            skip_sec1 = max(0, -diff)
            skip_sec2 = max(0, diff)
            
            self.log(f"视频 1 时间: {t1_sec} 秒, 视频 2 时间: {t2_sec} 秒")
            self.log(f"视频 1 开头需要跳过 {skip_sec1} 秒")
            self.log(f"视频 2 开头需要跳过 {skip_sec2} 秒")
            
            def calculate_skip_frames(skip_seconds, fps_round):
                # 统一加上 0.5 秒的相位偏移 (fps_round // 2)，保证在每一秒的正中间采样
                return int(skip_seconds * fps_round + fps_round // 2)

            def format_time(seconds):
                if seconds < 0: return "00:00"
                m, s = divmod(int(seconds), 60)
                h, m = divmod(m, 60)
                if h > 0:
                    return f"{h:02d}:{m:02d}:{s:02d}"
                return f"{m:02d}:{s:02d}"

            def extract_frames(video_path, output_folder, skip_seconds, prefix, progress_bar, eta_label, target_resolution):
                # 依然需要先用 cv2 获取一下视频总长度和 fps，用于进度条计算
                cap = cv2.VideoCapture(video_path)
                total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                
                # 针对损坏监控视频的启发式时长估算
                file_size_bytes = os.path.getsize(video_path)
                cv2_duration = total_frames_in_video / fps
                cv2_bitrate = (file_size_bytes * 8) / cv2_duration if cv2_duration > 0 else 0
                
                # 如果算出来的码率低得离谱（比如码率小于 500kbps，实际上只有 44kbps）
                # 几乎可以断定视频的元数据时间戳已损坏，总帧数是假的。
                is_corrupted = cv2_bitrate < 500000
                if is_corrupted:
                    # 动态估算真实码率：根据分辨率和帧率计算 (监控视频经验值：每像素每帧约 0.022 bits)
                    # 比如 2560x1440@25fps 算出来约 2 Mbps，1920x1080@25fps 算出来约 1.14 Mbps
                    estimated_bitrate = width * height * fps * 0.022
                    if estimated_bitrate <= 0:
                        estimated_bitrate = 2000000 # 兜底 2Mbps
                        
                    estimated_real_duration = (file_size_bytes * 8) / estimated_bitrate
                    total_frames_in_video = estimated_real_duration * fps
                    
                fps_round = round(fps)
                skip_frames = calculate_skip_frames(skip_seconds, fps_round)
                
                # 预期总提取的图片数量
                expected_total_images = int((total_frames_in_video - skip_frames) / fps_round)
                if expected_total_images <= 0: expected_total_images = 1
                
                if is_corrupted:
                    self.log(f"{prefix} 检测到视频时间戳损坏(标称{cv2_duration/3600:.1f}小时)。使用文件大小估算真实时长为 {estimated_real_duration/60:.1f} 分钟，预计提取 {expected_total_images} 帧。")
                
                self.log(f"正在处理 {prefix}... 帧率约 {fps_round} FPS, 需跳过前 {skip_frames} 帧 (包含 0.5 秒相位偏移)")
                
                # 动态计算分辨率缩放
                out_w, out_h = int(width), int(height)
                if target_resolution:
                    max_w, max_h = target_resolution
                    if out_w > max_w or out_h > max_h:
                        scale_factor = min(max_w / out_w, max_h / out_h)
                        out_w = int(out_w * scale_factor)
                        # 为了兼容某些编码器或处理逻辑，确保高度是偶数（通常是好的实践）
                        out_h = int(out_h * scale_factor)
                        out_h = out_h if out_h % 2 == 0 else out_h - 1
                        out_w = out_w if out_w % 2 == 0 else out_w - 1
                
                start_time = time.time()
                
                # 构建 ffmpeg 命令
                # 弃用 -ss 和 fps=1，改用精确的 select 滤镜，完美实现帧级同步，避开时间戳跳变
                out_pattern = os.path.join(output_folder, f"{prefix}_%06d.jpg")
                filter_str = f"select='gte(n,{skip_frames})*not(mod(n-{skip_frames},{fps_round}))'"
                
                if out_w != int(width) or out_h != int(height):
                    filter_str += f",scale={out_w}:{out_h}"
                    self.log(f"-> 图像将等比例缩放至 {out_w}x{out_h} (原分辨率 {int(width)}x{int(height)})")
                    
                cmd = [
                    "ffmpeg",
                    "-y", # 覆盖输出
                    "-i", video_path,
                    "-vf", filter_str,
                    "-vsync", "0", # 避免由于时间戳问题导致 ffmpeg 停止处理
                    "-q:v", "2", # 保证提取出的 jpg 质量较高
                    out_pattern
                ]
                
                # 使用 subprocess 执行，并捕获 stderr (ffmpeg 的进度输出在 stderr)
                process = subprocess.Popen(
                    cmd, 
                    stderr=subprocess.PIPE, 
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                self.current_process = process
                
                # 解析 ffmpeg 输出中的 "frame=  123"
                frame_regex = re.compile(r"frame=\s*(\d+)")
                
                saved_count = 0
                for line in process.stderr:
                    if self.stop_requested:
                        process.terminate()
                        break
                        
                    # 打印 stderr 看看 ffmpeg 为什么退出
                    # print("FFmpeg:", line.strip())
                    match = frame_regex.search(line)
                    if match:
                        saved_count = int(match.group(1))
                        if saved_count % 100 == 0:
                            self.log(f"{prefix} 已由 FFmpeg 提取 {saved_count} 帧...")
                            
                        elapsed = time.time() - start_time
                        if saved_count > 0:
                            eta_seconds = (elapsed / saved_count) * (expected_total_images - saved_count)
                            eta_str = format_time(eta_seconds)
                        else:
                            eta_str = "计算中..."
                            
                        # 更新进度条
                        # 我们用提取的图片数量作为进度条的值
                        self.root.after(0, self.update_progress, progress_bar, eta_label, saved_count, expected_total_images, eta_str)
                
                process.wait()
                if self.stop_requested:
                    self.log(f"🚫 {prefix} 提取已被用户终止。")
                    return saved_count
                    
                if process.returncode != 0:
                     self.log(f"{prefix} FFmpeg 异常退出, code {process.returncode}。可能由于视频损坏或参数不兼容。")
                
                self.root.after(0, self.update_progress, progress_bar, eta_label, expected_total_images, expected_total_images, "完成")
                self.log(f"{prefix} FFmpeg 提取完成，共提取 {saved_count} 帧。")
                return saved_count
                
            # 串行提取以避免过高磁盘IO和内存占用，同时进度条显示更清晰
            count1 = extract_frames(v1, forward_dir, skip_sec1, "front", self.progress1, self.eta_label1, res_target)
            if not self.stop_requested:
                self.root.after(0, lambda: self.eta_label2.config(text="剩余时间: 计算中..."))
                count2 = extract_frames(v2, backward_dir, skip_sec2, "back", self.progress2, self.eta_label2, res_target)
            else:
                count2 = 0
            
            if not self.stop_requested:
                self.log(f"视频 1 最终提取数量: {count1}, 视频 2 最终提取数量: {count2}")
                min_count = min(count1, count2)
                self.log(f"将以较短的长度 ({min_count}) 为基准裁剪结尾。")
                
                def trim_tails(folder, prefix, keep_count):
                    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.jpg")))
                    deleted = 0
                    for f in files[keep_count:]:
                        os.remove(f)
                        deleted += 1
                    if deleted > 0:
                        self.log(f"清理了 {folder} 中结尾多出的 {deleted} 帧。")
                        
                trim_tails(forward_dir, "front", min_count)
                trim_tails(backward_dir, "back", min_count)
                
                self.log("所有处理完成！两个文件夹时间已完全同步且数量相等。")
                messagebox.showinfo("完成", "时间同步与抽帧已完成！")
            else:
                self.log("🚫 处理已全部停止。输出文件夹可能包含不完整或未对齐的数据。")
                
        except Exception as e:
            self.log(f"发生错误: {str(e)}")
            messagebox.showerror("错误", f"处理过程中发生异常:\n{str(e)}")
        finally:
            self.is_processing = False
            self.stop_requested = False
            self.current_process = None
            self.stop_btn.config(state=tk.DISABLED, text="⏹ 停止")
            self.res_cb.config(state="readonly")
            self.root.after(0, self.update_ui_states)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSyncTool(root)
    root.mainloop()
