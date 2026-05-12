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

# ================= 样式常量 =================
COLOR_BG = "#F5F6F7"          # 浅灰背景
COLOR_CARD = "#FFFFFF"        # 白色卡片
COLOR_PRIMARY = "#2D5AF6"     # 科技蓝
COLOR_SUCCESS = "#28A745"     # 成功绿
COLOR_DANGER = "#DC3545"      # 警告红
COLOR_TEXT = "#333333"        # 主文字
COLOR_TEXT_LIGHT = "#666666"  # 次要文字
FONT_MAIN = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 12, "bold")

class ZoomPanCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, highlightthickness=1, highlightbackground="#D1D1D1", **kwargs)
        self.orig_img = None
        self.img_tk = None
        self.scale = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.img_x = 0
        self.img_y = 0

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
        rx = (mx - self.img_x) / old_w if old_w > 0 else 0
        ry = (my - self.img_y) / old_h if old_h > 0 else 0
        self.scale *= factor
        self.scale = max(1.0, min(self.scale, 20.0))
        new_w = img_w * fit_scale * self.scale
        new_h = img_h * fit_scale * self.scale
        self.img_x = mx - (rx * new_w)
        self.img_y = my - (ry * new_h)
        self.update_view()

    def on_zoom_windows(self, event):
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom(factor, event.x, event.y)

    def on_zoom_linux_up(self, event): self._zoom(1.2, event.x, event.y)
    def on_zoom_linux_down(self, event): self._zoom(1 / 1.2, event.x, event.y)
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

    def on_resize(self, event): self.update_view()

    def update_view(self):
        if not self.orig_img: return
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return
        img_w, img_h = self.orig_img.size
        fit_scale = min(canvas_w / img_w, canvas_h / img_h)
        S = fit_scale * self.scale
        current_w, current_h = img_w * S, img_h * S

        if current_w <= canvas_w: self.img_x = (canvas_w - current_w) / 2
        else: self.img_x = max(min(self.img_x, 0), canvas_w - current_w)
        if current_h <= canvas_h: self.img_y = (canvas_h - current_h) / 2
        else: self.img_y = max(min(self.img_y, 0), canvas_h - current_h)

        crop_x0, crop_y0 = -self.img_x / S, -self.img_y / S
        crop_x1, crop_y1 = (canvas_w - self.img_x) / S, (canvas_h - self.img_y) / S
        ax0, ay0 = int(max(0, crop_x0)), int(max(0, crop_y0))
        ax1, ay1 = int(min(img_w, crop_x1)), int(min(img_h, crop_y1))

        if ax1 <= ax0 or ay1 <= ay0: return
        cropped = self.orig_img.crop((ax0, ay0, ax1, ay1))
        dw, dh = int((ax1 - ax0) * S), int((ay1 - ay0) * S)
        if dw <= 0 or dh <= 0: return
        resized = cropped.resize((dw, dh), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(max(0, self.img_x), max(0, self.img_y), anchor=tk.NW, image=self.img_tk)

class VideoSyncTool:
    def __init__(self, root):
        self.root = root
        self.root.title("多视角视频同步抽帧工具")
        self.root.geometry("1100x850")
        self.root.configure(bg=COLOR_BG)
        
        self.video1_path = tk.StringVar()
        self.video2_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.h1_var, self.m1_var, self.s1_var = tk.StringVar(value="15"), tk.StringVar(value="00"), tk.StringVar(value="00")
        self.h2_var, self.m2_var, self.s2_var = tk.StringVar(value="15"), tk.StringVar(value="00"), tk.StringVar(value="00")
        self.hint_var = tk.StringVar(value="准备就绪")
        
        self.is_processing = False
        self.stop_requested = False
        self.current_process = None
        self.video_resolutions = {1: None, 2: None}

        self.apply_styles()
        self.setup_ui()
        self.update_ui_states()

    def apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 卡片容器样式
        style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        # 标题样式
        style.configure("Title.TLabel", background=COLOR_CARD, font=FONT_TITLE, foreground=COLOR_PRIMARY)
        # 普通文字
        style.configure("TLabel", background=COLOR_CARD, font=FONT_MAIN, foreground=COLOR_TEXT)
        # 进度条
        style.configure("TProgressbar", thickness=8, troughcolor=COLOR_BG, bordercolor=COLOR_BG, background=COLOR_PRIMARY)
        # 下拉框
        style.configure("TCombobox", padding=5)

    def setup_ui(self):
        # --- 顶部提示条 ---
        self.hint_label = tk.Label(self.root, textvariable=self.hint_var, bg="#E8F0FE", fg=COLOR_PRIMARY, 
                                   font=FONT_BOLD, pady=8)
        self.hint_label.pack(fill=tk.X)

        main_container = ttk.Frame(self.root, padding=20, style="Card.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- 步骤 1：输入区域 ---
        step1_frame = ttk.Frame(main_container, style="Card.TFrame")
        step1_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(step1_frame, text="Step 1: 视频路径与时间对齐", style="Title.TLabel").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0,10))

        # 视频 1
        self.btn_v1 = tk.Button(step1_frame, text="📁 选择视频 1 (Front)", command=lambda: self.load_video(1), 
                                relief="flat", bg=COLOR_PRIMARY, fg="white", font=FONT_BOLD, width=20, cursor="hand2")
        self.btn_v1.grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(step1_frame, textvariable=self.video1_path, width=45, foreground=COLOR_TEXT_LIGHT).grid(row=1, column=1, padx=10, sticky=tk.W)
        ttk.Label(step1_frame, text="对齐时间:").grid(row=1, column=2, padx=5)
        self.create_time_input(step1_frame, 1, 3, self.h1_var, self.m1_var, self.s1_var)

        # 视频 2
        self.btn_v2 = tk.Button(step1_frame, text="📁 选择视频 2 (Back)", command=lambda: self.load_video(2), 
                                relief="flat", bg=COLOR_PRIMARY, fg="white", font=FONT_BOLD, width=20, cursor="hand2")
        self.btn_v2.grid(row=2, column=0, padx=5, pady=5)
        ttk.Label(step1_frame, textvariable=self.video2_path, width=45, foreground=COLOR_TEXT_LIGHT).grid(row=2, column=1, padx=10, sticky=tk.W)
        ttk.Label(step1_frame, text="对齐时间:").grid(row=2, column=2, padx=5)
        self.create_time_input(step1_frame, 2, 3, self.h2_var, self.m2_var, self.s2_var)

        # --- 步骤 2：预览区域 ---
        preview_frame = ttk.Frame(main_container, style="Card.TFrame")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 使用更精致的 Canvas 容器
        c1_container = ttk.Frame(preview_frame, style="Card.TFrame")
        c1_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(c1_container, text="视频 1 第一帧预览", font=FONT_BOLD).pack(pady=2)
        self.img1_canvas = ZoomPanCanvas(c1_container, bg="#EBEDF0")
        self.img1_canvas.pack(fill=tk.BOTH, expand=True)

        c2_container = ttk.Frame(preview_frame, style="Card.TFrame")
        c2_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(c2_container, text="视频 2 第一帧预览", font=FONT_BOLD).pack(pady=2)
        self.img2_canvas = ZoomPanCanvas(c2_container, bg="#EBEDF0")
        self.img2_canvas.pack(fill=tk.BOTH, expand=True)

        # --- 步骤 3：输出设置 ---
        step3_frame = ttk.Frame(main_container, style="Card.TFrame")
        step3_frame.pack(fill=tk.X, pady=20)
        
        self.btn_select_out = tk.Button(step3_frame, text="⚙️ 设置输出目录", command=self.select_output_dir, 
                                        relief="flat", bg="#6C757D", fg="white", font=FONT_BOLD, width=20)
        self.btn_select_out.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(step3_frame, textvariable=self.output_dir, foreground=COLOR_TEXT_LIGHT).grid(row=0, column=1, padx=10, sticky=tk.W)
        
        ttk.Label(step3_frame, text="输出分辨率:").grid(row=0, column=2, padx=5)
        self.resolution_var = tk.StringVar(value="原分辨率 (Original)")
        self.res_cb = ttk.Combobox(step3_frame, textvariable=self.resolution_var, state="disabled", width=25)
        self.res_cb.grid(row=0, column=3, padx=5)

        # --- 控制与进度 ---
        ctrl_frame = ttk.Frame(main_container, style="Card.TFrame")
        ctrl_frame.pack(fill=tk.X)

        btn_box = ttk.Frame(ctrl_frame, style="Card.TFrame")
        btn_box.pack(pady=10)
        self.start_btn = tk.Button(btn_box, text="▶ 开始同步抽帧", command=self.start_processing, 
                                   bg=COLOR_SUCCESS, fg="white", font=FONT_TITLE, relief="flat", padx=30, pady=5, cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(btn_box, text="⏹ 停止处理", command=self.stop_processing, 
                                  bg=COLOR_DANGER, fg="white", font=FONT_TITLE, relief="flat", padx=30, pady=5, state=tk.DISABLED, cursor="hand2")
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # 进度条
        progress_box = ttk.Frame(ctrl_frame, style="Card.TFrame")
        progress_box.pack(fill=tk.X, pady=5)
        
        # P1
        p1_f = ttk.Frame(progress_box, style="Card.TFrame")
        p1_f.pack(fill=tk.X, pady=2)
        ttk.Label(p1_f, text="V1 Progress:", width=12).pack(side=tk.LEFT)
        self.progress1 = ttk.Progressbar(p1_f, orient=tk.HORIZONTAL, mode='determinate')
        self.progress1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.eta_label1 = ttk.Label(p1_f, text="--:--", width=10)
        self.eta_label1.pack(side=tk.RIGHT)

        # P2
        p2_f = ttk.Frame(progress_box, style="Card.TFrame")
        p2_f.pack(fill=tk.X, pady=2)
        ttk.Label(p2_f, text="V2 Progress:", width=12).pack(side=tk.LEFT)
        self.progress2 = ttk.Progressbar(p2_f, orient=tk.HORIZONTAL, mode='determinate')
        self.progress2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.eta_label2 = ttk.Label(p2_f, text="--:--", width=10)
        self.eta_label2.pack(side=tk.RIGHT)

        # 日志
        self.log_text = tk.Text(self.root, height=8, bg="#212529", fg="#F8F9FA", font=("Consolas", 9), 
                                relief="flat", padx=10, pady=10)
        self.log_text.pack(fill=tk.X, padx=20, pady=(0, 20))

    def create_time_input(self, parent, row, col, h_var, m_var, s_var):
        f = ttk.Frame(parent, style="Card.TFrame")
        f.grid(row=row, column=col, padx=5, pady=5, sticky=tk.W)
        opts = {"width": 4, "state": "readonly", "wrap": True, "justify": "center"}
        ttk.Spinbox(f, values=[f"{i:02d}" for i in range(24)], textvariable=h_var, **opts).pack(side=tk.LEFT)
        ttk.Label(f, text=":", font=FONT_BOLD).pack(side=tk.LEFT)
        ttk.Spinbox(f, values=[f"{i:02d}" for i in range(60)], textvariable=m_var, **opts).pack(side=tk.LEFT)
        ttk.Label(f, text=":", font=FONT_BOLD).pack(side=tk.LEFT)
        ttk.Spinbox(f, values=[f"{i:02d}" for i in range(60)], textvariable=s_var, **opts).pack(side=tk.LEFT)

    # ================= 业务逻辑保持不变 =================
    def log(self, msg):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_ui_states(self):
        v1_l, v2_l, out_l = bool(self.video1_path.get()), bool(self.video2_path.get()), bool(self.output_dir.get())
        
        if not v1_l and not v2_l:
            self.hint_var.set("💡 请先导入需要同步的视频文件")
        elif not out_l:
            self.hint_var.set("💡 视频已加载。请设置结果保存的目录")
        else:
            self.hint_var.set("✅ 所有配置已就绪，可以开始处理")

        self.btn_select_out.config(state=tk.NORMAL if (v1_l and v2_l) else tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL if (v1_l and v2_l and out_l) else tk.DISABLED)

    def load_video(self, video_num):
        f = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not f: return
        if any('\u4e00' <= c <= '\u9fff' for c in os.path.basename(f)):
            messagebox.showerror("文件名错误", "文件名包含中文，请修改后再导入。")
            return
        if video_num == 1:
            self.video1_path.set(f)
            self.show_first_frame(f, 1)
        else:
            self.video2_path.set(f)
            self.show_first_frame(f, 2)
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
                img = Image.fromarray(frame)
                h, w, _ = frame.shape
                self.video_resolutions[video_num] = (w, h)
                self.update_resolution_options()
                if video_num == 1: self.img1_canvas.set_image(img)
                else: self.img2_canvas.set_image(img)
        finally: os.chdir(cwd)

    def update_resolution_options(self):
        v1, v2 = self.video_resolutions[1], self.video_resolutions[2]
        if not v1 and not v2: return
        min_w = min([res[0] for res in [v1, v2] if res] or [99999])
        opts = ["原分辨率 (Original)"]
        if min_w > 1920: opts.append("1080p (最大 1920x1080)")
        if min_w > 1280: opts.append("720p (最大 1280x720)")
        if min_w > 854: opts.append("480p (最大 854x480)")
        self.res_cb.config(values=opts, state="readonly")
        if self.resolution_var.get() not in opts: self.resolution_var.set("原分辨率 (Original)")

    def select_output_dir(self):
        d = filedialog.askdirectory()
        if d: 
            self.output_dir.set(d)
            self.update_ui_states()

    def stop_processing(self):
        if self.is_processing:
            self.stop_requested = True
            self.log("终止进程中...")
            if self.current_process:
                try: self.current_process.terminate()
                except: pass

    def start_processing(self):
        try:
            t1 = int(self.h1_var.get())*3600 + int(self.m1_var.get())*60 + int(self.s1_var.get())
            t2 = int(self.h2_var.get())*3600 + int(self.m2_var.get())*60 + int(self.s2_var.get())
        except:
            messagebox.showwarning("错误", "时间输入格式不正确")
            return
        
        self.is_processing, self.stop_requested = True, False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=self.process_videos, args=(
            self.video1_path.get(), self.video2_path.get(), self.output_dir.get(), 
            t1, t2, {"原分辨率 (Original)": None, "1080p (最大 1920x1080)": (1920, 1080), 
                    "720p (最大 1280x720)": (1280, 720), "480p (最大 854x480)": (854, 480)}.get(self.resolution_var.get())
        ), daemon=True).start()

    def update_progress(self, pb, lbl, val, mx, eta):
        pb['maximum'], pb['value'] = mx, val
        lbl.config(text=eta)

    def process_videos(self, v1, v2, out, t1, t2, res_target):
        try:
            f_dir, b_dir = os.path.join(out, "forward"), os.path.join(out, "backward")
            os.makedirs(f_dir, exist_ok=True); os.makedirs(b_dir, exist_ok=True)
            diff = t1 - t2
            sk1, sk2 = max(0, -diff), max(0, diff)
            
            def extract(path, folder, skip, pref, pb, lbl):
                cap = cv2.VideoCapture(path)
                tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                
                f_size = os.path.getsize(path)
                real_dur = (f_size * 8) / (w * h * fps * 0.022) if (f_size*8)/(tot/fps if tot>0 else 1) < 500000 else tot/fps
                fps_r = round(fps)
                sk_f = int(skip * fps_r + fps_r // 2)
                exp_img = int((real_dur * fps - sk_f) / fps_r)
                
                out_w, out_h = int(w), int(h)
                if res_target:
                    mw, mh = res_target
                    if out_w > mw or out_h > mh:
                        sc = min(mw/out_w, mh/out_h)
                        out_w, out_h = int(out_w*sc)//2*2, int(out_h*sc)//2*2

                filter_str = f"select='gte(n,{sk_f})*not(mod(n-{sk_f},{fps_r}))'"
                if out_w != int(w): filter_str += f",scale={out_w}:{out_h}"
                
                cmd = ["ffmpeg", "-y", "-i", path, "-vf", filter_str, "-vsync", "0", "-q:v", "2", os.path.join(folder, f"{pref}_%06d.jpg")]
                proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='ignore')
                self.current_process = proc
                
                start_t, saved = time.time(), 0
                reg = re.compile(r"frame=\s*(\d+)")
                for line in proc.stderr:
                    if self.stop_requested: break
                    m = reg.search(line)
                    if m:
                        saved = int(m.group(1))
                        elap = time.time() - start_t
                        eta = f"{int((elap/saved)*(exp_img-saved))}s" if saved > 0 else "..."
                        self.root.after(0, self.update_progress, pb, lbl, saved, exp_img, eta)
                proc.wait()
                return saved

            c1 = extract(v1, f_dir, sk1, "front", self.progress1, self.eta_label1)
            c2 = 0 if self.stop_requested else extract(v2, b_dir, sk2, "back", self.progress2, self.eta_label2)
            
            if not self.stop_requested:
                min_c = min(c1, c2)
                for f, p in [(f_dir, "front"), (b_dir, "back")]:
                    files = sorted(glob.glob(os.path.join(f, f"{p}_*.jpg")))
                    for i in files[min_c:]: os.remove(i)
                messagebox.showinfo("完成", f"同步抽帧成功！共提取 {min_c} 组图片。")
            else:
                self.log("操作已由用户取消。")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            self.is_processing = False
            self.root.after(0, self.update_ui_states)
            self.stop_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSyncTool(root)
    root.mainloop()