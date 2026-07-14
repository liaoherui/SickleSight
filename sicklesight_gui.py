import argparse
import datetime
import importlib.util
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


APP_TITLE = "SickleSight Blood Cell Analysis Console"
SESSION_FILENAME = ".sicklesight_gui_state.json"
MODEL_FILES = [
    os.path.join("CellBox-Models", "best_model_vit_torch_macos_seven.pth"),
    os.path.join("CellBox-Models", "best_model_vit_torch_macos_raw_vit_large_binary.pth"),
    os.path.join("CellBox-Models", "best_model_vit_torch_macos_raw_vit_large_binary_pocked.pth"),
    os.path.join("CellBox-Models", "direct_vit_D.pt"),
    os.path.join("CellBox-Models", "direct_vit_E.pt"),
    os.path.join("CellBox-Models", "direct_vit_G.pt"),
    os.path.join("CellBox-Models", "siamese_vit_All_Haolin.pt"),
    os.path.join("CellBox-Models", "cyto3_train0327"),
]
LOW_RES_MODEL_FILES = [
    os.path.join("CellBox-Models", "yolo", "best.pt"),
    os.path.join("CellBox-Models", "seg", "best.pt"),
    os.path.join("CellBox-Models", "configs", "botsort_cell.yaml"),
    os.path.join("CellBox-Models", "efficientnet", "fold1_best.pth"),
    os.path.join("CellBox-Models", "efficientnet", "fold2_best.pth"),
    os.path.join("CellBox-Models", "efficientnet", "fold3_best.pth"),
    os.path.join("CellBox-Models", "efficientnet", "fold4_best.pth"),
    os.path.join("CellBox-Models", "efficientnet", "fold5_best.pth"),
    os.path.join("CellBox-Models", "siamese", "model.pth"),
]
PIPELINE_OPTIONS = [
    (
        "sicklesight_merged.py",
        "Combined analysis. Runs temporal state tracking and morphology in one pass.",
    ),
    (
        "sicklesight_part1.py",
        "Temporal state tracking. Measures sickled vs non-sickled dynamics over time.",
    ),
    (
        "sicklesight_part2.py",
        "Morphology analysis. Compares aspect ratio, eccentricity, and circularity.",
    ),
]
PIPELINE_ARGUMENTS = {
    "sicklesight_merged.py": ("frame_skip", "max_time", "analysis_fps", "full_video", "target_frames", "tracking_backend", "low_res_det_conf"),
    "sicklesight_part1.py": ("frame_skip", "max_time", "analysis_fps", "full_video", "tracking_backend", "low_res_det_conf"),
    "sicklesight_part2.py": ("frame_skip", "max_time", "analysis_fps", "full_video", "target_frames", "tracking_backend", "low_res_det_conf"),
}
TRACKING_BACKEND_OPTIONS = (
    ("Cellpose", "cellpose"),
    ("Low-resolution YOLO/BoT-SORT", "low_res"),
)
DEFAULT_CURSOR = "arrow"
ACTION_CURSOR = "hand2"
INSPECT_CURSOR = DEFAULT_CURSOR
SLIDER_CURSOR = ACTION_CURSOR
V_SCROLL_CURSOR = DEFAULT_CURSOR
GUI_RUNTIME_CHECK_ENV = "SICKLESIGHT_GUI_RUNTIME_CHECKED"


def module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def python_can_render_preview(python_path):
    try:
        result = subprocess.run(
            [python_path, "-c", "import cv2; import PIL"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def ensure_preview_runtime():
    if module_available("cv2") and module_available("PIL"):
        return
    if os.environ.get(GUI_RUNTIME_CHECK_ENV) == "1" or getattr(sys, "frozen", False):
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.environ.get("SICKLESIGHT_PYTHON", ""),
        os.path.join(script_dir, ".venv", "bin", "python"),
        os.path.join(script_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(script_dir, ".miniforge3", "envs", "sicklesight", "bin", "python"),
        os.path.join(script_dir, ".miniforge3", "envs", "sicklesight", "python.exe"),
    ]

    current_python = os.path.realpath(sys.executable)
    for candidate in candidate_paths:
        if not candidate or not os.path.exists(candidate):
            continue
        if os.path.realpath(candidate) == current_python:
            continue
        if not python_can_render_preview(candidate):
            continue
        env = os.environ.copy()
        env[GUI_RUNTIME_CHECK_ENV] = "1"
        os.execvpe(candidate, [candidate, os.path.abspath(__file__), *sys.argv[1:]], env)


class ModernButton(tk.Canvas):
    def __init__(
        self,
        parent,
        text,
        command,
        palette,
        font,
        variant="default",
        width=None,
        height=42,
    ):
        self.text = text
        self.command = command
        self.palette = palette
        self.font = tkfont.Font(font=font)
        self.variant = variant
        self.requested_width = width or max(self.font.measure(text) + 44, 104)
        self.requested_height = height
        self.hovered = False
        self.pressed = False
        self.enabled = True

        try:
            parent_bg = parent.cget("bg")
        except tk.TclError:
            parent_bg = palette["bg"]

        super().__init__(
            parent,
            width=self.requested_width,
            height=self.requested_height,
            bg=parent_bg,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            cursor=DEFAULT_CURSOR,
            takefocus=1,
        )
        self.bind("<Configure>", lambda _event: self.draw())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<KeyPress-Return>", self.on_keyboard_activate)
        self.bind("<KeyPress-space>", self.on_keyboard_activate)
        self.draw()

    def colors_for_state(self):
        if self.variant == "accent":
            fill = self.palette["primary_button_bg"]
            hover = self.palette["primary_button_hover"]
            pressed = self.palette["primary_button_pressed"]
            border = self.palette["accent"]
            text = "#FFFFFF"
        elif self.variant == "danger":
            fill = self.palette["danger_button_bg"]
            hover = self.palette["danger_soft"]
            pressed = "#FEE4E2"
            border = "#FDA29B"
            text = self.palette["danger"]
        else:
            fill = self.palette["button_bg"]
            hover = self.palette["button_hover"]
            pressed = self.palette["button_pressed"]
            border = self.palette["glass_border"]
            text = self.palette["button_text"]

        if not self.enabled:
            return self.palette["card_alt"], self.palette["glass_border"], self.palette["muted"]
        if self.pressed:
            return pressed, border, text
        if self.hovered:
            return hover, border, text
        return fill, border, text

    def rounded_points(self, x1, y1, x2, y2, radius):
        return [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]

    def draw(self):
        self.delete("all")
        width = max(self.winfo_width(), self.requested_width)
        height = max(self.winfo_height(), self.requested_height)
        fill, border, text_color = self.colors_for_state()
        radius = min(15, max(8, height // 2 - 3))
        y_shift = 1 if self.pressed else 0

        if self.enabled and self.variant == "accent" and not self.pressed:
            self.create_polygon(
                self.rounded_points(3, 5, width - 3, height - 1, radius),
                smooth=True,
                splinesteps=20,
                fill="#D0D5DD",
                outline="",
            )

        self.create_polygon(
            self.rounded_points(2, 2, width - 2, height - 4, radius),
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=border,
            width=1,
        )
        self.create_text(
            width / 2,
            (height - 2) / 2 + y_shift,
            text=self.text,
            fill=text_color,
            font=self.font,
        )

    def on_enter(self, _event):
        if self.enabled:
            self.hovered = True
            self.configure(cursor=ACTION_CURSOR)
            self.draw()

    def on_leave(self, _event):
        self.hovered = False
        self.pressed = False
        self.configure(cursor=DEFAULT_CURSOR)
        self.draw()

    def on_press(self, _event):
        if self.enabled:
            self.pressed = True
            self.draw()

    def on_release(self, event):
        if not self.enabled:
            return
        was_pressed = self.pressed
        self.pressed = False
        self.draw()
        if was_pressed and 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            self.command()

    def on_keyboard_activate(self, _event):
        if self.enabled:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.configure(cursor=DEFAULT_CURSOR)
        self.draw()


class SickleAnalysisGUI:
    def __init__(
        self,
        root,
        startup_video_paths=None,
        startup_folder_paths=None,
        startup_pipeline_dir=None,
        startup_output_dir=None,
    ):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(self.initial_window_geometry())
        self.root.minsize(1180, 740)
        self.root.configure(bg="#F4F7FB")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cwd = os.getcwd()
        self.is_windows = platform.system() == "Windows"
        self.compact_figure_mode = True
        self.colors = {
            "bg": "#F5F6F7",
            "bg_alt": "#EAECF0",
            "card": "#FFFFFF",
            "card_alt": "#F8FAFC",
            "glass": "#F9FAFB",
            "glass_alt": "#F2F4F7",
            "glass_border": "#D0D5DD",
            "glass_shine": "#FFFFFF",
            "glass_shadow": "#D0D5DD",
            "glass_tint": "#F2F4F7",
            "border": "#111827",
            "border_soft": "#E4E7EC",
            "text": "#101828",
            "muted": "#667085",
            "accent": "#111827",
            "accent_dark": "#0B1220",
            "accent_soft": "#EAECF0",
            "secondary": "#344054",
            "secondary_dark": "#1D2939",
            "secondary_soft": "#E6F5F3",
            "mahogany": "#991B1B",
            "warning": "#B54708",
            "danger": "#D92D20",
            "danger_soft": "#FEF3F2",
            "success": "#079455",
            "success_soft": "#ECFDF3",
            "pending": "#DC6803",
            "pending_soft": "#FFF8E6",
            "terminal_bg": "#111827",
            "terminal_fg": "#E5E7EB",
            "button_bg": "#FFFFFF",
            "button_hover": "#F8FAFC",
            "button_pressed": "#EAECF0",
            "button_text": "#344054",
            "primary_button_bg": "#111827",
            "primary_button_hover": "#1F2937",
            "primary_button_pressed": "#030712",
            "danger_button_bg": "#FFFFFF",
        }
        self.setup_fonts()

        self.selected_folders = []
        self.pipeline_dir = ""
        self.output_dir = ""
        self.python_executable = sys.executable
        self.last_script_path = ""
        self.last_results_dir = ""
        self.active_process = None
        self.process_thread = None
        self.run_started_at = None
        self.progress = None
        self.checks_tree = None
        self.latest_checks = []
        self.preferred_video_paths = []
        self.pinned_video_paths = []
        self.available_python_options = []
        self.preview_photo = None
        self.preview_capture = None
        self.preview_video_path = ""
        self.preview_total_frames = 0
        self.preview_fps = 0.0
        self.preview_width = 0
        self.preview_height = 0
        self.preview_slider_job = None
        self.preview_scrub_var = tk.DoubleVar(value=0)
        self.preview_status_var = tk.StringVar(value="Select a video to load an inline preview.")
        self.preview_zoom_levels = [1.0, 1.25, 1.5, 2.0, 3.0]
        self.preview_zoom_index = 0
        self.preview_zoom_var = tk.StringVar(value="1x")
        self.results_preview_window = None
        self.results_preview_tree = None
        self.results_preview_display = None
        self.results_preview_photo = None
        self.results_preview_status_var = tk.StringVar(value="No result file selected.")
        self.results_preview_files = {}
        self.results_preview_selected_path = ""
        self.inline_results_status_var = tk.StringVar(value="No results selected yet.")
        self.inline_results_position_var = tk.StringVar(value="0 / 0")
        self.inline_results_photo = None
        self.inline_results_preview_path = ""
        self.inline_results_files = []
        self.inline_results_index = 0
        self.inline_results_window_start = 0
        self.inline_results_slot_photos = []
        self.inline_results_slots = []
        self.fullscreen_preview_window = None
        self.fullscreen_preview_photo = None

        self.script_output_dir = os.path.join(self.cwd, "_tmp_scripts")
        self.runtime_cache_dir = os.path.join(self.cwd, "_runtime_cache")
        self.hf_cache_dir = os.path.join(self.runtime_cache_dir, "huggingface")
        self.session_path = os.path.join(self.cwd, SESSION_FILENAME)
        os.makedirs(self.script_output_dir, exist_ok=True)
        os.makedirs(self.runtime_cache_dir, exist_ok=True)
        os.makedirs(self.hf_cache_dir, exist_ok=True)

        self.selected_pipeline_var = tk.StringVar()
        self.selected_python_var = tk.StringVar()
        self.frame_skip_var = tk.StringVar(value="2")
        self.max_frame_var = tk.StringVar(value="480")
        self.max_seconds_var = tk.StringVar(value="120")
        self.analysis_fps_var = tk.StringVar(value="")
        self.target_frames_var = tk.StringVar(value="")
        self.low_res_det_conf_var = tk.StringVar(value="auto")
        self.tracking_backend_var = tk.StringVar(value="cellpose")
        self.full_video_var = tk.BooleanVar(value=False)
        self.stage_var = tk.StringVar(value="Ready for setup")
        self.output_summary_var = tk.StringVar(
            value="Results will appear here after a successful run."
        )
        self.script_summary_var = tk.StringVar(value="No script generated yet.")
        self.selection_count_var = tk.StringVar(value="0 folders in pool")

        self.session_data = self.load_session()

        self.configure_styles()
        self.create_layout()
        self.bind_state_events()
        self.restore_session_state()
        self.autodetect_project_context()
        self.refresh_python_candidates()
        self.apply_startup_paths(startup_video_paths or [], startup_folder_paths or [])
        self.apply_startup_config(startup_pipeline_dir, startup_output_dir)
        self.refresh_tree()
        self.run_quick_status_refresh()
        self.log_to_terminal("Ready. Configure inputs, run a preflight check, then launch analysis.")
        self.root.after(350, self.bring_window_forward)

    def initial_window_geometry(self):
        screen_width = max(self.root.winfo_screenwidth(), 1240)
        screen_height = max(self.root.winfo_screenheight(), 780)
        width = min(max(int(screen_width * 0.96), 1240), 1600)
        height = min(max(int(screen_height * 0.90), 780), 980)
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 3, 0)
        return f"{width}x{height}+{x}+{y}"

    def setup_fonts(self):
        available = set(tkfont.families())
        self.display_family = self.pick_font_family(
            available,
            ["Google Sans", "SF Pro Display", "Helvetica Neue", "Helvetica", "Arial"],
        )
        self.body_family = self.pick_font_family(
            available,
            ["Google Sans Text", "SF Pro Text", "Helvetica Neue", "Helvetica", "Arial"],
        )
        self.code_family = self.pick_font_family(available, ["JetBrains Mono", "SF Mono", "Menlo", "Monaco", "Courier New"])

        if self.compact_figure_mode:
            self.font_title_xl = (self.display_family, 16, "bold")
            self.font_title_md = (self.display_family, 12, "bold")
            self.font_body_lg = (self.body_family, 10, "bold")
            self.font_body = (self.body_family, 10)
            self.font_body_bold = (self.body_family, 10, "bold")
            self.font_button = (self.body_family, 9, "bold")
            self.font_small = (self.body_family, 9)
            self.font_small_bold = (self.body_family, 9, "bold")
            self.font_meta = (self.body_family, 10)
            self.font_code = (self.code_family, 9)
            self.font_code_small = (self.code_family, 8)
            return

        self.font_title_xl = (self.display_family, 26, "bold")
        self.font_title_md = (self.display_family, 16, "bold")
        self.font_body_lg = (self.body_family, 13, "bold")
        self.font_body = (self.body_family, 12)
        self.font_body_bold = (self.body_family, 12, "bold")
        self.font_button = (self.body_family, 11, "bold")
        self.font_small = (self.body_family, 10)
        self.font_small_bold = (self.body_family, 10, "bold")
        self.font_meta = (self.body_family, 12)
        self.font_code = (self.code_family, 10)
        self.font_code_small = (self.code_family, 9)

    def pick_font_family(self, available, candidates):
        for family in candidates:
            if family in available:
                return family
        return candidates[-1]

    def configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure("App.TFrame", background=self.colors["bg"])
        self.style.configure(
            "CardTitle.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=self.font_title_md,
        )
        self.style.configure(
            "CardMeta.TLabel",
            background=self.colors["card"],
            foreground=self.colors["muted"],
            font=self.font_body,
        )
        self.style.configure(
            "Section.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=self.font_body_bold,
        )
        self.style.configure(
            "Body.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=self.font_body,
        )
        self.style.configure(
            "Muted.TLabel",
            background=self.colors["card"],
            foreground=self.colors["muted"],
            font=self.font_small,
        )
        self.style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="white",
            padding=(14, 9),
            borderwidth=0,
            focusthickness=0,
            font=self.font_body_bold,
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_dark"])],
            foreground=[("disabled", self.colors["text"])],
        )
        self.style.configure(
            "Secondary.TButton",
            background=self.colors["glass"],
            foreground=self.colors["text"],
            padding=(12, 8),
            bordercolor=self.colors["glass_border"],
            lightcolor=self.colors["glass"],
            darkcolor=self.colors["glass"],
            font=self.font_body,
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", self.colors["glass_alt"])],
        )
        self.style.configure(
            "Danger.TButton",
            background=self.colors["danger"],
            foreground="white",
            padding=(12, 8),
            borderwidth=0,
            font=self.font_body_bold,
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", self.colors["mahogany"])],
        )
        self.style.configure(
            "App.TCombobox",
            fieldbackground=self.colors["glass"],
            background=self.colors["glass"],
            foreground=self.colors["text"],
            bordercolor=self.colors["glass_border"],
            padding=6,
        )
        self.style.configure(
            "App.TRadiobutton",
            background=self.colors["glass"],
            foreground=self.colors["text"],
            font=self.font_body,
        )
        self.style.map(
            "App.TRadiobutton",
            background=[("active", self.colors["glass"])],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "App.TCheckbutton",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=self.font_body,
        )
        self.style.map(
            "App.TCheckbutton",
            background=[("active", self.colors["card"])],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "Video.Treeview",
            background=self.colors["glass"],
            fieldbackground=self.colors["glass"],
            foreground=self.colors["text"],
            bordercolor=self.colors["glass_border"],
            rowheight=24 if self.compact_figure_mode else 30,
            font=self.font_body,
        )
        self.style.configure(
            "Video.Treeview.Heading",
            background=self.colors["card_alt"],
            foreground=self.colors["text"],
            bordercolor=self.colors["glass_border"],
            font=self.font_body_bold,
        )
        self.style.map(
            "Video.Treeview",
            background=[("selected", self.colors["accent_soft"])],
            foreground=[("selected", self.colors["text"])],
        )
        self.style.configure(
            "Checks.Treeview",
            background=self.colors["glass"],
            fieldbackground=self.colors["glass"],
            foreground=self.colors["text"],
            bordercolor=self.colors["glass_border"],
            rowheight=24 if self.compact_figure_mode else 30,
            font=self.font_body,
        )
        self.style.configure(
            "Checks.Treeview.Heading",
            background=self.colors["card_alt"],
            foreground=self.colors["text"],
            bordercolor=self.colors["glass_border"],
            font=self.font_body_bold,
        )
        self.style.map(
            "Checks.Treeview",
            background=[("selected", self.colors["accent_soft"])],
            foreground=[("selected", self.colors["text"])],
        )
        self.style.configure(
            "Run.Horizontal.TProgressbar",
            background=self.colors["accent"],
            troughcolor=self.colors["glass_alt"],
            bordercolor=self.colors["glass_alt"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )
        self.style.configure(
            "App.Vertical.TScrollbar",
            background=self.colors["glass_alt"],
            troughcolor=self.colors["bg_alt"],
            bordercolor=self.colors["glass_border"],
            arrowcolor=self.colors["text"],
            width=12,
        )
        self.style.configure(
            "App.Horizontal.TScrollbar",
            background=self.colors["glass_alt"],
            troughcolor=self.colors["bg_alt"],
            bordercolor=self.colors["glass_border"],
            arrowcolor=self.colors["text"],
            width=12,
        )

    def create_layout(self):
        page_shell = tk.Frame(self.root, bg=self.colors["bg"])
        page_shell.pack(fill=tk.BOTH, expand=True)

        self.page_canvas = tk.Canvas(
            page_shell,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        page_scroll = ttk.Scrollbar(
            page_shell,
            orient=tk.VERTICAL,
            command=self.page_canvas.yview,
            style="App.Vertical.TScrollbar",
            cursor=V_SCROLL_CURSOR,
        )
        self.page_scroll = page_scroll
        self.page_scroll_visible = False
        self.page_canvas.configure(yscrollcommand=page_scroll.set)
        self.page_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.page_content = ttk.Frame(self.page_canvas, style="App.TFrame")
        self.page_window = self.page_canvas.create_window((0, 0), window=self.page_content, anchor="nw")
        self.page_content.bind("<Configure>", self.on_page_content_configure)
        self.page_canvas.bind("<Configure>", self.on_page_canvas_configure)
        self.root.bind_all("<MouseWheel>", self.on_global_mousewheel, add="+")

        self.main = ttk.Frame(self.page_content, style="App.TFrame", padding=6 if self.compact_figure_mode else 12)
        self.main.pack(fill=tk.BOTH, expand=True)
        self.main.grid_columnconfigure(0, weight=1)
        if self.compact_figure_mode:
            self.main.grid_rowconfigure(1, weight=1, minsize=510)
        else:
            self.main.grid_rowconfigure(1, weight=10, minsize=520)
            self.main.grid_rowconfigure(2, weight=2, minsize=150)

        self.build_header()

        self.workspace = tk.Frame(self.main, bg=self.colors["bg"])
        self.workspace.grid(row=1, column=0, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=0, minsize=285 if self.compact_figure_mode else 350)
        self.workspace.grid_columnconfigure(1, weight=1)
        self.workspace.grid_rowconfigure(0, weight=1)

        self.build_left_column()
        self.build_analysis_workspace()
        if not self.compact_figure_mode:
            self.build_terminal_card()

    def create_button(self, parent, text, command, variant="glass", width=None, height=38):
        button_variant = "default" if variant == "glass" else variant
        return ModernButton(
            parent,
            text,
            command,
            self.colors,
            self.font_button,
            variant=button_variant,
            width=width,
            height=height,
        )

    def set_hover_cursor(self, widget, hover_cursor=ACTION_CURSOR, normal_cursor=DEFAULT_CURSOR):
        widget.configure(cursor=normal_cursor)
        widget.bind("<Enter>", lambda _event: widget.configure(cursor=hover_cursor), add="+")
        widget.bind("<Leave>", lambda _event: widget.configure(cursor=normal_cursor), add="+")

    def build_analysis_workspace(self):
        self.analysis_body = tk.Frame(self.workspace, bg=self.colors["bg"])
        self.analysis_body.grid(row=0, column=1, sticky="nsew", padx=(6 if self.compact_figure_mode else 12, 0))
        if self.compact_figure_mode:
            self.analysis_body.grid_columnconfigure(0, weight=7, minsize=540)
            self.analysis_body.grid_columnconfigure(1, weight=3, minsize=320)
            self.analysis_body.grid_rowconfigure(0, weight=1, minsize=500)
            self.analysis_body.grid_rowconfigure(1, weight=0, minsize=170)
        else:
            self.analysis_body.grid_columnconfigure(0, weight=1, minsize=0)
            self.analysis_body.grid_rowconfigure(0, weight=1, minsize=520)

        self.build_preview_card()

        if self.compact_figure_mode:
            self.build_inline_results_card()
            self.build_terminal_card(
                parent=self.analysis_body,
                row=1,
                column=0,
                columnspan=1,
                pady=(6, 0),
            )
            return

        evidence_row = tk.Frame(self.analysis_body, bg=self.colors["bg"])
        evidence_row.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        evidence_row.grid_columnconfigure(0, weight=3, uniform="evidence")
        evidence_row.grid_columnconfigure(1, weight=2, uniform="evidence")
        evidence_row.grid_rowconfigure(0, weight=1)

        self.status_card_body = self.create_card(
            evidence_row,
            row=0,
            column=0,
            title="Quality Gate",
            description="Input, model, runtime, and output checks before analysis starts.",
            pady=(0, 0),
        )
        self.build_status_card_contents()

        self.results_card_body = self.create_card(
            evidence_row,
            row=0,
            column=1,
            title="Reproducibility",
            description="Launcher script and latest output summary for auditability.",
            pady=(0, 0),
        )
        self.build_results_card_contents()

    def build_preview_card(self):
        parent = getattr(self, "analysis_body", self.main)
        self.preview_card_body = self.create_card(
            parent,
            row=0,
            column=0,
            title="Microscopy Imaging Workspace",
            description="Large-field preview of the selected blood-cell video before analysis.",
            columnspan=1 if self.compact_figure_mode else 2,
            pady=(0, 0),
        )

        preview_stack = tk.Frame(self.preview_card_body, bg=self.colors["card"])
        preview_stack.pack(fill=tk.BOTH, expand=True)
        preview_stack.grid_columnconfigure(0, weight=1)
        preview_stack.grid_rowconfigure(0, weight=1, minsize=300 if self.compact_figure_mode else 430)
        preview_stack.grid_rowconfigure(1, weight=0)
        preview_stack.grid_rowconfigure(2, weight=0)

        self.preview_frame_shell = tk.Frame(
            preview_stack,
            bg=self.colors["card"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
            bd=0,
        )
        self.preview_frame_shell.grid(row=0, column=0, sticky="nsew", pady=(0, 10 if self.compact_figure_mode else 12))

        self.preview_frame_container = tk.Frame(
            self.preview_frame_shell,
            bg=self.colors["terminal_bg"],
            height=300 if self.compact_figure_mode else 420,
            padx=6 if self.compact_figure_mode else 12,
            pady=6 if self.compact_figure_mode else 12,
        )
        self.set_hover_cursor(self.preview_frame_container, INSPECT_CURSOR)
        self.preview_frame_container.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_figure_mode else 12, pady=6 if self.compact_figure_mode else 12)
        self.preview_frame_container.pack_propagate(False)

        self.preview_visual = tk.Label(
            self.preview_frame_container,
            text="Inline preview will appear here",
            bg=self.colors["terminal_bg"],
            fg=self.colors["muted"],
            font=self.font_body_lg,
            justify=tk.CENTER,
            anchor="center",
            padx=12,
            pady=12,
        )
        self.set_hover_cursor(self.preview_visual, INSPECT_CURSOR)
        self.preview_visual.pack(fill=tk.BOTH, expand=True)

        scrub_row = tk.Frame(preview_stack, bg=self.colors["card"])
        scrub_row.grid(row=1, column=0, sticky="ew", pady=(0, 4 if self.compact_figure_mode else 12))
        scrub_row.grid_columnconfigure(0, weight=1)
        self.preview_scale = tk.Scale(
            scrub_row,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.preview_scrub_var,
            showvalue=False,
            resolution=1,
            state=tk.DISABLED,
            command=self.on_preview_slider_changed,
            bg=self.colors["card"],
            fg=self.colors["text"],
            troughcolor=self.colors["glass_alt"],
            highlightthickness=0,
            activebackground=self.colors["accent"],
            sliderlength=24 if self.compact_figure_mode else 34,
            width=10 if self.compact_figure_mode else 20,
            cursor=DEFAULT_CURSOR,
        )
        self.set_hover_cursor(self.preview_scale, SLIDER_CURSOR)
        self.preview_scale.grid(row=0, column=0, sticky="ew", pady=(0, 5 if self.compact_figure_mode else 8))
        control_height = 28 if self.compact_figure_mode else 38
        controls_row = tk.Frame(scrub_row, bg=self.colors["card"])
        controls_row.grid(row=1, column=0, sticky="ew")

        frame_nav = tk.Frame(controls_row, bg=self.colors["card"])
        frame_nav.pack(side=tk.LEFT)
        self.create_button(frame_nav, "First", self.preview_first_frame, width=70 if self.compact_figure_mode else 88, height=control_height).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.create_button(frame_nav, "Middle", self.preview_middle_frame, width=82 if self.compact_figure_mode else 104, height=control_height).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.create_button(frame_nav, "Last", self.preview_last_frame, width=70 if self.compact_figure_mode else 88, height=control_height).pack(
            side=tk.LEFT
        )

        zoom_group = tk.Frame(controls_row, bg=self.colors["card"])
        zoom_group.pack(side=tk.RIGHT)
        self.create_button(zoom_group, "-", self.preview_zoom_out, width=34 if self.compact_figure_mode else 42, height=control_height).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.preview_zoom_label = tk.Label(
            zoom_group,
            textvariable=self.preview_zoom_var,
            bg=self.colors["glass_alt"],
            fg=self.colors["text"],
            font=self.font_small_bold,
            width=6 if self.compact_figure_mode else 7,
            padx=6 if self.compact_figure_mode else 8,
            pady=5 if self.compact_figure_mode else 9,
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        self.preview_zoom_label.pack(side=tk.LEFT, padx=(0, 6))
        self.create_button(zoom_group, "+", self.preview_zoom_in, width=34 if self.compact_figure_mode else 42, height=control_height).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.create_button(zoom_group, "Reset", self.preview_zoom_fit, width=58 if self.compact_figure_mode else 72, height=control_height).pack(side=tk.LEFT)

        metadata_panel = tk.Frame(
            preview_stack,
            bg=self.colors["glass"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        metadata_panel.grid(row=2, column=0, sticky="ew")
        metadata_panel.grid_columnconfigure(0, weight=1)
        metadata_panel.grid_columnconfigure(1, weight=0, minsize=250 if self.compact_figure_mode else 340)

        tk.Label(
            metadata_panel,
            text="Selected Video",
            bg=self.colors["glass"],
            fg=self.colors["text"],
            font=self.font_body_bold,
            padx=10 if self.compact_figure_mode else 14,
            pady=5 if self.compact_figure_mode else 12,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.preview_status_label = tk.Label(
            metadata_panel,
            textvariable=self.preview_status_var,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            justify=tk.LEFT,
            anchor="w",
            wraplength=520 if self.compact_figure_mode else 820,
            padx=10 if self.compact_figure_mode else 14,
            pady=4 if self.compact_figure_mode else 10,
            font=self.font_code,
            highlightthickness=0,
        )
        self.preview_status_label.grid(row=1, column=0, sticky="ew", padx=(10 if self.compact_figure_mode else 14, 8), pady=(0, 5 if self.compact_figure_mode else 10))

        self.preview_text = tk.Text(
            metadata_panel,
            height=1 if self.compact_figure_mode else 5,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            relief=tk.FLAT,
            highlightthickness=0,
            highlightcolor=self.colors["border"],
            wrap=tk.WORD,
            font=self.font_code,
            padx=10 if self.compact_figure_mode else 14,
            pady=4 if self.compact_figure_mode else 8,
            insertbackground=self.colors["text"],
            cursor="xterm",
        )
        self.preview_text.grid(row=2, column=0, sticky="ew", padx=(10 if self.compact_figure_mode else 14, 8), pady=(0, 6 if self.compact_figure_mode else 12))
        self.preview_text.insert(
            tk.END,
            "No selection yet.\n\nPick a folder or video from the left to review its details.",
        )
        self.preview_text.config(state=tk.DISABLED)

        preview_actions = tk.Frame(metadata_panel, bg=self.colors["glass"])
        preview_actions.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(8, 10 if self.compact_figure_mode else 14), pady=(0, 6 if self.compact_figure_mode else 14))
        preview_actions.grid_columnconfigure(0, weight=1)
        preview_actions.grid_columnconfigure(1, weight=1)
        self.create_button(
            preview_actions,
            "Player" if self.compact_figure_mode else "Open Player",
            self.play_video,
            height=control_height,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 5))
        self.create_button(
            preview_actions,
            "Folder" if self.compact_figure_mode else "Open Folder",
            self.open_selected_location,
            height=control_height,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 5))
        self.create_button(
            preview_actions,
            "Save Frame",
            self.save_current_preview_frame,
            height=control_height,
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

    def build_inline_results_card(self):
        self.inline_results_card_body = self.create_card(
            self.analysis_body,
            row=0,
            column=1,
            title="Output Preview",
            description="Three key result figures from the selected output folder.",
            rowspan=2,
            pady=(0, 0),
            padx=(8, 0),
        )

        shell = tk.Frame(self.inline_results_card_body, bg=self.colors["card"])
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=1)
        self.inline_results_image_holder = None
        self.inline_results_image_label = None
        figure_titles = (
            "Sickled Fraction",
            "Class Distribution",
            "State Dynamics",
        )
        self.inline_results_slots = []
        for slot_index in range(3):
            shell.grid_rowconfigure(slot_index, weight=1, uniform="result_slots")
            slot = tk.Frame(
                shell,
                bg=self.colors["glass"],
                height=118,
                highlightthickness=1,
                highlightbackground=self.colors["glass_border"],
                cursor=ACTION_CURSOR,
            )
            slot.grid(row=slot_index, column=0, sticky="nsew", pady=(0 if slot_index == 0 else 6, 0))
            slot.grid_propagate(False)
            slot.grid_columnconfigure(0, weight=1)
            slot.grid_rowconfigure(1, weight=1)

            title_label = tk.Label(
                slot,
                text=figure_titles[slot_index],
                bg=self.colors["glass"],
                fg=self.colors["text"],
                font=self.font_small_bold,
                anchor="w",
                padx=8,
                pady=3,
            )
            title_label.grid(row=0, column=0, sticky="ew")

            image_label = tk.Label(
                slot,
                text="Result preview",
                bg=self.colors["glass"],
                fg=self.colors["muted"],
                font=self.font_code_small,
                justify=tk.CENTER,
            )
            image_label.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 2))

            text_label = tk.Label(
                slot,
                text="",
                bg=self.colors["glass"],
                fg=self.colors["muted"],
                font=self.font_code_small,
                justify=tk.LEFT,
                anchor="w",
                wraplength=280,
            )
            text_label.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 4))

            for widget in (slot, title_label, image_label, text_label):
                widget.bind("<Button-1>", lambda _event, index=slot_index: self.open_inline_result_slot_fullscreen(index), add="+")
                self.set_hover_cursor(widget, ACTION_CURSOR)
            self.inline_results_slots.append((slot, image_label, text_label))

        self.root.after(150, self.refresh_inline_results_preview)

    def build_header(self):
        header = tk.Frame(self.main, bg=self.colors["bg"])
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5 if self.compact_figure_mode else 18))
        header.grid_columnconfigure(0, weight=1)

        brand_frame = tk.Frame(header, bg=self.colors["bg"])
        brand_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            brand_frame,
            text="SickleSight Analysis Console" if self.compact_figure_mode else APP_TITLE,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.font_title_xl,
        ).pack(anchor="w")
        if self.compact_figure_mode:
            tk.Label(
                brand_frame,
                textvariable=self.stage_var,
                bg=self.colors["bg"],
                fg=self.colors["muted"],
                font=self.font_small_bold,
                wraplength=360,
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(2, 0))
        else:
            tk.Label(
                brand_frame,
                text="Microscopy video input, frame preview, pipeline setup, and reproducible sickle-cell analysis.",
                bg=self.colors["bg"],
                fg=self.colors["muted"],
                font=self.font_meta,
                wraplength=720,
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(6, 0))

        action_frame = tk.Frame(header, bg=self.colors["bg"])
        action_frame.grid(row=0, column=1, sticky="e")
        action_buttons = tk.Frame(action_frame, bg=self.colors["bg"])
        action_buttons.pack(anchor="e")

        self.create_button(
            action_buttons,
            "Run" if self.compact_figure_mode else "Run Analysis",
            self.run_analysis,
            variant="accent",
            width=80 if self.compact_figure_mode else 176,
            height=30 if self.compact_figure_mode else 38,
        ).pack(side=tk.LEFT)
        self.create_button(
            action_buttons,
            "Check" if self.compact_figure_mode else "Run Preflight",
            self.run_preflight_check,
            width=80 if self.compact_figure_mode else 166,
            height=30 if self.compact_figure_mode else 38,
        ).pack(side=tk.LEFT, padx=(6 if self.compact_figure_mode else 10, 0))
        self.create_button(
            action_buttons,
            "Output" if self.compact_figure_mode else "Open Output",
            self.open_output_folder,
            width=84 if self.compact_figure_mode else 156,
            height=30 if self.compact_figure_mode else 38,
        ).pack(side=tk.LEFT, padx=(6 if self.compact_figure_mode else 10, 0))
        self.create_button(
            action_buttons,
            "Results" if self.compact_figure_mode else "Preview Results",
            self.open_results_preview_window,
            width=84 if self.compact_figure_mode else 164,
            height=30 if self.compact_figure_mode else 38,
        ).pack(side=tk.LEFT, padx=(6 if self.compact_figure_mode else 10, 0))
        self.stop_button = self.create_button(
            action_buttons,
            "Stop",
            self.stop_analysis,
            variant="danger",
            width=68 if self.compact_figure_mode else 96,
            height=30 if self.compact_figure_mode else 38,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(6 if self.compact_figure_mode else 10, 0))

        badges_frame = tk.Frame(action_frame, bg=self.colors["bg"])
        badges_frame.pack(anchor="e", pady=(10, 0))

        self.badge_widgets = {}
        for label in ("Videos", "Pipeline", "Models", "Environment"):
            badge = tk.Label(
                badges_frame,
                text=f"{label}: Pending",
                bg=self.colors["card_alt"],
                fg=self.colors["muted"],
                padx=12,
                pady=6,
                font=self.font_code_small,
                bd=0,
                highlightthickness=1,
                highlightbackground=self.colors["glass_border"],
            )
            if not self.compact_figure_mode:
                badge.pack(side=tk.LEFT, padx=(0, 10))
            self.badge_widgets[label] = badge

    def build_left_column(self):
        left = tk.Frame(self.workspace, bg=self.colors["bg"], width=295 if self.compact_figure_mode else 390)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_rowconfigure(0, weight=0 if self.compact_figure_mode else 3, minsize=148 if self.compact_figure_mode else 0)
        left.grid_rowconfigure(1, weight=1, minsize=0)
        left.grid_columnconfigure(0, weight=1)

        self.sources_card_body = self.create_card(
            left,
            row=0,
            title="Input Cohort",
            description="Folders and videos selected for analysis.",
        )
        toolbar = tk.Frame(self.sources_card_body, bg=self.colors["card"])
        toolbar.pack(fill=tk.X, pady=(0, 6 if self.compact_figure_mode else 12))
        toolbar.grid_columnconfigure(0, weight=1)
        toolbar.grid_columnconfigure(1, weight=1)
        source_button_height = 24 if self.compact_figure_mode else 38
        self.create_button(toolbar, "Add Folder", self.add_folders, height=source_button_height).grid(
            row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 5)
        )
        self.create_button(toolbar, "Add Video", self.add_video_files, height=source_button_height).grid(
            row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 5)
        )
        self.create_button(toolbar, "Remove", self.remove_selection, height=source_button_height).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        self.create_button(toolbar, "Clear", self.clear_all_sources, height=source_button_height).grid(
            row=1, column=1, sticky="ew", padx=(6, 0)
        )

        count_row = tk.Frame(self.sources_card_body, bg=self.colors["card"])
        count_row.pack(fill=tk.X, pady=(0, 5 if self.compact_figure_mode else 8))
        tk.Label(
            count_row,
            textvariable=self.selection_count_var,
            bg=self.colors["card"],
            fg=self.colors["muted"],
            font=self.font_body,
        ).pack(anchor="w")

        tree_frame = tk.Frame(self.sources_card_body, bg=self.colors["card"])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("path",)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree",
            selectmode="extended",
            style="Video.Treeview",
            height=3 if self.compact_figure_mode else 24,
            cursor=DEFAULT_CURSOR,
        )
        self.set_hover_cursor(self.tree, ACTION_CURSOR)
        self.tree.column("#0", width=330, stretch=True)
        self.tree.column("path", width=0, stretch=False)
        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview,
            style="App.Vertical.TScrollbar",
            cursor=V_SCROLL_CURSOR,
        )
        self.tree.configure(yscroll=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_file)
        self.tree.bind("<Double-1>", self.on_double_click_file)

        self.setup_card_body = self.create_card(
            left,
            row=1,
            title="Analysis Protocol",
            description="Pipeline, sampling parameters, runtime, and output.",
            pady=(6 if self.compact_figure_mode else 14, 0),
        )
        self.build_setup_card_contents()

    def build_setup_card_contents(self):
        setup_panel = tk.Frame(self.setup_card_body, bg=self.colors["card"])
        setup_panel.pack(fill=tk.X)

        self.pipeline_options_frame = tk.Frame(setup_panel, bg=self.colors["card"])
        self.build_subsection_label(
            setup_panel,
            "Pipeline Mode",
            "Select the SickleSight script for this run.",
        )
        self.pipeline_options_frame.pack(fill=tk.X, pady=(0, 4 if self.compact_figure_mode else 14))

        self.build_subsection_label(
            setup_panel,
            "Segmentation / Tracking",
            "Choose Cellpose for standard videos or YOLO/BoT-SORT for low-resolution videos.",
        )
        backend_frame = tk.Frame(setup_panel, bg=self.colors["card"])
        backend_frame.pack(fill=tk.X, pady=(0, 4 if self.compact_figure_mode else 14))
        for label, value in TRACKING_BACKEND_OPTIONS:
            row = tk.Frame(
                backend_frame,
                bg=self.colors["glass"],
                highlightthickness=1,
                highlightbackground=self.colors["glass_border"],
            )
            row.pack(fill=tk.X, pady=(0, 3 if self.compact_figure_mode else 6))
            self.set_hover_cursor(row, ACTION_CURSOR)
            radio = ttk.Radiobutton(
                row,
                text=label,
                value=value,
                variable=self.tracking_backend_var,
                style="App.TRadiobutton",
            )
            self.set_hover_cursor(radio, ACTION_CURSOR)
            radio.pack(anchor="w", padx=9 if self.compact_figure_mode else 12, pady=3 if self.compact_figure_mode else 9)

        self.build_subsection_label(
            setup_panel,
            "Analysis Duration",
            "Duration, sampling, and target-frame controls used by the selected script.",
        )
        params_grid = tk.Frame(setup_panel, bg=self.colors["card"])
        params_grid.pack(fill=tk.X, pady=(0, 4 if self.compact_figure_mode else 14))
        if self.compact_figure_mode:
            compact_params = (
                ("Skip", self.frame_skip_var),
                ("Max seconds", self.max_seconds_var),
                ("Frames/sec", self.analysis_fps_var),
                ("Targets", self.target_frames_var),
                ("YOLO conf", self.low_res_det_conf_var),
            )
            for column, (label, variable) in enumerate(compact_params):
                grid_column = column % 2
                grid_row = column // 2
                params_grid.grid_columnconfigure(grid_column, weight=1, uniform="params")
                cell = tk.Frame(params_grid, bg=self.colors["card"])
                cell.grid(
                    row=grid_row,
                    column=grid_column,
                    sticky="ew",
                    padx=(0 if grid_column == 0 else 6, 0),
                    pady=(0 if grid_row == 0 else 3, 0),
                )
                tk.Label(
                    cell,
                    text=label,
                    bg=self.colors["card"],
                    fg=self.colors["muted"],
                    font=self.font_small_bold,
                ).pack(anchor="w")
                tk.Entry(
                    cell,
                    textvariable=variable,
                    bg=self.colors["glass"],
                    fg=self.colors["text"],
                    relief=tk.FLAT,
                    highlightbackground=self.colors["glass_border"],
                    highlightcolor=self.colors["accent"],
                    highlightthickness=1,
                    insertbackground=self.colors["text"],
                    font=self.font_code_small,
                    width=7,
                    cursor="xterm",
                ).pack(fill=tk.X, pady=(1, 0))
            full_video_check = ttk.Checkbutton(
                params_grid,
                text="Full video",
                variable=self.full_video_var,
                style="App.TCheckbutton",
                cursor=ACTION_CURSOR,
            )
            full_video_check.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(16, 0))
        else:
            self.create_entry_row(params_grid, 0, "Frame skip", self.frame_skip_var)
            self.create_entry_row(params_grid, 1, "Max seconds", self.max_seconds_var)
            self.create_entry_row(params_grid, 2, "Frames/sec override", self.analysis_fps_var)
            self.create_entry_row(params_grid, 3, "Target frames", self.target_frames_var)
            self.create_entry_row(params_grid, 4, "Low-res YOLO conf", self.low_res_det_conf_var)
            ttk.Checkbutton(
                params_grid,
                text="Process full video",
                variable=self.full_video_var,
                style="App.TCheckbutton",
                cursor=ACTION_CURSOR,
            ).grid(row=5, column=1, sticky="w", pady=(0, 8))

        self.build_subsection_label(
            setup_panel,
            "Paths",
            "Project folder, model files, and output location.",
        )
        path_block = tk.Frame(setup_panel, bg=self.colors["card"])
        path_block.pack(fill=tk.X, pady=(0, 2 if self.compact_figure_mode else 14))

        self.pipeline_path_label = self.create_path_row(
            path_block,
            "Pipeline folder",
            "[Not selected yet]",
            self.select_pipeline_folder,
            "Browse",
            extra_button=("Use Project Folder", self.use_project_folder),
        )
        self.output_path_label = self.create_path_row(
            path_block,
            "Output folder",
            "[Not selected yet]",
            self.select_output_folder,
            "Browse",
            extra_button=("Use Default Output", self.use_default_output_folder),
        )

        if self.compact_figure_mode:
            self.python_combo = ttk.Combobox(
                setup_panel,
                textvariable=self.selected_python_var,
                style="App.TCombobox",
                state="readonly",
                height=10,
                cursor=DEFAULT_CURSOR,
            )
            return

        self.build_subsection_label(
            setup_panel,
            "Python Environment",
            "Interpreter with the required analysis dependencies.",
        )
        env_block = tk.Frame(setup_panel, bg=self.colors["card"])
        env_block.pack(fill=tk.X)
        self.python_combo = ttk.Combobox(
            env_block,
            textvariable=self.selected_python_var,
            style="App.TCombobox",
            state="readonly",
            height=10,
            cursor=DEFAULT_CURSOR,
        )
        self.set_hover_cursor(self.python_combo, ACTION_CURSOR)
        self.python_combo.pack(fill=tk.X, pady=(0, 8))
        self.python_combo.bind("<<ComboboxSelected>>", self.on_python_selection_changed)

        env_buttons = tk.Frame(env_block, bg=self.colors["card"])
        env_buttons.pack(fill=tk.X)
        env_buttons.grid_columnconfigure(0, weight=1)
        env_buttons.grid_columnconfigure(1, weight=1)
        self.create_button(
            env_buttons,
            "Rescan Env",
            self.refresh_python_candidates,
            height=36,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.create_button(
            env_buttons,
            "Browse Python",
            self.select_python_executable,
            height=36,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def build_status_card_contents(self):
        stage_wrap = tk.Frame(
            self.status_card_body,
            bg=self.colors["glass"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        stage_wrap.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            stage_wrap,
            text="Current stage",
            bg=self.colors["glass"],
            fg=self.colors["muted"],
            font=self.font_small_bold,
            padx=12,
            pady=8,
        ).pack(anchor="w")
        tk.Label(
            stage_wrap,
            textvariable=self.stage_var,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            font=self.font_body_lg,
            padx=12,
            pady=0,
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 12))

        action_wrap = tk.Frame(self.status_card_body, bg=self.colors["card"])
        action_wrap.pack(fill=tk.X, pady=(0, 10))
        action_wrap.grid_columnconfigure(0, weight=1)
        action_wrap.grid_columnconfigure(1, weight=1)
        self.create_button(
            action_wrap,
            "Run Preflight",
            self.run_preflight_check,
            height=38,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.create_button(
            action_wrap,
            "Refresh Checks",
            self.run_quick_status_refresh,
            height=38,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress = ttk.Progressbar(
            self.status_card_body,
            mode="indeterminate",
            style="Run.Horizontal.TProgressbar",
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))

        checks_frame = tk.Frame(self.status_card_body, bg=self.colors["card"])
        checks_frame.pack(fill=tk.BOTH, expand=True)
        self.checks_tree = ttk.Treeview(
            checks_frame,
            columns=("status", "item", "detail"),
            show="headings",
            style="Checks.Treeview",
            cursor=DEFAULT_CURSOR,
            height=7,
        )
        self.set_hover_cursor(self.checks_tree, ACTION_CURSOR)
        self.checks_tree.heading("status", text="Status")
        self.checks_tree.heading("item", text="Check")
        self.checks_tree.heading("detail", text="Detail")
        self.checks_tree.column("status", width=90, anchor=tk.CENTER)
        self.checks_tree.column("item", width=170, anchor=tk.W)
        self.checks_tree.column("detail", width=360, anchor=tk.W)
        self.checks_tree.tag_configure("ok", foreground=self.colors["success"])
        self.checks_tree.tag_configure("warn", foreground=self.colors["warning"])
        self.checks_tree.tag_configure("fail", foreground=self.colors["danger"])
        self.checks_tree.tag_configure("info", foreground=self.colors["muted"])
        checks_scroll = ttk.Scrollbar(
            checks_frame,
            orient=tk.VERTICAL,
            command=self.checks_tree.yview,
            style="App.Vertical.TScrollbar",
            cursor=V_SCROLL_CURSOR,
        )
        self.checks_tree.configure(yscroll=checks_scroll.set)
        self.checks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        checks_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def build_results_card_contents(self):
        self.build_subsection_label(
            self.results_card_body,
            "Latest Launcher Script",
            "Each run generates a launcher script for reproducibility.",
        )
        self.script_label = tk.Label(
            self.results_card_body,
            textvariable=self.script_summary_var,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            justify=tk.LEFT,
            wraplength=620,
            padx=12,
            pady=12,
            font=self.font_code_small,
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        self.script_label.pack(fill=tk.X, pady=(0, 14))

        self.build_subsection_label(
            self.results_card_body,
            "Latest Output Summary",
            "After each run, this panel summarizes new files and the output location.",
        )
        self.results_label = tk.Label(
            self.results_card_body,
            textvariable=self.output_summary_var,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            justify=tk.LEFT,
            wraplength=620,
            padx=12,
            pady=12,
            font=self.font_body,
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        self.results_label.pack(fill=tk.X)

        results_actions = tk.Frame(self.results_card_body, bg=self.colors["card"])
        results_actions.pack(fill=tk.X, pady=(14, 0))
        results_actions.grid_columnconfigure(0, weight=1)
        results_actions.grid_columnconfigure(1, weight=1)
        results_actions.grid_columnconfigure(2, weight=1)
        self.create_button(
            results_actions,
            "Preview Results",
            self.open_results_preview_window,
            height=38,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.create_button(
            results_actions,
            "Open Results",
            self.open_last_results,
            height=38,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.create_button(
            results_actions,
            "Open Scripts",
            lambda: self.open_path(self.script_output_dir),
            height=38,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def build_terminal_card(self, parent=None, row=2, column=0, columnspan=1, pady=(18, 0)):
        parent = parent or self.main
        terminal_body = self.create_card(
            parent,
            row=row,
            column=column,
            title="Run Log",
            description="Live stdout from the generated analysis launcher script.",
            columnspan=columnspan,
            pady=pady,
        )

        if not self.compact_figure_mode:
            terminal_actions = tk.Frame(terminal_body, bg=self.colors["card"])
            terminal_actions.pack(fill=tk.X, pady=(0, 10))
            self.create_button(
                terminal_actions,
                "Clear Log",
                self.clear_terminal,
                width=138,
                height=38,
            ).pack(side=tk.RIGHT)

        terminal_frame = tk.Frame(terminal_body, bg=self.colors["card"])
        terminal_frame.pack(fill=tk.BOTH, expand=True)
        self.terminal = tk.Text(
            terminal_frame,
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border_soft"],
            highlightcolor=self.colors["accent"],
            wrap=tk.WORD,
            height=3 if self.compact_figure_mode else 14,
            font=self.font_code,
            padx=10 if self.compact_figure_mode else 14,
            pady=5 if self.compact_figure_mode else 12,
            cursor="xterm",
        )
        terminal_scroll = ttk.Scrollbar(
            terminal_frame,
            orient=tk.VERTICAL,
            command=self.terminal.yview,
            style="App.Vertical.TScrollbar",
            cursor=V_SCROLL_CURSOR,
        )
        self.terminal.configure(yscrollcommand=terminal_scroll.set)
        self.terminal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        terminal_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.terminal.config(state=tk.DISABLED)

    def create_card(self, parent, row, title, description, column=0, columnspan=1, rowspan=1, pady=(0, 12), padx=(0, 0)):
        card = tk.Frame(
            parent,
            bg=self.colors["card"],
            highlightbackground=self.colors["glass_border"],
            highlightthickness=1,
            bd=0,
        )
        card.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky="nsew", padx=padx, pady=pady)

        header = tk.Frame(card, bg=self.colors["card"])
        header_pad_x = 7 if self.compact_figure_mode else 18
        header_pad_y = (4, 1) if self.compact_figure_mode else (18, 10)
        header.pack(fill=tk.X, padx=header_pad_x, pady=header_pad_y)
        ttk.Label(header, text=title, style="CardTitle.TLabel").pack(anchor="w")
        if description and not self.compact_figure_mode:
            ttk.Label(header, text=description, style="CardMeta.TLabel", wraplength=720, justify=tk.LEFT).pack(
                anchor="w", pady=(4, 0)
            )

        body = tk.Frame(card, bg=self.colors["card"])
        body_pad_x = 7 if self.compact_figure_mode else 18
        body_pad_y = (0, 5) if self.compact_figure_mode else (0, 18)
        body.pack(fill=tk.BOTH, expand=True, padx=body_pad_x, pady=body_pad_y)
        return body

    def build_subsection_label(self, parent, title, description):
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor="w")
        if self.compact_figure_mode:
            return
        ttk.Label(parent, text=description, style="Muted.TLabel", wraplength=320, justify=tk.LEFT).pack(
            anchor="w", pady=(2, 8)
        )

    def create_entry_row(self, parent, row, label, variable):
        label_widget = tk.Label(
            parent,
            text=label,
            bg=self.colors["card"],
            fg=self.colors["text"],
            font=self.font_body,
        )
        label_widget.grid(row=row, column=0, sticky="w", pady=(0, 8))
        entry = tk.Entry(
            parent,
            textvariable=variable,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            relief=tk.FLAT,
            highlightbackground=self.colors["glass_border"],
            highlightcolor=self.colors["accent"],
            highlightthickness=1,
            insertbackground=self.colors["text"],
            font=self.font_code,
            width=18,
            cursor="xterm",
        )
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=(0, 8))
        parent.grid_columnconfigure(1, weight=1)

    def create_path_row(self, parent, title, placeholder, command, button_text, extra_button=None):
        if self.compact_figure_mode:
            block = tk.Frame(
                parent,
                bg=self.colors["glass"],
                highlightthickness=1,
                highlightbackground=self.colors["glass_border"],
            )
            block.pack(fill=tk.X, pady=(0, 5))
            block.grid_columnconfigure(0, weight=1)

            text_block = tk.Frame(block, bg=self.colors["glass"])
            text_block.grid(row=0, column=0, sticky="ew", padx=7, pady=5)
            tk.Label(
                text_block,
                text=title,
                bg=self.colors["glass"],
                fg=self.colors["muted"],
                font=self.font_small_bold,
            ).pack(anchor="w")
            label = tk.Label(
                text_block,
                text=placeholder,
                bg=self.colors["glass"],
                fg=self.colors["text"],
                font=self.font_code_small,
                justify=tk.LEFT,
                anchor="w",
                width=24,
                height=1,
            )
            label.pack(anchor="w", fill=tk.X, pady=(1, 0))

            buttons = tk.Frame(block, bg=self.colors["glass"])
            buttons.grid(row=0, column=1, sticky="e", padx=(0, 7), pady=5)
            if extra_button:
                compact_extra = "Project" if "Project" in extra_button[0] else "Default"
                self.create_button(buttons, compact_extra, extra_button[1], width=58, height=24).pack(
                    side=tk.LEFT, padx=(0, 4)
                )
            self.create_button(buttons, button_text, command, width=58, height=24).pack(side=tk.LEFT)
            return label

        block = tk.Frame(
            parent,
            bg=self.colors["glass"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        block.pack(fill=tk.X, pady=(0, 12))

        top = tk.Frame(block, bg=self.colors["glass"])
        top.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(
            top,
            text=title,
            bg=self.colors["glass"],
            fg=self.colors["muted"],
            font=self.font_small_bold,
        ).pack(anchor="w")

        label = tk.Label(
            block,
            text=placeholder,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            font=self.font_code_small,
            justify=tk.LEFT,
            wraplength=300,
            padx=12,
            pady=10,
        )
        label.pack(fill=tk.X)

        buttons = tk.Frame(block, bg=self.colors["glass"])
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        buttons.grid_columnconfigure(0, weight=1)
        buttons.grid_columnconfigure(1, weight=1)
        if extra_button:
            self.create_button(
                buttons,
                extra_button[0],
                extra_button[1],
                height=36,
            ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
            self.create_button(
                buttons,
                button_text,
                command,
                height=36,
            ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        else:
            self.create_button(
                buttons,
                button_text,
                command,
                height=36,
            ).grid(row=0, column=0, columnspan=2, sticky="ew")
        return label

    def bind_state_events(self):
        self.selected_pipeline_var.trace_add("write", self.on_settings_changed)
        self.selected_python_var.trace_add("write", self.on_settings_changed)
        self.frame_skip_var.trace_add("write", self.on_settings_changed)
        self.max_frame_var.trace_add("write", self.on_settings_changed)
        self.max_seconds_var.trace_add("write", self.on_settings_changed)
        self.analysis_fps_var.trace_add("write", self.on_settings_changed)
        self.target_frames_var.trace_add("write", self.on_settings_changed)
        self.low_res_det_conf_var.trace_add("write", self.on_settings_changed)
        self.tracking_backend_var.trace_add("write", self.on_settings_changed)
        self.full_video_var.trace_add("write", self.on_settings_changed)

    def bring_window_forward(self):
        try:
            self.root.lift()
            self.root.focus_force()
            self.root.attributes("-topmost", True)
            self.root.after(500, lambda: self.root.attributes("-topmost", False))
        except tk.TclError:
            pass

    def on_page_content_configure(self, _event=None):
        self.resize_page_window()
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))
        self.root.after_idle(self.sync_page_scrollbar)

    def on_page_canvas_configure(self, event):
        self.resize_page_window(viewport_width=event.width, viewport_height=event.height)
        self.root.after_idle(self.sync_page_scrollbar)
        if self.preview_capture is not None and self.preview_total_frames > 0:
            current_frame = int(float(self.preview_scrub_var.get()))
            self.root.after_idle(lambda: self.show_preview_frame(current_frame, sync_slider=False))

    def resize_page_window(self, viewport_width=None, viewport_height=None):
        if not hasattr(self, "page_window"):
            return
        width = viewport_width or self.page_canvas.winfo_width()
        height = viewport_height or self.page_canvas.winfo_height()
        requested_height = self.page_content.winfo_reqheight()
        self.page_canvas.itemconfigure(
            self.page_window,
            width=width,
            height=max(height, requested_height),
        )

    def sync_page_scrollbar(self):
        if not hasattr(self, "page_scroll"):
            return
        bbox = self.page_canvas.bbox("all")
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        viewport_height = self.page_canvas.winfo_height()
        needs_scroll = content_height > viewport_height + 2
        if needs_scroll and not self.page_scroll_visible:
            self.page_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.page_scroll_visible = True
        elif not needs_scroll and self.page_scroll_visible:
            self.page_scroll.pack_forget()
            self.page_scroll_visible = False
            self.page_canvas.yview_moveto(0)

    def on_global_mousewheel(self, event):
        widget_class = event.widget.winfo_class()
        if widget_class in {"Text", "Entry", "TCombobox", "Treeview", "Scrollbar", "Scale"}:
            return
        if not getattr(self, "page_scroll_visible", False):
            return
        direction = -1 if event.delta > 0 else 1
        self.page_canvas.yview_scroll(direction, "units")

    def load_session(self):
        if not os.path.exists(self.session_path):
            return {}
        try:
            with open(self.session_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_session(self):
        data = {
            "selected_folders": [path for path in self.selected_folders if os.path.isdir(path)],
            "pinned_video_paths": [
                path for path in self.pinned_video_paths if os.path.isfile(path) and path.lower().endswith(".mp4")
            ],
            "pipeline_dir": self.pipeline_dir,
            "output_dir": self.output_dir,
            "python_executable": self.selected_python_var.get().strip(),
            "selected_pipeline": self.selected_pipeline_var.get().strip(),
            "frame_skip": self.frame_skip_var.get().strip(),
            "max_frame": self.max_frame_var.get().strip(),
            "max_seconds": self.max_seconds_var.get().strip(),
            "analysis_fps": self.analysis_fps_var.get().strip(),
            "target_frames": self.target_frames_var.get().strip(),
            "low_res_det_conf": self.low_res_det_conf_var.get().strip(),
            "tracking_backend": self.tracking_backend_var.get().strip(),
            "full_video": bool(self.full_video_var.get()),
        }
        try:
            with open(self.session_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except OSError:
            pass

    def restore_session_state(self):
        if not self.session_data:
            return

        self.selected_folders = [
            os.path.normpath(path)
            for path in self.session_data.get("selected_folders", [])
            if os.path.isdir(path)
        ]
        self.pinned_video_paths = [
            os.path.normpath(path)
            for path in self.session_data.get("pinned_video_paths", [])
            if os.path.isfile(path) and path.lower().endswith(".mp4")
        ]
        self.frame_skip_var.set(self.session_data.get("frame_skip", "2"))
        self.max_frame_var.set(self.session_data.get("max_frame", "480"))
        self.max_seconds_var.set(self.session_data.get("max_seconds", "120"))
        self.analysis_fps_var.set(self.session_data.get("analysis_fps", ""))
        self.target_frames_var.set(self.session_data.get("target_frames", ""))
        self.low_res_det_conf_var.set(self.session_data.get("low_res_det_conf", "auto"))
        self.tracking_backend_var.set(self.session_data.get("tracking_backend", "cellpose"))
        self.full_video_var.set(bool(self.session_data.get("full_video", False)))

        pipeline_dir = self.session_data.get("pipeline_dir")
        if pipeline_dir and os.path.isdir(pipeline_dir):
            self.set_pipeline_dir(pipeline_dir, save=False)

        output_dir = self.session_data.get("output_dir")
        if output_dir and self.is_legacy_default_output_directory(output_dir):
            output_dir = self.default_output_directory()
        if output_dir and os.path.isdir(output_dir):
            self.set_output_dir(output_dir, save=False)

        saved_python = self.session_data.get("python_executable")
        if saved_python:
            self.python_executable = saved_python
            self.selected_python_var.set(saved_python)

        saved_pipeline = self.session_data.get("selected_pipeline")
        if saved_pipeline:
            self.selected_pipeline_var.set(saved_pipeline)

    def autodetect_project_context(self):
        if not self.pipeline_dir:
            project_dir = self.find_pipeline_directory(self.cwd)
            if project_dir:
                self.set_pipeline_dir(project_dir, save=False)

        if not self.output_dir:
            self.set_output_dir(self.default_output_directory(), save=False)

    def default_output_directory(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_default")

    def is_legacy_default_output_directory(self, path):
        legacy = os.path.join(os.path.expanduser("~"), "Downloads", "SickleSight-output")
        return os.path.normpath(path) == os.path.normpath(legacy)

    def find_pipeline_directory(self, start_dir):
        candidate_dirs = [start_dir, os.path.dirname(os.path.abspath(__file__))]
        for candidate in candidate_dirs:
            if not candidate:
                continue
            if all(
                os.path.exists(os.path.join(candidate, filename))
                for filename in ("sicklesight_part1.py", "sicklesight_part2.py", "sicklesight_merged.py")
            ):
                return os.path.normpath(candidate)
        return None

    def refresh_python_candidates(self):
        candidates = []

        def add_candidate(path):
            if not path:
                return
            normalized = os.path.normpath(path)
            if normalized not in candidates and os.path.isfile(normalized):
                candidates.append(normalized)

        add_candidate(self.selected_python_var.get())
        add_candidate(self.python_executable)
        add_candidate(sys.executable)

        relative_candidates = [
            os.path.join(self.cwd, ".miniforge3", "envs", "sicklesight", "bin", "python"),
            os.path.join(self.cwd, ".venv", "bin", "python"),
            os.path.join(self.cwd, "venv", "bin", "python"),
        ]
        if self.is_windows:
            relative_candidates = [
                os.path.join(self.cwd, ".miniforge3", "envs", "sicklesight", "python.exe"),
                os.path.join(self.cwd, ".venv", "Scripts", "python.exe"),
                os.path.join(self.cwd, "venv", "Scripts", "python.exe"),
            ]
        for path in relative_candidates:
            add_candidate(path)

        add_candidate(shutil.which("python3"))
        add_candidate(shutil.which("python"))

        self.available_python_options = candidates
        self.python_combo["values"] = candidates

        if candidates:
            current = self.selected_python_var.get()
            preferred = self.choose_preferred_python_candidate(candidates)
            if current not in candidates or self.should_replace_python_choice(current, preferred):
                self.selected_python_var.set(preferred)
                self.python_executable = preferred
        self.run_quick_status_refresh()

    def choose_preferred_python_candidate(self, candidates):
        for path in candidates:
            normalized = path.replace("\\", "/")
            if "/.miniforge3/envs/sicklesight/" in normalized:
                return path
        for path in candidates:
            normalized = path.replace("\\", "/")
            if "/.venv/" in normalized or "/venv/" in normalized:
                return path
        return candidates[0]

    def should_replace_python_choice(self, current, preferred):
        if not current or current == preferred:
            return False
        normalized = current.replace("\\", "/")
        if "Python.framework/Versions/3.13" in normalized:
            return True
        if normalized == sys.executable.replace("\\", "/"):
            return True
        if normalized.endswith("/python3") or normalized.endswith("/python"):
            return True
        return False

    def apply_startup_paths(self, video_paths, folder_paths):
        folders_to_add = []
        for folder in folder_paths:
            if os.path.isdir(folder):
                folders_to_add.append(os.path.normpath(folder))

        for video_path in video_paths:
            if os.path.isfile(video_path) and video_path.lower().endswith(".mp4"):
                folders_to_add.append(os.path.normpath(os.path.dirname(video_path)))

        if video_paths:
            self.preferred_video_paths = [
                os.path.normpath(path)
                for path in video_paths
                if os.path.isfile(path) and path.lower().endswith(".mp4")
            ]
            self.pinned_video_paths = sorted(set(self.pinned_video_paths).union(self.preferred_video_paths))

        for folder in folders_to_add:
            if folder not in self.selected_folders:
                self.selected_folders.append(folder)

        self.refresh_tree()
        if self.preferred_video_paths:
            self.select_first_matching_video(self.preferred_video_paths)

    def apply_startup_config(self, pipeline_dir, output_dir):
        if pipeline_dir and os.path.isdir(pipeline_dir):
            self.set_pipeline_dir(pipeline_dir, save=False)
        if output_dir:
            self.set_output_dir(output_dir, save=False)
        self.save_session()

    def add_folders(self):
        parent_path = filedialog.askdirectory(title="Choose a parent folder with .mp4 files")
        if not parent_path:
            return

        added_any = False
        for root_dir, _, files in os.walk(parent_path):
            if any(filename.lower().endswith(".mp4") for filename in files):
                clean_path = os.path.normpath(root_dir)
                if clean_path not in self.selected_folders:
                    self.selected_folders.append(clean_path)
                    added_any = True

        if not added_any:
            messagebox.showinfo(
                "No videos found",
                "The selected directory does not contain any nested folders with .mp4 files.",
            )
        self.refresh_tree()
        self.save_session()

    def add_video_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose video files",
            filetypes=[("MP4 videos", "*.mp4"), ("All files", "*.*")],
        )
        valid_paths = [
            os.path.normpath(path)
            for path in paths
            if os.path.isfile(path) and path.lower().endswith(".mp4")
        ]
        if not valid_paths:
            return

        for path in valid_paths:
            folder = os.path.dirname(path)
            if folder not in self.selected_folders:
                self.selected_folders.append(folder)
        self.preferred_video_paths = valid_paths
        self.pinned_video_paths = sorted(set(self.pinned_video_paths).union(valid_paths))
        self.refresh_tree()
        self.select_first_matching_video(valid_paths)
        self.save_session()

    def clear_all_sources(self):
        if not self.selected_folders:
            return
        if messagebox.askyesno("Clear sources", "Remove all folders and videos from the current pool?"):
            self.selected_folders.clear()
            self.preferred_video_paths.clear()
            self.pinned_video_paths.clear()
            self.refresh_tree()
            self.save_session()

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        grouped = {}
        for folder in sorted(self.selected_folders):
            parent = os.path.dirname(folder)
            grouped.setdefault(parent, []).append(folder)

        for parent_path, folders in grouped.items():
            parent_node = self.tree.insert("", tk.END, text=parent_path, open=True, values=(parent_path,))
            for folder in folders:
                folder_name = os.path.basename(folder)
                folder_node = self.tree.insert(parent_node, tk.END, text=folder_name, open=False, values=(folder,))
                try:
                    files = sorted(
                        filename for filename in os.listdir(folder) if filename.lower().endswith(".mp4")
                    )
                except OSError:
                    files = []
                for filename in files:
                    full_path = os.path.join(folder, filename)
                    self.tree.insert(folder_node, tk.END, text=filename, values=(full_path,))

        self.selection_count_var.set(
            f"{len(self.selected_folders)} folders in pool, {len(self.collect_all_video_files())} videos available, "
            f"{len(self.pinned_video_paths)} pinned for default runs"
        )
        self.run_quick_status_refresh()

    def select_first_matching_video(self, video_paths):
        normalized_targets = {os.path.normpath(path) for path in video_paths}
        for item in self.walk_tree(self.tree.get_children()):
            values = self.tree.item(item, "values")
            if not values:
                continue
            item_path = os.path.normpath(values[0])
            if item_path in normalized_targets:
                self.tree.selection_set(item)
                self.tree.focus(item)
                self.tree.see(item)
                self.update_preview_for_path(item_path)
                return

    def walk_tree(self, items):
        for item in items:
            yield item
            yield from self.walk_tree(self.tree.get_children(item))

    def remove_selection(self):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        folders_to_remove = set()
        for item in selected_items:
            values = self.tree.item(item, "values")
            if not values:
                continue
            path = values[0]
            if path in self.selected_folders:
                folders_to_remove.add(path)
            elif os.path.isfile(path):
                folders_to_remove.add(os.path.dirname(path))

        if folders_to_remove:
            self.selected_folders = [folder for folder in self.selected_folders if folder not in folders_to_remove]
            self.pinned_video_paths = [
                path for path in self.pinned_video_paths if os.path.dirname(path) not in folders_to_remove
            ]
            self.refresh_tree()
            self.save_session()

    def on_select_file(self, _event):
        selected = self.tree.selection()
        if not selected:
            return
        values = self.tree.item(selected[0], "values")
        if not values:
            return
        self.update_preview_for_path(values[0])

    def update_preview_for_path(self, path):
        lines = []
        if os.path.isfile(path):
            stat_result = os.stat(path)
            lines.extend(
                [
                    f"Selected video: {os.path.basename(path)}",
                    f"Path: {path}",
                    f"Size: {stat_result.st_size / (1024 * 1024):.2f} MB",
                    f"Modified: {datetime.datetime.fromtimestamp(stat_result.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Parent folder: {os.path.dirname(path)}",
                ]
            )
            self.prepare_video_preview(path)
        elif os.path.isdir(path):
            try:
                files = [filename for filename in os.listdir(path) if filename.lower().endswith(".mp4")]
            except OSError:
                files = []
            lines.extend(
                [
                    f"Selected folder: {os.path.basename(path) or path}",
                    f"Path: {path}",
                    f"Videos inside folder: {len(files)}",
                    "Tip: highlight specific .mp4 files to run a narrower job.",
                ]
            )
            self.reset_inline_preview(
                f"{len(files)} videos found in this folder. Select a single .mp4 file to load a frame preview here."
            )
        else:
            lines.append(path)
            self.reset_inline_preview("Preview is unavailable for this selection.")

        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "\n".join(lines))
        self.preview_text.config(state=tk.DISABLED)

    def reset_inline_preview(self, message):
        self.release_preview_capture()
        self.preview_photo = None
        self.preview_video_path = ""
        self.preview_total_frames = 0
        self.preview_fps = 0.0
        self.preview_width = 0
        self.preview_height = 0
        self.preview_zoom_index = 0
        self.update_preview_zoom_label()
        self.preview_visual.config(image="", text="Inline preview will appear here")
        self.preview_status_var.set(message)
        self.preview_scrub_var.set(0)
        self.preview_scale.config(from_=0, to=100, state=tk.DISABLED)

    def release_preview_capture(self):
        if self.preview_slider_job is not None:
            try:
                self.root.after_cancel(self.preview_slider_job)
            except ValueError:
                pass
            self.preview_slider_job = None
        if self.preview_capture is not None:
            try:
                self.preview_capture.release()
            except Exception:
                pass
            self.preview_capture = None

    def prepare_video_preview(self, path):
        if Image is None or ImageTk is None:
            self.reset_inline_preview("Inline preview needs Pillow in the selected Python environment.")
            return

        try:
            import cv2
        except ImportError:
            self.reset_inline_preview("Inline preview needs OpenCV in the selected Python environment.")
            return

        self.release_preview_capture()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self.reset_inline_preview("Could not open this video for inline preview.")
            return

        self.preview_capture = capture
        self.preview_video_path = path
        self.preview_total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
        self.preview_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.preview_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.preview_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.preview_zoom_index = 0
        self.update_preview_zoom_label()

        self.preview_scale.config(
            from_=0,
            to=max(self.preview_total_frames - 1, 0),
            state=tk.NORMAL if self.preview_total_frames > 1 else tk.DISABLED,
        )
        self.show_preview_frame(0)

    def on_preview_slider_changed(self, value):
        if self.preview_capture is None:
            return
        frame_index = int(float(value))
        if self.preview_slider_job is not None:
            try:
                self.root.after_cancel(self.preview_slider_job)
            except ValueError:
                pass
        self.preview_slider_job = self.root.after(60, lambda: self.show_preview_frame(frame_index, sync_slider=False))

    def preview_first_frame(self):
        if self.preview_capture is not None:
            self.show_preview_frame(0)

    def preview_middle_frame(self):
        if self.preview_capture is not None:
            self.show_preview_frame(max(self.preview_total_frames // 2, 0))

    def preview_last_frame(self):
        if self.preview_capture is not None:
            self.show_preview_frame(max(self.preview_total_frames - 1, 0))

    def current_preview_zoom(self):
        return self.preview_zoom_levels[self.preview_zoom_index]

    def update_preview_zoom_label(self):
        zoom = self.current_preview_zoom()
        if zoom == 1.0:
            self.preview_zoom_var.set("1x")
        else:
            self.preview_zoom_var.set(f"{zoom:g}x")

    def rerender_current_preview_frame(self):
        if self.preview_capture is None:
            return
        current_frame = int(float(self.preview_scrub_var.get()))
        self.show_preview_frame(current_frame, sync_slider=False)

    def preview_zoom_in(self):
        if self.preview_zoom_index < len(self.preview_zoom_levels) - 1:
            self.preview_zoom_index += 1
            self.update_preview_zoom_label()
            self.rerender_current_preview_frame()

    def preview_zoom_out(self):
        if self.preview_zoom_index > 0:
            self.preview_zoom_index -= 1
            self.update_preview_zoom_label()
            self.rerender_current_preview_frame()

    def preview_zoom_fit(self):
        self.preview_zoom_index = 0
        self.update_preview_zoom_label()
        self.rerender_current_preview_frame()

    def save_current_preview_frame(self):
        if self.preview_capture is None or not self.preview_video_path:
            messagebox.showinfo("No frame loaded", "Select a video and load a preview frame first.")
            return

        try:
            import cv2
        except ImportError:
            messagebox.showerror("Save failed", "Saving frames requires OpenCV in the selected Python environment.")
            return

        frame_index = int(float(self.preview_scrub_var.get()))
        frame_index = max(0, min(frame_index, max(self.preview_total_frames - 1, 0)))
        self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.preview_capture.read()
        if not ok or frame is None:
            messagebox.showerror("Save failed", "Could not decode the current frame.")
            return

        video_name = os.path.splitext(os.path.basename(self.preview_video_path))[0] or "preview"
        default_name = f"{video_name}_frame_{frame_index + 1:04d}.png"
        initial_dir = self.output_dir if self.output_dir and os.path.isdir(self.output_dir) else ""
        if not initial_dir and self.preview_video_path:
            initial_dir = os.path.dirname(self.preview_video_path)
        if not initial_dir:
            initial_dir = os.path.expanduser("~/Downloads")

        save_path = filedialog.asksaveasfilename(
            title="Save current frame",
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"), ("All files", "*.*")),
        )
        if not save_path:
            self.show_preview_frame(frame_index, sync_slider=False)
            return
        if not save_path.lower().endswith(".png"):
            save_path += ".png"

        if not cv2.imwrite(save_path, frame):
            messagebox.showerror("Save failed", f"Could not write the frame to:\n{save_path}")
            self.show_preview_frame(frame_index, sync_slider=False)
            return

        self.show_preview_frame(frame_index, sync_slider=False)
        self.log_to_terminal(f"Saved frame {frame_index + 1} to {save_path}")
        messagebox.showinfo("Frame saved", f"Saved current frame:\n{save_path}")

    def build_zoomed_preview_image(self, image, target_width, target_height, resampling):
        zoom = self.current_preview_zoom()
        source_width, source_height = image.size
        base_scale = min(target_width / source_width, target_height / source_height)
        scale = max(base_scale * zoom, 0.01)
        resized_width = max(int(source_width * scale), 1)
        resized_height = max(int(source_height * scale), 1)
        resized = image.resize((resized_width, resized_height), resampling)

        canvas = Image.new("RGB", (target_width, target_height), self.colors["terminal_bg"])
        if resized_width > target_width or resized_height > target_height:
            left = max((resized_width - target_width) // 2, 0)
            top = max((resized_height - target_height) // 2, 0)
            cropped = resized.crop((left, top, left + min(target_width, resized_width), top + min(target_height, resized_height)))
            paste_x = max((target_width - cropped.width) // 2, 0)
            paste_y = max((target_height - cropped.height) // 2, 0)
            canvas.paste(cropped, (paste_x, paste_y))
        else:
            paste_x = (target_width - resized_width) // 2
            paste_y = (target_height - resized_height) // 2
            canvas.paste(resized, (paste_x, paste_y))
        return canvas

    def show_preview_frame(self, frame_index, sync_slider=True):
        if self.preview_capture is None or Image is None or ImageTk is None:
            return

        try:
            import cv2
        except ImportError:
            self.reset_inline_preview("Inline preview needs OpenCV in the selected Python environment.")
            return

        frame_index = max(0, min(frame_index, max(self.preview_total_frames - 1, 0)))
        self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.preview_capture.read()
        if not ok or frame is None:
            self.preview_status_var.set("Could not decode the requested frame for preview.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        min_preview_width = 300 if self.compact_figure_mode else 360
        min_preview_height = 120 if self.compact_figure_mode else 260
        target_width = max(self.preview_frame_container.winfo_width() - 28, min_preview_width)
        target_height = max(self.preview_frame_container.winfo_height() - 28, min_preview_height)
        image = self.build_zoomed_preview_image(image, target_width, target_height, resampling)
        photo = ImageTk.PhotoImage(image)
        self.preview_photo = photo
        self.preview_visual.config(image=photo, text="")

        if sync_slider and int(float(self.preview_scrub_var.get())) != frame_index:
            self.preview_scrub_var.set(frame_index)

        timestamp_seconds = (frame_index / self.preview_fps) if self.preview_fps else 0.0
        self.preview_status_var.set(
            f"Frame {frame_index + 1} of {self.preview_total_frames}  |  "
            f"{timestamp_seconds:0.2f}s  |  "
            f"{self.preview_width}x{self.preview_height}  |  "
            f"{self.preview_fps:0.2f} fps  |  "
            f"Zoom {self.preview_zoom_var.get()}"
        )

    def on_double_click_file(self, _event):
        self.play_video()

    def play_video(self):
        path = self.get_first_selected_path()
        if not path or not os.path.isfile(path) or not path.lower().endswith(".mp4"):
            messagebox.showinfo("No video selected", "Choose an .mp4 file from the pool first.")
            return
        self.open_path(path)

    def open_selected_location(self):
        path = self.get_first_selected_path()
        if not path:
            return
        if os.path.isfile(path):
            path = os.path.dirname(path)
        self.open_path(path)

    def get_first_selected_path(self):
        selected = self.tree.selection()
        if not selected:
            return ""
        values = self.tree.item(selected[0], "values")
        return values[0] if values else ""

    def select_pipeline_folder(self):
        path = filedialog.askdirectory(title="Choose the SickleSight pipeline folder")
        if path:
            self.set_pipeline_dir(path)

    def set_pipeline_dir(self, path, save=True):
        self.pipeline_dir = os.path.normpath(path)
        self.pipeline_path_label.config(text=self.pipeline_dir)
        self.populate_pipeline_options(self.pipeline_dir)
        self.run_quick_status_refresh()
        if save:
            self.save_session()

    def use_project_folder(self):
        project_dir = self.find_pipeline_directory(self.cwd)
        if project_dir:
            self.set_pipeline_dir(project_dir)
        else:
            messagebox.showwarning(
                "Project folder not found",
                "Could not find a folder with sicklesight_part1.py, sicklesight_part2.py, and sicklesight_merged.py.",
            )

    def populate_pipeline_options(self, path):
        for widget in self.pipeline_options_frame.winfo_children():
            widget.destroy()

        available_scripts = []
        if os.path.isdir(path):
            for filename, description in PIPELINE_OPTIONS:
                if os.path.exists(os.path.join(path, filename)):
                    available_scripts.append((filename, description))

        previous_selection = self.selected_pipeline_var.get()
        self.selected_pipeline_var.set("")

        for filename, description in available_scripts:
            row = tk.Frame(
                self.pipeline_options_frame,
                bg=self.colors["glass"],
                highlightthickness=1,
                highlightbackground=self.colors["glass_border"],
            )
            row.pack(fill=tk.X, pady=(0, 3 if self.compact_figure_mode else 8))
            self.set_hover_cursor(row, ACTION_CURSOR)
            radio = ttk.Radiobutton(
                row,
                text=filename,
                value=filename,
                variable=self.selected_pipeline_var,
                style="App.TRadiobutton",
            )
            self.set_hover_cursor(radio, ACTION_CURSOR)
            radio.pack(anchor="w", padx=9 if self.compact_figure_mode else 12, pady=(3, 3) if self.compact_figure_mode else (10, 0))
            if self.compact_figure_mode:
                continue
            description_label = tk.Label(
                row,
                text=description,
                bg=self.colors["glass"],
                fg=self.colors["muted"],
                wraplength=300,
                justify=tk.LEFT,
                font=self.font_small,
                padx=34,
                pady=0,
            )
            self.set_hover_cursor(description_label, ACTION_CURSOR)
            description_label.pack(anchor="w", pady=(2, 10))

        if available_scripts:
            preferred = "sicklesight_merged.py"
            valid_names = [filename for filename, _ in available_scripts]
            if previous_selection in valid_names:
                self.selected_pipeline_var.set(previous_selection)
            elif preferred in valid_names:
                self.selected_pipeline_var.set(preferred)
            else:
                self.selected_pipeline_var.set(valid_names[0])
        else:
            message = (
                "No SickleSight pipeline scripts were found in this folder.\n\n"
                "Expected at least one of:\n"
                "  - sicklesight_merged.py\n"
                "  - sicklesight_part1.py\n"
                "  - sicklesight_part2.py"
            )
            self.stage_var.set("Pipeline folder needs attention")
            self.render_checks(
                [
                    {
                        "status": "FAIL",
                        "item": "Pipeline scripts",
                        "detail": message.replace("\n", " "),
                    }
                ]
            )

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Choose where analysis results should be written")
        if path:
            self.set_output_dir(path)

    def set_output_dir(self, path, save=True):
        if not path:
            return
        self.output_dir = os.path.normpath(path)
        self.output_path_label.config(text=self.output_dir)
        self.run_quick_status_refresh()
        self.refresh_inline_results_preview()
        if save:
            self.save_session()

    def use_default_output_folder(self):
        self.set_output_dir(self.default_output_directory())

    def select_python_executable(self):
        filetypes = [("Python executables", "python*"), ("All files", "*.*")]
        if self.is_windows:
            filetypes = [("Python executables", "*.exe"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Choose a Python interpreter", filetypes=filetypes)
        if path:
            normalized = os.path.normpath(path)
            self.python_executable = normalized
            if normalized not in self.available_python_options:
                self.available_python_options.insert(0, normalized)
                self.python_combo["values"] = self.available_python_options
            self.selected_python_var.set(normalized)
            self.save_session()

    def on_python_selection_changed(self, _event):
        self.python_executable = self.selected_python_var.get().strip()
        self.run_quick_status_refresh()
        self.save_session()

    def on_settings_changed(self, *_args):
        selected_python = self.selected_python_var.get().strip()
        if selected_python:
            self.python_executable = selected_python
        self.run_quick_status_refresh()
        self.save_session()

    def collect_all_video_files(self):
        target_files = []
        for folder in self.selected_folders:
            try:
                files = [
                    os.path.join(folder, filename)
                    for filename in os.listdir(folder)
                    if filename.lower().endswith(".mp4")
                ]
            except OSError:
                files = []
            target_files.extend(files)
        return sorted(set(target_files))

    def collect_target_files(self):
        selected_items = self.tree.selection()
        target_files = set()

        if selected_items:
            for item in selected_items:
                values = self.tree.item(item, "values")
                if not values:
                    continue
                path = values[0]
                if os.path.isfile(path) and path.lower().endswith(".mp4"):
                    target_files.add(os.path.normpath(path))
                elif os.path.isdir(path):
                    try:
                        files = [
                            os.path.join(path, filename)
                            for filename in os.listdir(path)
                            if filename.lower().endswith(".mp4")
                        ]
                    except OSError:
                        files = []
                    for file_path in files:
                        target_files.add(os.path.normpath(file_path))

        if not target_files:
            if self.pinned_video_paths:
                target_files.update(
                    path for path in self.pinned_video_paths
                    if os.path.isfile(path) and path.lower().endswith(".mp4")
                )
            else:
                target_files.update(self.collect_all_video_files())

        return sorted(target_files)

    def missing_model_files(self):
        if not self.pipeline_dir or not os.path.isdir(self.pipeline_dir):
            expected = MODEL_FILES[:]
            if self.tracking_backend_var.get() == "low_res":
                expected.extend(LOW_RES_MODEL_FILES)
            return expected
        expected = MODEL_FILES[:]
        if self.tracking_backend_var.get() == "low_res":
            expected.extend(LOW_RES_MODEL_FILES)
        return [
            filename
            for filename in expected
            if not os.path.exists(os.path.join(self.pipeline_dir, filename))
        ]

    def current_pipeline_scripts(self):
        return [
            filename
            for filename, _ in PIPELINE_OPTIONS
            if self.pipeline_dir and os.path.exists(os.path.join(self.pipeline_dir, filename))
        ]

    def run_quick_status_refresh(self):
        checks = self.build_quick_checks()
        self.latest_checks = checks
        self.render_checks(checks)
        self.refresh_badges(checks)

    def build_quick_checks(self):
        checks = []

        target_files = self.collect_target_files()
        if target_files:
            checks.append(
                {
                    "status": "OK",
                    "item": "Input videos",
                    "detail": f"{len(target_files)} videos ready for processing.",
                }
            )
        else:
            checks.append(
                {
                    "status": "WARN",
                    "item": "Input videos",
                    "detail": "Add folders or select videos to build the analysis pool.",
                }
            )

        scripts = self.current_pipeline_scripts()
        if scripts:
            checks.append(
                {
                    "status": "OK",
                    "item": "Pipeline scripts",
                    "detail": f"Found {len(scripts)} SickleSight scripts in the selected project folder.",
                }
            )
        else:
            checks.append(
                {
                    "status": "FAIL",
                    "item": "Pipeline scripts",
                    "detail": "Choose the project folder that contains sicklesight_part1.py, sicklesight_part2.py, and sicklesight_merged.py.",
                }
            )

        if self.tracking_backend_var.get() == "low_res":
            low_res_backend = os.path.join(self.pipeline_dir, "low_res_backend.py") if self.pipeline_dir else ""
            if low_res_backend and os.path.exists(low_res_backend):
                checks.append(
                    {
                        "status": "OK",
                        "item": "Low-res backend",
                        "detail": "low_res_backend.py is available for YOLO/BoT-SORT processing.",
                    }
                )
            else:
                checks.append(
                    {
                        "status": "FAIL",
                        "item": "Low-res backend",
                        "detail": "low_res_backend.py is required when Low-resolution YOLO/BoT-SORT is selected.",
                    }
                )

        missing_models = self.missing_model_files()
        if missing_models:
            preview = ", ".join(missing_models[:3])
            if len(missing_models) > 3:
                preview += f", and {len(missing_models) - 3} more"
            checks.append(
                {
                    "status": "FAIL",
                    "item": "Model files",
                    "detail": f"Missing expected model assets: {preview}.",
                }
            )
        else:
            checks.append(
                {
                    "status": "OK",
                    "item": "Model files",
                    "detail": "Required CellBox-Models assets are present for the selected backend.",
                }
            )

        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                checks.append(
                    {
                        "status": "OK",
                        "item": "Output directory",
                        "detail": f"Writable output folder: {self.output_dir}",
                    }
                )
            except OSError as exc:
                checks.append(
                    {
                        "status": "FAIL",
                        "item": "Output directory",
                        "detail": f"Could not create or write to output folder: {exc}",
                    }
                )
        else:
            checks.append(
                {
                    "status": "WARN",
                    "item": "Output directory",
                    "detail": "Choose where reports, plots, and videos should be saved.",
                }
            )

        python_path = self.selected_python_var.get().strip()
        if python_path and os.path.isfile(python_path):
            checks.append(
                {
                    "status": "OK",
                    "item": "Python interpreter",
                    "detail": python_path,
                }
            )
        else:
            checks.append(
                {
                    "status": "FAIL",
                    "item": "Python interpreter",
                    "detail": "Select a valid Python runtime before running preflight or launching analysis.",
                }
            )

        self.stage_var.set(self.derive_stage_from_checks(checks))
        return checks

    def derive_stage_from_checks(self, checks):
        statuses = {check["status"] for check in checks}
        if "FAIL" in statuses:
            return "Setup needs attention"
        if "WARN" in statuses:
            return "Almost ready. A few setup details remain."
        return "Ready for a full preflight check"

    def refresh_badges(self, checks):
        summary = {
            "Videos": "WARN",
            "Pipeline": "FAIL",
            "Models": "FAIL",
            "Environment": "WARN",
        }
        for check in checks:
            item = check["item"]
            if item == "Input videos":
                summary["Videos"] = check["status"]
            elif item in ("Pipeline scripts", "Low-res backend"):
                summary["Pipeline"] = check["status"]
            elif item == "Model files":
                summary["Models"] = check["status"]
            elif item in ("Python interpreter", "Runtime imports"):
                summary["Environment"] = self.merge_status(summary["Environment"], check["status"])

        for label, status in summary.items():
            self.paint_badge(label, status)

    def merge_status(self, current, incoming):
        priority = {"FAIL": 3, "WARN": 2, "OK": 1}
        if incoming not in priority:
            return current
        if current not in priority:
            return incoming
        return incoming if priority[incoming] >= priority[current] else current

    def paint_badge(self, label, status):
        widget = self.badge_widgets[label]
        if status == "OK":
            widget.config(
                text=f"{label}: READY",
                bg=self.colors["glass_alt"],
                fg=self.colors["text"],
                highlightbackground=self.colors["glass_border"],
            )
        elif status == "WARN":
            widget.config(
                text=f"{label}: REVIEW",
                bg=self.colors["glass_alt"],
                fg=self.colors["muted"],
                highlightbackground=self.colors["glass_border"],
            )
        else:
            widget.config(
                text=f"{label}: ACTION",
                bg=self.colors["danger_soft"],
                fg=self.colors["danger"],
                highlightbackground="#FECDCA",
            )

    def render_checks(self, checks):
        if self.checks_tree is None:
            return

        for item in self.checks_tree.get_children():
            self.checks_tree.delete(item)

        for check in checks:
            status = check["status"]
            tag = "info"
            if status == "OK":
                tag = "ok"
            elif status == "WARN":
                tag = "warn"
            elif status == "FAIL":
                tag = "fail"
            self.checks_tree.insert(
                "",
                tk.END,
                values=(status, check["item"], check["detail"]),
                tags=(tag,),
            )

    def build_runtime_env(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLCONFIGDIR"] = self.script_output_dir
        env["HF_HOME"] = self.hf_cache_dir
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"
        return env

    def run_preflight_check(self):
        self.set_stage("Running full preflight checks")
        self.start_progress()
        self.log_to_terminal("Running preflight checks...")
        thread = threading.Thread(target=self.perform_preflight_check, daemon=True)
        thread.start()

    def perform_preflight_check(self):
        checks = self.build_quick_checks()

        python_path = self.selected_python_var.get().strip()
        if python_path and os.path.isfile(python_path):
            runtime_check = self.run_python_runtime_probe(python_path)
            checks.extend(runtime_check)
        else:
            checks.append(
                {
                    "status": "FAIL",
                    "item": "Runtime imports",
                    "detail": "Skipped because no valid Python interpreter is selected.",
                }
            )

        parameter_issues = self.validate_parameter_values(as_checks=True)
        checks.extend(parameter_issues)

        self.latest_checks = checks
        self.root.after(0, lambda: self.complete_preflight(checks))

    def run_python_runtime_probe(self, python_path):
        required_modules = ["torch", "transformers", "cellpose", "cv2", "skimage", "matplotlib", "pandas"]
        if self.tracking_backend_var.get() == "low_res":
            required_modules.append("ultralytics")
        probe_script = f"""
import importlib.util
import json
import sys

modules = {required_modules!r}
missing = [name for name in modules if importlib.util.find_spec(name) is None]
payload = {{
    "python": sys.version.split()[0],
    "missing": missing,
    "cellpose": None,
    "device": "cpu",
    "torch": None,
    "torch_cuda": None,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_name": None,
    "cuda_capability": None,
    "cuda_sm": None,
    "cuda_arch_list": [],
    "cuda_sm_supported": None,
    "cuda_smoke_ok": None,
    "cuda_smoke_error": None,
}}
if importlib.util.find_spec("cellpose") is not None:
    import cellpose
    payload["cellpose"] = getattr(cellpose, "__version__", "unknown")
if importlib.util.find_spec("torch") is not None:
    import torch
    payload["torch"] = getattr(torch, "__version__", "unknown")
    payload["torch_cuda"] = getattr(torch.version, "cuda", None)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if payload["cuda_available"]:
        payload["device"] = "cuda"
        payload["cuda_device_count"] = torch.cuda.device_count()
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        try:
            capability = torch.cuda.get_device_capability(0)
            payload["cuda_capability"] = list(capability)
            payload["cuda_sm"] = f"sm_{{capability[0]}}{{capability[1]}}"
        except Exception as exc:
            payload["cuda_smoke_error"] = f"Could not read CUDA capability: {{type(exc).__name__}}: {{exc}}"
        try:
            payload["cuda_arch_list"] = list(torch.cuda.get_arch_list())
        except Exception:
            payload["cuda_arch_list"] = []
        if payload["cuda_sm"] is not None:
            payload["cuda_sm_supported"] = payload["cuda_sm"] in payload["cuda_arch_list"]
        try:
            x = torch.randn(1, 3, 32, 32, device="cuda")
            model = torch.nn.Conv2d(3, 8, 3).cuda()
            y = model(x)
            torch.cuda.synchronize()
            payload["cuda_smoke_ok"] = True
        except Exception as exc:
            payload["cuda_smoke_ok"] = False
            payload["cuda_smoke_error"] = f"{{type(exc).__name__}}: {{str(exc).splitlines()[0]}}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        payload["device"] = "mps"
print(json.dumps(payload))
"""

        try:
            completed = subprocess.run(
                [python_path, "-c", probe_script],
                capture_output=True,
                text=True,
                timeout=40,
                env=self.build_runtime_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [
                {
                    "status": "FAIL",
                    "item": "Runtime imports",
                    "detail": f"Could not inspect the selected Python environment: {exc}",
                }
            ]

        payload = None
        for line in reversed(completed.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = None
                break

        if completed.returncode != 0 or payload is None:
            detail = completed.stderr.strip() or completed.stdout.strip() or "Runtime probe did not return structured output."
            return [
                {
                    "status": "FAIL",
                    "item": "Runtime imports",
                    "detail": detail,
                }
            ]

        checks = []
        if payload["missing"]:
            checks.append(
                {
                    "status": "FAIL",
                    "item": "Runtime imports",
                    "detail": "Missing modules: " + ", ".join(payload["missing"]),
                }
            )
        else:
            checks.append(
                {
                    "status": "OK",
                    "item": "Runtime imports",
                    "detail": f"Python {payload['python']} with required modules available: {', '.join(required_modules)}.",
                }
            )

        torch_version = payload.get("torch")
        if torch_version:
            if payload["device"] == "cuda":
                capability = payload.get("cuda_capability")
                if isinstance(capability, list) and len(capability) >= 2:
                    capability_text = f"{capability[0]}.{capability[1]}"
                else:
                    capability_text = "unknown"
                sm_text = payload.get("cuda_sm") or "unknown"
                if payload.get("cuda_smoke_ok") is False:
                    checks.append(
                        {
                            "status": "FAIL",
                            "item": "Accelerator",
                            "detail": (
                                f"CUDA is detected, but a test Conv2d failed on {payload.get('cuda_device_name')} "
                                f"(compute capability {capability_text}, {sm_text}; PyTorch CUDA build "
                                f"{payload.get('torch_cuda')}): {payload.get('cuda_smoke_error')}. "
                                "For RTX 50-series GPUs, install a CUDA 12.8 or newer PyTorch wheel."
                            ),
                        }
                    )
                else:
                    arch_note = ""
                    if payload.get("cuda_sm_supported") is False:
                        arch_note = f" Build arch list does not include {sm_text}; install a newer PyTorch CUDA wheel if CUDA kernels fail."
                    checks.append(
                        {
                            "status": "OK",
                            "item": "Accelerator",
                            "detail": (
                                f"CUDA is available and a test Conv2d passed. PyTorch {torch_version}, "
                                f"CUDA build {payload.get('torch_cuda')}, GPU: {payload.get('cuda_device_name')}, "
                                f"compute capability {capability_text} ({sm_text}).{arch_note}"
                            ),
                        }
                    )
            elif payload["device"] == "mps":
                checks.append(
                    {
                        "status": "OK",
                        "item": "Accelerator",
                        "detail": f"Apple MPS is available. PyTorch {torch_version}.",
                    }
                )
            else:
                cuda_build = payload.get("torch_cuda") or "none"
                checks.append(
                    {
                        "status": "WARN",
                        "item": "Accelerator",
                        "detail": (
                            f"PyTorch {torch_version} reports torch.cuda.is_available()=False "
                            f"(CUDA build: {cuda_build}); SickleSight will run on CPU."
                        ),
                    }
                )

        cellpose_version = payload.get("cellpose")
        if cellpose_version:
            custom_cellpose_model = os.path.join(self.pipeline_dir, "CellBox-Models", "cyto3_train0327")
            if cellpose_version.startswith("4") and os.path.exists(custom_cellpose_model):
                checks.append(
                    {
                        "status": "FAIL",
                        "item": "Cellpose compatibility",
                        "detail": (
                            f"Cellpose {cellpose_version} looks incompatible with the custom CP3 model "
                            "cyto3_train0327. Use a Python environment with Cellpose 3.x."
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "status": "OK",
                        "item": "Cellpose compatibility",
                        "detail": f"Cellpose {cellpose_version} on device {payload['device']}.",
                    }
                )
        else:
            checks.append(
                {
                    "status": "WARN",
                    "item": "Cellpose compatibility",
                    "detail": "Cellpose version could not be determined.",
                }
            )

        return checks

    def validate_parameter_values(self, as_checks=False):
        issues = []
        validators = []
        script_name = self.selected_pipeline_var.get().strip()
        supported = set(PIPELINE_ARGUMENTS.get(script_name, ()))

        if "frame_skip" in supported:
            validators.append(("Frame skip", self.frame_skip_var.get().strip(), int, lambda value: value >= 1))
        if "max_frame" in supported:
            validators.append(("Max frame", self.max_frame_var.get().strip(), int, lambda value: value >= 1))
        if "max_time" in supported and not self.full_video_var.get():
            validators.append(("Max seconds", self.max_seconds_var.get().strip(), float, lambda value: value > 0))
        if "analysis_fps" in supported:
            fps_value = self.analysis_fps_var.get().strip()
            if fps_value and fps_value.lower() != "auto":
                validators.append(("Frames/sec", fps_value, float, lambda value: value > 0))
        if "target_frames" in supported:
            validators.append(("Target frames", self.target_frames_var.get().strip(), "frames", None))
        if "low_res_det_conf" in supported and self.tracking_backend_var.get() == "low_res":
            validators.append(("Low-res YOLO conf", self.low_res_det_conf_var.get().strip(), "confidence", None))
        if "tracking_backend" in supported:
            valid_backends = {value for _label, value in TRACKING_BACKEND_OPTIONS}
            if self.tracking_backend_var.get() not in valid_backends:
                issues.append(("Segmentation / Tracking", "Choose either Cellpose or Low-resolution YOLO/BoT-SORT."))

        for label, raw_value, parser, predicate in validators:
            if parser == "frames":
                if not raw_value:
                    continue
                try:
                    frames = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
                    if not frames or any(value < 0 for value in frames):
                        raise ValueError
                except ValueError:
                    issues.append((label, "Use a comma-separated list of non-negative frame numbers, for example 0,480"))
                continue
            if parser == "confidence":
                if not raw_value or raw_value.lower() == "auto":
                    continue
                try:
                    confidence = float(raw_value)
                    if confidence <= 0 or confidence > 1:
                        raise ValueError
                except ValueError:
                    issues.append((label, "Use 'auto' or a number between 0 and 1, for example 0.25"))
                continue

            try:
                value = parser(raw_value)
                if not predicate(value):
                    raise ValueError
            except ValueError:
                issues.append((label, "Enter a positive integer value."))

        if not as_checks:
            return issues

        if not issues:
            return [
                {
                    "status": "OK",
                    "item": "Parameter validation",
                    "detail": "Pipeline parameters look valid for the selected script.",
                }
            ]

        return [
            {
                "status": "FAIL",
                "item": f"Parameter: {label}",
                "detail": detail,
            }
            for label, detail in issues
        ]

    def complete_preflight(self, checks):
        self.stop_progress()
        self.render_checks(checks)
        self.refresh_badges(checks)

        if any(check["status"] == "FAIL" for check in checks):
            self.set_stage("Preflight found blocking issues")
            self.log_to_terminal("Preflight failed. Review the checklist before running analysis.")
        elif any(check["status"] == "WARN" for check in checks):
            self.set_stage("Preflight complete with warnings")
            self.log_to_terminal("Preflight completed with warnings.")
        else:
            self.set_stage("Preflight passed")
            self.log_to_terminal("Preflight passed. The pipeline is ready to run.")

    def run_analysis(self):
        issues = self.validate_parameter_values()
        if issues:
            message = "\n".join(f"{label}: {detail}" for label, detail in issues)
            messagebox.showwarning("Check parameters", message)
            return

        target_files = self.collect_target_files()
        if not target_files:
            messagebox.showwarning("No videos selected", "Add at least one .mp4 file before starting an analysis.")
            return

        quick_checks = self.build_quick_checks()
        blocking = [check for check in quick_checks if check["status"] == "FAIL"]
        if blocking:
            self.render_checks(quick_checks)
            self.refresh_badges(quick_checks)
            self.log_blocking_setup_issues(blocking)
            messagebox.showwarning(
                "Setup incomplete",
                self.format_blocking_setup_message(blocking),
            )
            return

        script_path = self.write_launcher_script(target_files)
        if not script_path:
            return

        self.last_script_path = script_path
        self.script_summary_var.set(script_path)
        self.run_started_at = time.time()
        self.last_results_dir = self.output_dir
        self.log_to_terminal("")
        self.log_to_terminal("=" * 72)
        self.log_to_terminal(f"Launching {self.selected_pipeline_var.get()} for {len(target_files)} video(s)")
        self.log_to_terminal("=" * 72)
        self.start_progress()
        self.set_stage("Launching analysis")

        self.process_thread = threading.Thread(target=self.execute_script, args=(script_path,), daemon=True)
        self.process_thread.start()
        self.save_session()

    def format_blocking_setup_message(self, blocking_checks):
        lines = ["Fix these setup issues before running analysis:"]
        for check in blocking_checks[:6]:
            lines.append(f"- {check['item']}: {check['detail']}")
        if len(blocking_checks) > 6:
            lines.append(f"- ...and {len(blocking_checks) - 6} more issue(s)")
        lines.append("")
        lines.append("Use Check to re-run the full preflight after fixing them.")
        return "\n".join(lines)

    def log_blocking_setup_issues(self, blocking_checks):
        self.set_stage("Setup needs attention")
        self.log_to_terminal("Run blocked by setup issues:")
        for check in blocking_checks:
            self.log_to_terminal(f"- {check['item']}: {check['detail']}")

    def write_launcher_script(self, target_files):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".bat" if self.is_windows else ".sh"
        script_path = os.path.join(self.script_output_dir, f"analyze_{timestamp}{ext}")
        try:
            with open(script_path, "w", encoding="utf-8") as handle:
                if self.is_windows:
                    handle.write(self.generate_bat_content(target_files))
                else:
                    handle.write(self.generate_sh_content(target_files))
        except OSError as exc:
            messagebox.showerror("Launcher script error", f"Could not write the launcher script:\n{exc}")
            return ""
        return script_path

    def generate_common_args(self):
        script_name = self.selected_pipeline_var.get().strip()
        args = []
        supported = set(PIPELINE_ARGUMENTS.get(script_name, ()))
        if "frame_skip" in supported:
            args.extend(["--frame_skip", self.frame_skip_var.get().strip()])
        if "full_video" in supported and self.full_video_var.get():
            args.append("--full_video")
        elif "max_time" in supported:
            args.extend(["--max_time", self.max_seconds_var.get().strip()])
        if "analysis_fps" in supported:
            fps_text = self.analysis_fps_var.get().strip()
            if fps_text and fps_text.lower() != "auto":
                args.extend(["--analysis_fps", fps_text])
        if "max_frame" in supported:
            args.extend(["--max_frame", self.max_frame_var.get().strip()])
        if "target_frames" in supported and self.target_frames_var.get().strip():
            args.extend(["--target_frames", self.target_frames_var.get().strip()])
        if "tracking_backend" in supported:
            args.extend(["--tracking_backend", self.tracking_backend_var.get().strip()])
        if "low_res_det_conf" in supported and self.tracking_backend_var.get() == "low_res":
            confidence_text = self.low_res_det_conf_var.get().strip() or "auto"
            args.extend(["--low_res_det_conf", confidence_text])
        return args

    def generate_bat_content(self, target_files):
        script_name = self.selected_pipeline_var.get().strip()
        python_exe = self.selected_python_var.get().strip()
        inputs = ",".join(target_files)
        extra_args = self.generate_common_args()
        quoted_extra = " ".join(f'"{arg}"' if " " in arg else arg for arg in extra_args)

        lines = [
            "@echo off",
            "setlocal EnableDelayedExpansion",
            f'cd /d "{self.pipeline_dir}"',
            'set "PYTHONIOENCODING=utf-8"',
            f'set "PYTHON_EXE={python_exe}"',
            f'set "MPLCONFIGDIR={self.script_output_dir}"',
            f'set "HF_HOME={self.hf_cache_dir}"',
            'set "KMP_DUPLICATE_LIB_OK=TRUE"',
            'set "OMP_NUM_THREADS=1"',
            f'set "INPUTS={inputs}"',
            f'set "OUTPUT_ROOT={self.output_dir}"',
            f'echo Running {script_name}...',
            f'"%PYTHON_EXE%" "{script_name}" -i "%INPUTS%" -o "%OUTPUT_ROOT%" {quoted_extra}'.rstrip(),
            "echo Done.",
            "endlocal",
        ]
        return "\n".join(lines)

    def generate_sh_content(self, target_files):
        script_name = self.selected_pipeline_var.get().strip()
        python_exe = self.selected_python_var.get().strip()
        inputs = ",".join(target_files)
        extra_args = self.generate_common_args()
        quoted_extra = " ".join(shlex.quote(arg) for arg in extra_args)

        lines = [
            "#!/bin/bash",
            "set -e",
            f"cd {shlex.quote(self.pipeline_dir)}",
            'export PYTHONIOENCODING="utf-8"',
            f'export MPLCONFIGDIR={shlex.quote(self.script_output_dir)}',
            f'export HF_HOME={shlex.quote(self.hf_cache_dir)}',
            'export KMP_DUPLICATE_LIB_OK="TRUE"',
            'export OMP_NUM_THREADS="1"',
            f'PYTHON_EXE={shlex.quote(python_exe)}',
            f'INPUTS={shlex.quote(inputs)}',
            f'OUTPUT_ROOT={shlex.quote(self.output_dir)}',
            'mkdir -p "$MPLCONFIGDIR"',
            'mkdir -p "$HF_HOME"',
            f'echo "Running {script_name}..."',
            f'"$PYTHON_EXE" {shlex.quote(script_name)} -i "$INPUTS" -o "$OUTPUT_ROOT" {quoted_extra}'.rstrip(),
            'echo "Done."',
        ]
        return "\n".join(lines)

    def execute_script(self, script_path):
        abs_path = os.path.abspath(script_path)
        self.root.after(0, lambda: self.set_stage(f"Running {os.path.basename(abs_path)}"))

        if not self.is_windows:
            try:
                current_mode = os.stat(abs_path).st_mode
                os.chmod(abs_path, current_mode | stat.S_IXUSR)
            except OSError as exc:
                self.root.after(0, lambda: self.log_to_terminal(f"Warning: could not mark script executable: {exc}"))

        if self.is_windows:
            cmd = [abs_path]
            use_shell = True
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            cmd = ["/bin/bash", abs_path]
            use_shell = False
            creation_flags = 0

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=use_shell,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
                env=self.build_runtime_env(),
            )
            self.active_process = process

            if process.stdout:
                for line in process.stdout:
                    clean = line.rstrip()
                    if clean:
                        self.root.after(0, lambda message=clean: self.handle_runtime_line(message))

            process.wait()
            exit_code = process.returncode
        except Exception as exc:
            exit_code = -1
            self.root.after(0, lambda: self.log_to_terminal(f"Failed to start process: {exc}"))
        finally:
            self.active_process = None
            self.root.after(0, lambda: self.finish_run(exit_code))

    def handle_runtime_line(self, message):
        self.log_to_terminal(message)
        lowered = message.lower()
        if "loading models" in lowered:
            self.set_stage("Loading models")
        elif "initialization" in lowered:
            self.set_stage("Initializing video context")
        elif "process cells in frame 0" in lowered:
            self.set_stage("Segmenting and classifying cells in frame 0")
        elif "processing video" in lowered:
            self.set_stage("Processing selected videos")
        elif "done" == lowered.strip():
            self.set_stage("Finalizing outputs")

    def finish_run(self, exit_code):
        self.stop_progress()
        self.log_to_terminal(f"Process finished with exit code: {exit_code}")

        if exit_code == 0:
            self.set_stage("Analysis complete")
            summary = self.build_results_summary()
            self.output_summary_var.set(summary)
            self.refresh_results_preview_tree()
            self.refresh_inline_results_preview()
            messagebox.showinfo("Analysis complete", "The analysis finished successfully.")
        else:
            self.set_stage("Run failed")
            summary = self.build_results_summary(failed=True)
            self.output_summary_var.set(summary)
            self.refresh_results_preview_tree()
            self.refresh_inline_results_preview()
            messagebox.showerror(
                "Analysis stopped",
                "The run did not finish successfully. Review the run log for the first traceback or error line.",
            )

    def build_results_summary(self, failed=False):
        if not self.output_dir or not os.path.isdir(self.output_dir):
            return "No output directory is available yet."

        threshold = (self.run_started_at or 0) - 2
        new_files = []
        for root_dir, _, files in os.walk(self.output_dir):
            for filename in files:
                full_path = os.path.join(root_dir, filename)
                try:
                    if os.path.getmtime(full_path) >= threshold:
                        new_files.append(full_path)
                except OSError:
                    continue

        if new_files:
            self.last_results_dir = self.output_dir
            counts = {}
            for path in new_files:
                extension = os.path.splitext(path)[1].lower() or "[no extension]"
                counts[extension] = counts.get(extension, 0) + 1
            parts = [f"{count} {ext}" for ext, count in sorted(counts.items())]
            status = "Partial output created before the run stopped." if failed else "New output files created."
            return (
                f"{status}\n"
                f"Output root: {self.output_dir}\n"
                f"New files detected: {len(new_files)} ({', '.join(parts)})"
            )

        if failed:
            return (
                "The run stopped before new files were fully written.\n"
                f"Output root: {self.output_dir}\n"
                "Check the run log for the blocking traceback."
            )

        return (
            f"The run completed, but no newly modified files were detected in {self.output_dir}.\n"
            "This usually means the selected output root already contained old results or the pipeline writes later than expected."
        )

    def stop_analysis(self):
        if self.active_process and self.active_process.poll() is None:
            self.log_to_terminal("Stopping the active analysis process...")
            try:
                self.active_process.terminate()
            except OSError as exc:
                self.log_to_terminal(f"Could not terminate the process cleanly: {exc}")
            self.set_stage("Stopping run")
        else:
            self.log_to_terminal("No active process to stop.")

    def start_progress(self):
        if self.progress is not None:
            self.progress.start(10)

    def stop_progress(self):
        if self.progress is not None:
            self.progress.stop()

    def set_stage(self, text):
        self.stage_var.set(text)

    def clear_terminal(self):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.delete("1.0", tk.END)
        self.terminal.config(state=tk.DISABLED)

    def log_to_terminal(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.terminal.config(state=tk.NORMAL)
        if message:
            self.terminal.insert(tk.END, f"[{timestamp}] {message}\n")
        else:
            self.terminal.insert(tk.END, "\n")
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def open_output_folder(self):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.open_path(self.output_dir)
        else:
            messagebox.showinfo("No output folder", "Choose an output folder first.")

    def refresh_inline_results_preview(self):
        if not getattr(self, "inline_results_slots", None):
            return

        root_dir = self.get_results_preview_root()
        files = self.select_inline_figure_files(self.collect_results_preview_files(root_dir))
        if not root_dir or not files:
            self.inline_results_preview_path = ""
            self.inline_results_photo = None
            self.inline_results_files = []
            self.inline_results_index = 0
            self.inline_results_window_start = 0
            self.inline_results_position_var.set("0 / 0")
            self.inline_results_status_var.set("No output files found yet.")
            if getattr(self, "inline_results_image_label", None) is not None:
                self.inline_results_image_label.config(
                    image="",
                    text="Run analysis or select an output folder",
                    bg=self.colors["glass"],
                    fg=self.colors["muted"],
                )
            self.write_inline_results_list([])
            return

        previous_path = self.inline_results_preview_path
        self.inline_results_files = files
        self.inline_results_index = self.find_inline_result_index(files, previous_path)
        if self.inline_results_index is None:
            self.inline_results_index = self.find_inline_result_index(files, self.choose_inline_result_preview(files))
        if self.inline_results_index is None:
            self.inline_results_index = 0
        self.ensure_inline_result_visible(self.inline_results_index)
        self.show_inline_result_at_index(self.inline_results_index)
        self.write_inline_results_list(files)

    def find_inline_result_index(self, files, path):
        if not path:
            return None
        normalized = os.path.normpath(path)
        for index, (candidate, _extension, _size, _modified) in enumerate(files):
            if os.path.normpath(candidate) == normalized:
                return index
        return None

    def navigate_inline_result(self, direction):
        if not self.inline_results_files:
            self.refresh_inline_results_preview()
            return
        self.inline_results_index = (self.inline_results_index + direction) % len(self.inline_results_files)
        self.ensure_inline_result_visible(self.inline_results_index)
        self.show_inline_result_at_index(self.inline_results_index)
        self.write_inline_results_list(self.inline_results_files)

    def select_inline_result_slot(self, slot_index):
        target_index = self.inline_results_window_start + slot_index
        if target_index >= len(self.inline_results_files):
            return
        self.show_inline_result_at_index(target_index)
        self.write_inline_results_list(self.inline_results_files)

    def ensure_inline_result_visible(self, index):
        if index < self.inline_results_window_start:
            self.inline_results_window_start = index
        elif index >= self.inline_results_window_start + 3:
            self.inline_results_window_start = max(0, index - 2)

    def show_inline_result_at_index(self, index):
        if not self.inline_results_files:
            return
        bounded_index = max(0, min(index, len(self.inline_results_files) - 1))
        self.inline_results_index = bounded_index
        path = self.inline_results_files[bounded_index][0]
        self.inline_results_preview_path = path
        self.inline_results_position_var.set(f"{bounded_index + 1} / {len(self.inline_results_files)}")
        if getattr(self, "inline_results_image_label", None) is not None:
            self.render_inline_result_preview(path)
            return
        try:
            size = self.format_file_size(os.path.getsize(path))
        except OSError:
            size = "Unknown size"
        self.inline_results_status_var.set(f"{os.path.basename(path)}\n{size}")

    def choose_inline_result_preview(self, files):
        image_files = [item for item in files if item[1] in {".png", ".jpg", ".jpeg"}]
        if not image_files:
            return files[0][0]

        priority_fragments = (
            "state_ratio_plot_binary",
            "state_ratio_plot_pocked",
            "state_ratio_plot",
            "multiframe_comparison",
            "violin_overall",
            "class_pie",
            "annotated",
            "first_frame",
        )
        for fragment in priority_fragments:
            for path, _extension, _size, _modified in image_files:
                if fragment in os.path.basename(path).lower():
                    return path
        return image_files[0][0]

    def render_inline_result_preview(self, path):
        if getattr(self, "inline_results_image_label", None) is None:
            return
        filename = os.path.basename(path)
        extension = os.path.splitext(path)[1].lower()
        try:
            size = self.format_file_size(os.path.getsize(path))
        except OSError:
            size = "Unknown size"
        self.inline_results_status_var.set(f"{filename}\n{size}")

        if extension not in {".png", ".jpg", ".jpeg"} or Image is None or ImageTk is None:
            self.inline_results_photo = None
            self.inline_results_image_label.config(
                image="",
                text=f"Preview selected:\n{filename}",
                bg=self.colors["glass"],
                fg=self.colors["text"],
            )
            return

        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            self.inline_results_photo = None
            self.inline_results_image_label.config(
                image="",
                text=f"Could not preview:\n{filename}",
                bg=self.colors["glass"],
                fg=self.colors["danger"],
            )
            return

        target_width = max(self.inline_results_image_holder.winfo_width() - 16, 280)
        target_height = 170
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((target_width, target_height), resampling)
        self.inline_results_photo = ImageTk.PhotoImage(image)
        self.inline_results_image_label.config(image=self.inline_results_photo, text="")
        self.set_hover_cursor(self.inline_results_image_label, INSPECT_CURSOR)

    def write_inline_results_list(self, files):
        if not getattr(self, "inline_results_slots", None):
            return
        for _slot, image_label, _text_label in self.inline_results_slots:
            image_label.config(image="", text="")
        self.inline_results_slot_photos = []
        if not files:
            for slot, image_label, text_label in self.inline_results_slots:
                slot.config(highlightbackground=self.colors["glass_border"])
                image_label.config(image="", text="-", bg=self.colors["glass"], fg=self.colors["muted"])
                text_label.config(text="No previewable result", bg=self.colors["glass"], fg=self.colors["muted"])
            return

        self.inline_results_window_start = 0
        root_dir = self.get_results_preview_root()
        for slot_index, (slot, image_label, text_label) in enumerate(self.inline_results_slots):
            file_index = self.inline_results_window_start + slot_index
            if file_index >= len(files):
                slot.config(highlightbackground=self.colors["glass_border"])
                image_label.config(image="", text="-", bg=self.colors["glass"], fg=self.colors["muted"])
                text_label.config(text="No more results", bg=self.colors["glass"], fg=self.colors["muted"])
                continue

            path, extension, size, _modified = files[file_index]
            relative = os.path.relpath(path, root_dir) if root_dir else os.path.basename(path)
            selected = file_index == self.inline_results_index
            slot_bg = self.colors["card_alt"] if selected else self.colors["glass"]
            border = self.colors["accent"] if selected else self.colors["glass_border"]
            slot.config(bg=slot_bg, highlightbackground=border)
            image_label.config(bg=slot_bg)
            text_label.config(
                bg=slot_bg,
                fg=self.colors["text"] if selected else self.colors["muted"],
                text=f"{os.path.basename(relative)}  |  {self.format_file_size(size)}",
            )
            self.render_inline_result_slot_image(image_label, path, extension)

    def render_inline_result_slot_image(self, image_label, path, extension):
        if extension not in {".png", ".jpg", ".jpeg"} or Image is None or ImageTk is None:
            image_label.config(image="", text=extension.upper().lstrip(".") or "FILE")
            return
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            image_label.config(image="", text="ERR")
            return
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        target_width = max(image_label.winfo_width() - 8, 238)
        target_height = max(image_label.winfo_height() - 8, 86)
        image.thumbnail((target_width, target_height), resampling)
        photo = ImageTk.PhotoImage(image)
        self.inline_results_slot_photos.append(photo)
        image_label.config(image=photo, text="")

    def open_current_inline_result_fullscreen(self, _event=None):
        self.open_result_fullscreen(self.inline_results_preview_path)
        return "break"

    def open_inline_result_slot_fullscreen(self, slot_index):
        target_index = self.inline_results_window_start + slot_index
        if target_index >= len(self.inline_results_files):
            return "break"
        self.show_inline_result_at_index(target_index)
        self.write_inline_results_list(self.inline_results_files)
        self.open_result_fullscreen(self.inline_results_files[target_index][0])
        return "break"

    def open_result_fullscreen(self, path):
        if not path or not os.path.exists(path):
            return
        extension = os.path.splitext(path)[1].lower()
        if extension not in {".png", ".jpg", ".jpeg"} or Image is None or ImageTk is None:
            self.open_path(path)
            return

        self.close_fullscreen_result_preview()
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            messagebox.showerror("Preview failed", f"Could not open image:\n{exc}")
            return

        window = tk.Toplevel(self.root)
        self.fullscreen_preview_window = window
        window.configure(bg="#030712")
        window.attributes("-fullscreen", True)
        window.bind("<Escape>", self.close_fullscreen_result_preview)
        window.bind("<Button-1>", self.close_fullscreen_result_preview)

        screen_width = max(window.winfo_screenwidth(), 1024)
        screen_height = max(window.winfo_screenheight(), 768)
        max_width = screen_width - 96
        max_height = screen_height - 142
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((max_width, max_height), resampling)
        self.fullscreen_preview_photo = ImageTk.PhotoImage(image)

        title = tk.Label(
            window,
            text=os.path.basename(path),
            bg="#030712",
            fg="#F9FAFB",
            font=self.font_body_bold,
            padx=24,
            pady=18,
        )
        title.pack(fill=tk.X)

        image_label = tk.Label(window, image=self.fullscreen_preview_photo, bg="#030712")
        image_label.pack(expand=True)

        hint = tk.Label(
            window,
            text="Click anywhere or press Esc to close",
            bg="#030712",
            fg="#98A2B3",
            font=self.font_small,
            padx=24,
            pady=18,
        )
        hint.pack(fill=tk.X)

        window.lift()
        window.focus_force()

    def close_fullscreen_result_preview(self, _event=None):
        if self.fullscreen_preview_window is not None and self.fullscreen_preview_window.winfo_exists():
            self.fullscreen_preview_window.destroy()
        self.fullscreen_preview_window = None
        self.fullscreen_preview_photo = None

    def open_inline_result_file(self):
        if self.inline_results_preview_path and os.path.exists(self.inline_results_preview_path):
            self.open_path(self.inline_results_preview_path)
            return
        self.open_results_preview_window()

    def open_last_results(self):
        if self.last_results_dir and os.path.isdir(self.last_results_dir):
            self.open_path(self.last_results_dir)
            return
        if self.output_dir:
            self.open_path(self.output_dir)
            return
        messagebox.showinfo("No results yet", "Run an analysis first or choose an output folder.")

    def get_results_preview_root(self):
        if self.last_results_dir and os.path.isdir(self.last_results_dir):
            return self.last_results_dir
        if self.output_dir and os.path.isdir(self.output_dir):
            return self.output_dir
        return ""

    def select_inline_figure_files(self, files):
        if not files:
            return []

        root_dir = self.get_results_preview_root()
        target_groups = (
            ("state_ratio_plot_binary.png",),
            ("combined_frame0_class_pie.png", "frame0_class_pie.png"),
            ("combined_state_ratio_plot_14groups.png", "state_ratio_plot_14groups.png"),
        )
        selected = []
        used_paths = set()

        for names in target_groups:
            ranked_matches = []
            for item in files:
                path, extension, _size, _modified = item
                basename = os.path.basename(path).lower()
                if extension not in {".png", ".jpg", ".jpeg"} or basename not in names:
                    continue
                try:
                    name_rank = names.index(basename)
                except ValueError:
                    name_rank = len(names)
                root_rank = 0 if root_dir and os.path.dirname(path) == root_dir else 1
                ranked_matches.append((name_rank, root_rank, path.lower(), item))
            if ranked_matches:
                _name_rank, _root_rank, _path_key, item = sorted(ranked_matches)[0]
                selected.append(item)
                used_paths.add(os.path.normpath(item[0]))

        if len(selected) < 3:
            for item in files:
                path, extension, _size, _modified = item
                if extension not in {".png", ".jpg", ".jpeg"}:
                    continue
                normalized = os.path.normpath(path)
                if normalized in used_paths:
                    continue
                selected.append(item)
                used_paths.add(normalized)
                if len(selected) == 3:
                    break

        return selected[:3]

    def collect_results_preview_files(self, root_dir):
        if not root_dir or not os.path.isdir(root_dir):
            return []

        preferred_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".csv",
            ".txt",
            ".log",
            ".avi",
            ".mp4",
            ".pkl",
            ".npy",
        }
        files = []
        for current_root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                full_path = os.path.join(current_root, filename)
                extension = os.path.splitext(filename)[1].lower()
                if extension not in preferred_extensions:
                    continue
                try:
                    modified = os.path.getmtime(full_path)
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                files.append((full_path, extension, size, modified))

        def sort_key(item):
            path, extension, _, modified = item
            rank = {
                ".png": 0,
                ".jpg": 0,
                ".jpeg": 0,
                ".csv": 1,
                ".txt": 2,
                ".log": 2,
                ".avi": 3,
                ".mp4": 3,
                ".pkl": 4,
                ".npy": 4,
            }.get(extension, 9)
            return (rank, -modified, os.path.basename(path).lower())

        return sorted(files, key=sort_key)

    def format_file_size(self, size):
        units = ["B", "KB", "MB", "GB"]
        value = float(size)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{size} B"

    def open_results_preview_window(self):
        root_dir = self.get_results_preview_root()
        if not root_dir:
            messagebox.showinfo("No results folder", "Run an analysis first or choose an output folder.")
            return

        if self.results_preview_window is not None and self.results_preview_window.winfo_exists():
            self.results_preview_window.lift()
            self.refresh_results_preview_tree()
            return

        window = tk.Toplevel(self.root)
        self.results_preview_window = window
        window.title("SickleSight Results Preview")
        window.geometry("1180x760")
        window.minsize(980, 620)
        window.configure(bg=self.colors["bg"])
        window.protocol("WM_DELETE_WINDOW", self.close_results_preview_window)

        shell = tk.Frame(window, bg=self.colors["bg"], padx=16, pady=16)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=0, minsize=360)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg=self.colors["bg"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Results Preview",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.font_title_md,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text=root_dir,
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=self.font_code_small,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        header_actions = tk.Frame(header, bg=self.colors["bg"])
        header_actions.grid(row=0, column=1, rowspan=2, sticky="e")
        self.create_button(header_actions, "Refresh", self.refresh_results_preview_tree, width=104, height=38).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.create_button(header_actions, "Open Folder", lambda: self.open_path(root_dir), width=132, height=38).pack(
            side=tk.LEFT
        )

        list_frame = tk.Frame(
            shell,
            bg=self.colors["card"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.results_preview_tree = ttk.Treeview(
            list_frame,
            columns=("type", "size"),
            show="tree headings",
            selectmode="browse",
            style="Video.Treeview",
            cursor=DEFAULT_CURSOR,
        )
        self.set_hover_cursor(self.results_preview_tree, ACTION_CURSOR)
        self.results_preview_tree.heading("#0", text="File", anchor=tk.W)
        self.results_preview_tree.heading("type", text="Type", anchor=tk.CENTER)
        self.results_preview_tree.heading("size", text="Size", anchor=tk.E)
        self.results_preview_tree.column("#0", width=230, stretch=True)
        self.results_preview_tree.column("type", width=70, anchor=tk.CENTER, stretch=False)
        self.results_preview_tree.column("size", width=82, anchor=tk.E, stretch=False)
        self.results_preview_tree.grid(row=0, column=0, sticky="nsew")
        self.results_preview_tree.bind("<<TreeviewSelect>>", self.on_results_preview_select)
        self.results_preview_tree.bind("<Double-1>", lambda _event: self.open_selected_result_file())

        list_scroll = ttk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=self.results_preview_tree.yview,
            style="App.Vertical.TScrollbar",
            cursor=V_SCROLL_CURSOR,
        )
        self.results_preview_tree.configure(yscrollcommand=list_scroll.set)
        list_scroll.grid(row=0, column=1, sticky="ns")

        preview_frame = tk.Frame(
            shell,
            bg=self.colors["card"],
            highlightthickness=1,
            highlightbackground=self.colors["glass_border"],
        )
        preview_frame.grid(row=1, column=1, sticky="nsew")
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        tk.Label(
            preview_frame,
            textvariable=self.results_preview_status_var,
            bg=self.colors["card"],
            fg=self.colors["text"],
            font=self.font_body_bold,
            anchor="w",
            padx=14,
            pady=12,
        ).grid(row=0, column=0, sticky="ew")

        self.results_preview_display = tk.Frame(preview_frame, bg=self.colors["glass"])
        self.results_preview_display.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.results_preview_display.grid_columnconfigure(0, weight=1)
        self.results_preview_display.grid_rowconfigure(0, weight=1)

        self.refresh_results_preview_tree()

    def close_results_preview_window(self):
        if self.results_preview_window is not None:
            self.results_preview_window.destroy()
        self.results_preview_window = None
        self.results_preview_tree = None
        self.results_preview_display = None
        self.results_preview_photo = None
        self.results_preview_selected_path = ""

    def refresh_results_preview_tree(self):
        if self.results_preview_tree is None:
            return
        root_dir = self.get_results_preview_root()
        self.results_preview_files = {}
        for item in self.results_preview_tree.get_children():
            self.results_preview_tree.delete(item)

        files = self.collect_results_preview_files(root_dir)
        if not files:
            self.results_preview_status_var.set("No previewable result files found.")
            self.render_results_preview_empty("No PNG, CSV, text, video, PKL, or NPY files were found in the results folder.")
            return

        for full_path, extension, size, _modified in files:
            relative = os.path.relpath(full_path, root_dir)
            item_id = self.results_preview_tree.insert(
                "",
                tk.END,
                text=relative,
                values=(extension.upper().lstrip(".") or "FILE", self.format_file_size(size)),
            )
            self.results_preview_files[item_id] = full_path

        first_item = self.results_preview_tree.get_children()[0]
        self.results_preview_tree.selection_set(first_item)
        self.results_preview_tree.focus(first_item)
        self.show_result_preview(self.results_preview_files[first_item])

    def clear_results_preview_display(self):
        if self.results_preview_display is None:
            return
        for child in self.results_preview_display.winfo_children():
            child.destroy()
        self.results_preview_photo = None

    def on_results_preview_select(self, _event):
        if self.results_preview_tree is None:
            return
        selection = self.results_preview_tree.selection()
        if not selection:
            return
        path = self.results_preview_files.get(selection[0])
        if path:
            self.show_result_preview(path)

    def show_result_preview(self, path):
        self.results_preview_selected_path = path
        extension = os.path.splitext(path)[1].lower()
        self.results_preview_status_var.set(os.path.basename(path))
        self.clear_results_preview_display()

        if extension in {".png", ".jpg", ".jpeg"}:
            self.render_result_image(path)
        elif extension in {".csv", ".txt", ".log"}:
            self.render_result_text(path, is_csv=extension == ".csv")
        else:
            self.render_result_file_card(path)

    def render_result_image(self, path):
        if self.results_preview_display is None:
            return
        if Image is None or ImageTk is None:
            self.render_results_preview_empty("Image preview needs Pillow.")
            return
        try:
            image = Image.open(path)
            image = image.convert("RGB")
        except OSError as exc:
            self.render_results_preview_empty(f"Could not open image:\n{exc}")
            return

        target_width = max(self.results_preview_display.winfo_width() - 28, 640)
        target_height = max(self.results_preview_display.winfo_height() - 28, 420)
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((target_width, target_height), resampling)
        photo = ImageTk.PhotoImage(image)
        self.results_preview_photo = photo

        holder = tk.Frame(self.results_preview_display, bg=self.colors["glass"])
        holder.grid(row=0, column=0, sticky="nsew")
        label = tk.Label(holder, image=photo, bg=self.colors["glass"])
        label.pack(expand=True)
        self.set_hover_cursor(label, INSPECT_CURSOR)

    def render_result_text(self, path, is_csv=False):
        if self.results_preview_display is None:
            return
        text_frame = tk.Frame(self.results_preview_display, bg=self.colors["glass"])
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)

        text = tk.Text(
            text_frame,
            bg=self.colors["glass"],
            fg=self.colors["text"],
            relief=tk.FLAT,
            wrap=tk.NONE if is_csv else tk.WORD,
            font=self.font_code,
            padx=12,
            pady=12,
            cursor="xterm",
        )
        x_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text.xview, style="App.Horizontal.TScrollbar")
        y_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text.yview, style="App.Vertical.TScrollbar")
        text.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()[:220]
        except OSError as exc:
            lines = [f"Could not read file: {exc}"]
        text.insert(tk.END, "".join(lines))
        text.config(state=tk.DISABLED)

    def render_result_file_card(self, path):
        extension = os.path.splitext(path)[1].lower() or "file"
        try:
            size = self.format_file_size(os.path.getsize(path))
            modified = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            size = "Unknown size"
            modified = "Unknown modified time"
        message = (
            f"Preview is not available for {extension.upper()} files.\n\n"
            f"Path: {path}\n"
            f"Size: {size}\n"
            f"Modified: {modified}"
        )
        self.render_results_preview_empty(message, include_open_button=True)

    def render_results_preview_empty(self, message, include_open_button=False):
        self.clear_results_preview_display()
        if self.results_preview_display is None:
            return
        panel = tk.Frame(self.results_preview_display, bg=self.colors["glass"], padx=20, pady=20)
        panel.grid(row=0, column=0, sticky="nsew")
        tk.Label(
            panel,
            text=message,
            bg=self.colors["glass"],
            fg=self.colors["muted"],
            font=self.font_body,
            justify=tk.LEFT,
            wraplength=620,
        ).pack(anchor="w")
        if include_open_button:
            self.create_button(panel, "Open File", self.open_selected_result_file, width=120, height=38).pack(
                anchor="w", pady=(16, 0)
            )

    def open_selected_result_file(self):
        if self.results_preview_selected_path and os.path.exists(self.results_preview_selected_path):
            self.open_path(self.results_preview_selected_path)

    def open_path(self, path):
        if not path:
            return
        try:
            if self.is_windows:
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except OSError as exc:
            messagebox.showerror("Open path failed", str(exc))

    def on_close(self):
        self.close_fullscreen_result_preview()
        self.release_preview_capture()
        self.stop_progress()
        try:
            self.root.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.save_session()
        if self.active_process and self.active_process.poll() is None:
            if not messagebox.askyesno(
                "Close while running?",
                "An analysis process is still running. Close the window and terminate the process?",
            ):
                return
            try:
                self.active_process.terminate()
            except OSError:
                pass
        self.root.destroy()


def main():
    ensure_preview_runtime()

    parser = argparse.ArgumentParser(description="Launch the SickleSight GUI.")
    parser.add_argument(
        "--add-video",
        action="append",
        default=[],
        help="Preload an .mp4 file into the pool by adding its parent folder.",
    )
    parser.add_argument(
        "--add-folder",
        action="append",
        default=[],
        help="Preload a folder containing .mp4 files into the pool.",
    )
    parser.add_argument(
        "--pipeline-dir",
        default=None,
        help="Preselect the folder that contains the SickleSight scripts and model files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Preselect the root folder for analysis results.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    SickleAnalysisGUI(
        root,
        startup_video_paths=args.add_video,
        startup_folder_paths=args.add_folder,
        startup_pipeline_dir=args.pipeline_dir,
        startup_output_dir=args.output_dir,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
