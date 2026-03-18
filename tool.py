import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import mannwhitneyu
import numpy as np
import json
import os

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------

GROUPS = [
    {'id': 'A', 'label': '0%', 'color': '#60a5fa'},   # Blue
    {'id': 'B', 'label': '30%', 'color': '#34d399'},  # Emerald Green
    {'id': 'C', 'label': '60%', 'color': '#facc15'},  # Yellow
    {'id': 'D', 'label': '100%', 'color': '#f87171'}  # Red
]

SNAME = {0: 'Sickle', 1: "Non-sickle"}
# Class ID map: CSV Integer -> Label
CLASS_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
CLASS_MAP = {label: i for i, label in enumerate(CLASS_LABELS)}

# --- Nature Style Colors ---
COLOR_NS_AR = "#2878B5"  # Science Blue
COLOR_S_AR = "#C82423"   # Nature Red
PALETTE_AR = {0: COLOR_S_AR, 1: COLOR_NS_AR, 'Sickle': COLOR_S_AR, 'Non-sickle': COLOR_NS_AR}

COLOR_NS_ECC = "#9C27B0" # Purple
COLOR_S_ECC = "#FF6F00"  # Deep Orange
PALETTE_ECC = {0: COLOR_S_ECC, 1: COLOR_NS_ECC, 'Sickle': COLOR_S_ECC, 'Non-sickle': COLOR_NS_ECC}

COLOR_NS_CIRC = "#4CAF50" # Green
COLOR_S_CIRC = "#E91E63"  # Pink
PALETTE_CIRC = {0: COLOR_S_CIRC, 1: COLOR_NS_CIRC, 'Sickle': COLOR_S_CIRC, 'Non-sickle': COLOR_NS_CIRC}

# Increase font scale for readability
sns.set_theme(style="ticks", font_scale=1.3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_star_string(p_value):
    if p_value > 0.05: return 'ns'
    elif p_value > 0.01: return '*'
    elif p_value > 0.001: return '**'
    elif p_value > 0.0001: return '***'
    else: return '****'

def draw_stat_annotation(ax, x1, x2, y, h, p_val, color='k'):
    star_str = get_star_string(p_val)
    # Draw bracket: down-tick, horizontal line, down-tick
    # x1, x2 are the x-coordinates (indices) of the groups
    # y is the base of the legs, y+h is the crossbar height
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    # Text above the bracket
    ax.text((x1 + x2) * .5, y + h, star_str, ha='center', va='bottom', color=color, fontsize=12, weight='bold')

def plot_nature_style(ax, df, x_col, y_col, hue_col=None, palette=None, order=None, 
                      title="", y_label="", show_legend=True, 
                      y_min=None, y_max=None, show_stats=True, simplify_ns=False,
                      stats_style="Legend", # "Legend" or "Bracket"
                      label_normal="Non-sickle", label_sickle="Sickle"):
    if df.empty: return

    # Determine what to use for coloring. 
    actual_hue = hue_col if hue_col else x_col

    # 1. Violin Plot
    sns.violinplot(data=df, x=x_col, y=y_col, hue=actual_hue, palette=palette, 
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=order, dodge=False, legend=False)

    # 2. Box Plot
    sns.boxplot(data=df, x=x_col, y=y_col, hue=actual_hue, width=0.15,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9, 'zorder': 2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.5},
                capprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, order=order, dodge=False)

    # 3. Strip Plot
    sns.stripplot(data=df, x=x_col, y=y_col, hue=actual_hue, palette=palette,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=order, dodge=False, legend=False)

    # Simplified X-Axis Labels
    current_labels = [item.get_text() for item in ax.get_xticklabels()]
    simplified_labels = [lbl.split('\n')[0] for lbl in current_labels]
    ax.set_xticklabels(simplified_labels)

    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Collect handles for legend
    legend_handles = []
    
    # 4. Custom Legend
    if show_legend and 'Sickle_Label' in df.columns:
        n_total = len(df)
        if n_total > 0:
            n_s = len(df[df['Sickle_Label'] == 0])
            n_ns = len(df[df['Sickle_Label'] == 1])
            
            c_ns = palette.get(1, 'blue') if palette else 'blue'
            c_s = palette.get(0, 'red') if palette else 'red'

            legend_handles.append(mlines.Line2D([], [], color=c_ns, linewidth=3, label=f'{label_normal} (n={n_ns})'))
            legend_handles.append(mlines.Line2D([], [], color=c_s, linewidth=3, label=f'{label_sickle} (n={n_s})'))

    # 5. Statistical Annotation
    max_bracket_y = 0 # To track top of brackets for auto-scaling
    
    if show_stats:
        if order is not None:
            # We must use list(order) to ensure we can find indices
            unique_groups = [g for g in order if g in df[x_col].unique()]
            full_order = list(order) # This contains the strings in the x-axis order
        else:
            unique_groups = df[x_col].unique()
            full_order = list(unique_groups)
        
        # Check if we have at least 2 groups to compare
        if len(unique_groups) >= 2:
            g1 = unique_groups[0] # Reference group (string label)
            d1 = df[df[x_col] == g1][y_col]
            
            # Robust range calculation (excluding infs)
            clean_series = df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_series.empty:
                clean_max = clean_series.max()
                clean_min = clean_series.min()
            else:
                clean_max = 1.0
                clean_min = 0.0
            
            clean_range = clean_max - clean_min
            if clean_range == 0: clean_range = 1.0

            # --- Skyline Logic for Brackets ---
            # Initialize skyline with max data value for each x-position
            n_cols = len(full_order)
            skyline = [0.0] * n_cols
            
            # Calculate data max per column to establish baseline skyline
            for i, grp in enumerate(full_order):
                grp_data = df[df[x_col] == grp][y_col]
                # Clean local group data
                clean_grp = grp_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if not clean_grp.empty:
                    m = clean_grp.max()
                    # Add small buffer above data using robust range
                    skyline[i] = m + (clean_range * 0.05)
            
            # Global Step size
            h_step = clean_range * 0.1
            bracket_leg_h = h_step * 0.4
            
            ns_counter = 0 
            
            # Helper to clean labels for Legend Style
            def clean_lbl(l):
                parts = l.split('\n')
                if len(parts) >= 2:
                    return f"{parts[0]} {parts[1]}"
                return l.replace('\n', ' ')

            for i in range(1, len(unique_groups)):
                g2 = unique_groups[i]
                d2 = df[df[x_col] == g2][y_col]
                
                if len(d1) > 1 and len(d2) > 1:
                    try:
                        stat, p_val = mannwhitneyu(d1, d2, alternative='two-sided')
                        star_str = get_star_string(p_val)
                        
                        is_ns = (p_val > 0.05)
                        if simplify_ns and is_ns:
                            if ns_counter > 0: continue
                            ns_counter += 1
                        
                        if stats_style == "Legend":
                            lbl2 = clean_lbl(g2)
                            legend_text = f"vs {lbl2}: {star_str}"
                            legend_handles.append(mlines.Line2D([], [], color='none', label=legend_text))
                        else: # Bracket
                            try:
                                x_idx_1 = full_order.index(g1)
                                x_idx_2 = full_order.index(g2)
                                
                                # Find range max in skyline
                                start = min(x_idx_1, x_idx_2)
                                end = max(x_idx_1, x_idx_2)
                                
                                # Current height needed
                                current_max = max(skyline[start : end+1])
                                draw_y = current_max + (h_step * 0.2)
                                
                                draw_stat_annotation(ax, x_idx_1, x_idx_2, draw_y, bracket_leg_h, p_val)
                                
                                # Update skyline
                                new_height = draw_y + h_step
                                for k in range(start, end+1):
                                    skyline[k] = new_height
                                    
                                max_bracket_y = max(max_bracket_y, new_height)
                                
                            except ValueError:
                                print(f"Could not find index for {g1} or {g2}")
                            
                        print(f"Stats: {g1} vs {g2} | p-value={p_val:.4e}")
                        
                    except Exception as e:
                        print(f"Stat calc failed: {e}")
            
    # Draw Legend
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=9, 
                  framealpha=0.9, edgecolor='gray', ncol=2)

    # Apply Limits
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
        
    if y_max is not None:
        ax.set_ylim(top=y_max)
    elif stats_style == "Bracket" and max_bracket_y > 0:
        # Auto-expand to fit brackets if no manual max set
        current_top = ax.get_ylim()[1]
        ax.set_ylim(top=max(current_top, max_bracket_y + (max_bracket_y * 0.05)))

    sns.despine(offset=10, trim=True, ax=ax)

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------

class SickleAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sickle Cell Analysis Pro")
        self.root.geometry("520x950")
        
        # Data storage
        self.data_store = {g['id']: {'filepath': None, 'df': None} for g in GROUPS}
        self.selections = {g['id']: {} for g in GROUPS} 
        # Store class states per group
        self.class_states = {g['id']: {c: True for c in CLASS_LABELS} for g in GROUPS}
        
        self.check_vars = {} 
        self.frame_vars = {}
        self.class_vars = {} 
        self.title_vars = {}
        self.comp_title_var = tk.StringVar(value="Composite Analysis")
        
        # Toggles
        self.show_legend_var = tk.BooleanVar(value=True) 
        self.show_threshold_var = tk.BooleanVar(value=True) 
        self.show_stats_var = tk.BooleanVar(value=True) 
        self.simplify_ns_var = tk.BooleanVar(value=False)
        self.stats_style_var = tk.StringVar(value="Legend") 
        
        # Range Variables
        self.y_min_var = tk.StringVar(value="")
        self.y_max_var = tk.StringVar(value="")
        
        # Label Variables
        self.lbl_normal_var = tk.StringVar(value="Non-sickle")
        self.lbl_sickle_var = tk.StringVar(value="Sickle")
        
        self.setup_ui()

    def setup_ui(self):
        # Top container
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Sickle Analysis Pro", font=("Helvetica", 16, "bold")).pack()
        ttk.Label(top_frame, text="Nature-style Visualization", font=("Helvetica", 9, "italic"), foreground="gray").pack()

        # Metric Selection
        type_frame = ttk.LabelFrame(top_frame, text="Analysis Metric (Y-Axis)", padding=5)
        type_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.metric_var = tk.StringVar(value="Aspect_Ratio")
        metrics = [("Aspect Ratio", "Aspect_Ratio"), ("Eccentricity", "Eccentricity"), ("Circularity", "Circularity")]
        
        for i, (label, val) in enumerate(metrics):
            ttk.Radiobutton(type_frame, text=label, variable=self.metric_var, value=val).pack(side=tk.LEFT, padx=5)

        # Options Frame
        opts_frame = ttk.Frame(top_frame)
        opts_frame.pack(fill=tk.X, pady=5)
        
        # Row 1: Toggles
        row1 = ttk.Frame(opts_frame)
        row1.pack(fill=tk.X, anchor='w')
        ttk.Checkbutton(row1, text="Legend", variable=self.show_legend_var).pack(side=tk.LEFT, padx=(0,10))
        ttk.Checkbutton(row1, text="AR Threshold", variable=self.show_threshold_var).pack(side=tk.LEFT, padx=(0,10))
        ttk.Checkbutton(row1, text="Show Stats", variable=self.show_stats_var).pack(side=tk.LEFT, padx=(0,10))
        ttk.Checkbutton(row1, text="Single NS", variable=self.simplify_ns_var).pack(side=tk.LEFT, padx=(0,10))

        # Row 2: Stats Style & Y-Axis
        row2 = ttk.Frame(opts_frame)
        row2.pack(fill=tk.X, anchor='w', pady=(5,0))
        
        ttk.Label(row2, text="Stats Loc:").pack(side=tk.LEFT)
        style_combo = ttk.Combobox(row2, textvariable=self.stats_style_var, values=["Legend", "Bracket"], state="readonly", width=8)
        style_combo.pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row2, text="Y-Min:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.y_min_var, width=5).pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(row2, text="Y-Max:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.y_max_var, width=5).pack(side=tk.LEFT, padx=(2,0))

        # Row 3: Custom Labels
        row3 = ttk.Frame(opts_frame)
        row3.pack(fill=tk.X, anchor='w', pady=(5,0))
        ttk.Label(row3, text="Norm Label:").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.lbl_normal_var, width=10).pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(row3, text="Sickle Label:").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.lbl_sickle_var, width=10).pack(side=tk.LEFT, padx=(2,0))

        # --- SCROLLABLE AREA ---
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scroll_window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.columnconfigure(0, weight=1)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Groups
        for group in GROUPS:
            self.create_group_section(self.scrollable_frame, group)

        # Bottom container
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        comp_frame = ttk.LabelFrame(bottom_frame, text="Composite Figure", padding=5)
        comp_frame.pack(fill=tk.X)
        
        tf_frame = ttk.Frame(comp_frame)
        tf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tf_frame, text="Title Base:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        ttk.Entry(tf_frame, textvariable=self.comp_title_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        btn_composite = ttk.Button(comp_frame, text="Generate Comparison Figure", command=self.plot_composite)
        btn_composite.pack(fill=tk.X, ipady=3, pady=5)
        
        # Config Buttons
        cfg_frame = ttk.Frame(bottom_frame)
        cfg_frame.pack(fill=tk.X, pady=5)
        ttk.Button(cfg_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        ttk.Button(cfg_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,2))
        ttk.Button(cfg_frame, text="Clear", command=self.clear_selections).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.scroll_window_id, width=event.width)

    def create_group_section(self, parent, group_conf):
        gid = group_conf['id']
        label = group_conf['label']
        
        frame = ttk.LabelFrame(parent, text=f"{label} Concentration", padding=5)
        frame.pack(fill=tk.X, pady=5, padx=5, expand=True)

        # Row 1
        r1 = ttk.Frame(frame)
        r1.pack(fill=tk.X)
        status_lbl = ttk.Label(r1, text="No Data", foreground="gray", font=("Helvetica", 8))
        status_lbl.pack(side=tk.LEFT)
        btn_upload = ttk.Button(r1, text="Load CSV", width=8,
                                command=lambda: self.upload_csv(gid, status_lbl, frame_combo, btn_plot, c0, c1))
        btn_upload.pack(side=tk.RIGHT)

        # Row 2
        r2 = ttk.Frame(frame)
        r2.pack(fill=tk.X, pady=4)
        
        f_sub = ttk.Frame(r2)
        f_sub.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(f_sub, text="Frame:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.frame_vars[gid] = tk.StringVar()
        frame_combo = ttk.Combobox(f_sub, textvariable=self.frame_vars[gid], state="disabled", width=8)
        frame_combo.pack(side=tk.LEFT, padx=(5,5))
        frame_combo.bind("<<ComboboxSelected>>", lambda e, g=gid: self.on_frame_changed(g))

        # UPDATED: Class Selector (Multi-Select)
        ttk.Label(f_sub, text="Class:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.class_vars[gid] = tk.StringVar(value="Select Class")
        class_combo = ttk.Combobox(f_sub, textvariable=self.class_vars[gid], state="readonly", width=8)
        class_combo.pack(side=tk.LEFT, padx=(5,5))
        class_combo.bind("<<ComboboxSelected>>", lambda e, g=gid: self.on_class_toggle(g))

        t_sub = ttk.Frame(r2)
        t_sub.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Label(t_sub, text="Title Base:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.title_vars[gid] = tk.StringVar(value=f"{label} Concentration")
        ttk.Entry(t_sub, textvariable=self.title_vars[gid], width=12).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Row 3
        btn_plot = ttk.Button(frame, text=f"Plot {label} Distribution (Current Frame)", state=tk.DISABLED,
                              command=lambda: self.plot_individual(gid))
        btn_plot.pack(fill=tk.X, pady=5)

        # Row 4
        r4 = ttk.Frame(frame)
        r4.pack(fill=tk.X)
        v0 = tk.BooleanVar()
        self.check_vars[f"{gid}_0"] = v0
        c0 = ttk.Checkbutton(r4, text="Normal", variable=v0, state=tk.DISABLED,
                             command=lambda: self.toggle_selection(gid, 0))
        c0.pack(side=tk.LEFT, padx=5)

        v1 = tk.BooleanVar()
        self.check_vars[f"{gid}_1"] = v1
        c1 = ttk.Checkbutton(r4, text="Sickle", variable=v1, state=tk.DISABLED,
                             command=lambda: self.toggle_selection(gid, 1))
        c1.pack(side=tk.LEFT, padx=5)

        self.data_store[gid]['widgets'] = [btn_plot, frame_combo, c0, c1, frame_combo, status_lbl, class_combo]

        # Initialize class combo values
        self.update_class_combo_visuals(gid)

    def update_class_combo_visuals(self, gid):
        # Build list like ["Select All", "Deselect All", "A (✓)", "B", ...]
        values = ["Select All", "Deselect All"]
        for cls_label in CLASS_LABELS:
            is_sel = self.class_states[gid][cls_label]
            if is_sel:
                values.append(f"{cls_label} (✓)")
            else:
                values.append(cls_label)
        
        if 'widgets' in self.data_store[gid] and len(self.data_store[gid]['widgets']) > 6:
            self.data_store[gid]['widgets'][6]['values'] = values

    def on_class_toggle(self, gid):
        widget = self.data_store[gid]['widgets'][6]
        selection = widget.get()
        if not selection: return
        
        if selection == "Select All":
            for c in CLASS_LABELS: self.class_states[gid][c] = True
        elif selection == "Deselect All":
            for c in CLASS_LABELS: self.class_states[gid][c] = False
        else:
            cls_label = selection.split()[0]
            if cls_label in self.class_states[gid]:
                self.class_states[gid][cls_label] = not self.class_states[gid][cls_label]
        
        self.update_class_combo_visuals(gid)
        widget.set("Select Class") 

    def filter_by_class(self, df, gid):
        checked_labels = [c for c, state in self.class_states[gid].items() if state]
        checked_ids = [CLASS_MAP[c] for c in checked_labels]
        if not checked_ids:
            return df.iloc[0:0] 
        return df[df['Class_ID'].isin(checked_ids)]

    def upload_csv(self, gid, status_lbl, frame_combo, btn_plot, c0, c1, filepath=None):
        if not filepath:
            filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath: return

        try:
            df = pd.read_csv(filepath)
            required = ['Sickle_Label', 'Frame_Index', 'Class_ID']
            if not all(col in df.columns for col in required):
                messagebox.showerror("Error", f"CSV {filepath} missing required columns.")
                return
            
            df['Condition'] = df['Sickle_Label'].map(SNAME)
            for col in ['Aspect_Ratio', 'Eccentricity', 'Circularity']:
                if col not in df.columns: df[col] = np.nan

            frames = sorted(df['Frame_Index'].unique())
            
            self.data_store[gid].update({
                'df': df,
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'frames_list': frames,
                'combo_widget': frame_combo 
            })
            
            if not self.selections[gid]:
                self.selections[gid] = {f: {0: False, 1: False} for f in frames}
            
            for f in frames:
                if f not in self.selections[gid]:
                    self.selections[gid][f] = {0: False, 1: False}

            status_lbl.config(text=f"Rows: {len(df)}", foreground="green")
            self.update_combo_visuals(gid)
            
            if frames and not self.frame_vars[gid].get():
                frame_combo.set(frame_combo['values'][0]) 
                self.on_frame_changed(gid)
                
            frame_combo.config(state="readonly")
            btn_plot.config(state=tk.NORMAL)
            c0.config(state=tk.NORMAL)
            c1.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Parse Error:\n{e}")

    def on_frame_changed(self, gid):
        raw_val = self.frame_vars[gid].get()
        if not raw_val: return
        try:
            frame_id = int(raw_val.split()[0])
        except: return

        if gid in self.selections and frame_id in self.selections[gid]:
            state = self.selections[gid][frame_id]
            self.check_vars[f"{gid}_0"].set(state[0]) 
            self.check_vars[f"{gid}_1"].set(state[1]) 

    def toggle_selection(self, gid, type_idx):
        raw_val = self.frame_vars[gid].get()
        if not raw_val: return
        try:
            frame_id = int(raw_val.split()[0])
        except: return

        is_checked = self.check_vars[f"{gid}_{type_idx}"].get()
        self.selections[gid][frame_id][type_idx] = is_checked
        self.update_combo_visuals(gid)
        
        has_any = self.selections[gid][frame_id][0] or self.selections[gid][frame_id][1]
        suffix = " (✓)" if has_any else ""
        self.frame_vars[gid].set(f"{frame_id}{suffix}")

    def update_combo_visuals(self, gid):
        if 'frames_list' not in self.data_store[gid]: return
        frames = self.data_store[gid]['frames_list']
        display_values = []
        for f in frames:
            state = self.selections[gid].get(f, {0: False, 1: False})
            if state[0] or state[1]:
                display_values.append(f"{f} (✓)")
            else:
                display_values.append(str(f))
        self.data_store[gid]['combo_widget']['values'] = display_values

    def get_palette(self, metric):
        if metric == 'Aspect_Ratio': return PALETTE_AR
        if metric == 'Eccentricity': return PALETTE_ECC
        if metric == 'Circularity': return PALETTE_CIRC
        return None

    def get_plotting_params(self):
        y_min, y_max = None, None
        if self.y_min_var.get().strip(): 
            try: y_min = float(self.y_min_var.get())
            except: pass
        if self.y_max_var.get().strip(): 
            try: y_max = float(self.y_max_var.get())
            except: pass
        return y_min, y_max

    def plot_individual(self, gid):
        data = self.data_store[gid]
        if not data or 'df' not in data or data['df'] is None: return

        try:
            raw_val = self.frame_vars[gid].get()
            selected_frame = int(raw_val.split()[0])
        except (ValueError, IndexError):
            messagebox.showwarning("Warning", "Please select a valid frame.")
            return

        metric = self.metric_var.get()
        df = data['df']
        
        subset = df[(df['Frame_Index'] == selected_frame)]
        subset = self.filter_by_class(subset, gid)
        subset = subset.dropna(subset=[metric])
        
        if subset.empty:
            messagebox.showwarning("Warning", f"No data for Frame {selected_frame}")
            return

        lbl_norm = self.lbl_normal_var.get()
        lbl_sickle = self.lbl_sickle_var.get()
        
        plot_df = subset.copy()
        plot_df['Condition'] = plot_df['Sickle_Label'].map({0: lbl_sickle, 1: lbl_norm})

        fig, ax = plt.subplots(figsize=(6, 7))
        y_min, y_max = self.get_plotting_params()
        
        base_title = self.title_vars[gid].get()
        final_title = f"{base_title} (Frame {selected_frame})"

        plot_nature_style(
            ax=ax, df=plot_df, x_col='Condition', y_col=metric,
            palette=self.get_palette(metric), order=[lbl_norm, lbl_sickle],
            title=final_title, y_label=metric.replace('_', ' '),
            show_legend=self.show_legend_var.get(),
            hue_col='Sickle_Label',
            y_min=y_min, y_max=y_max,
            show_stats=self.show_stats_var.get(),
            simplify_ns=self.simplify_ns_var.get(),
            stats_style=self.stats_style_var.get(),
            label_normal=lbl_norm, label_sickle=lbl_sickle
        )
        
        if metric == 'Aspect_Ratio' and self.show_threshold_var.get():
            ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_composite(self):
        metric = self.metric_var.get()
        composite_data = []
        
        lbl_norm = self.lbl_normal_var.get()
        lbl_sickle = self.lbl_sickle_var.get()
        
        for group_conf in GROUPS:
            gid = group_conf['id']
            if gid not in self.selections or 'df' not in self.data_store[gid] or self.data_store[gid]['df'] is None:
                continue
            df = self.data_store[gid]['df']
            sorted_frames = sorted(self.selections[gid].keys())
            
            for frame_id in sorted_frames:
                state = self.selections[gid][frame_id]
                targets = []
                if state[0]: targets.append(1) 
                if state[1]: targets.append(0) 
                
                if not targets: continue
                frame_subset = df[df['Frame_Index'] == frame_id]
                frame_subset = self.filter_by_class(frame_subset, gid)
                
                for target_label in targets:
                    subset = frame_subset[frame_subset['Sickle_Label'] == target_label].dropna(subset=[metric])
                    if subset.empty: continue
                    
                    status_str = lbl_norm if target_label == 1 else lbl_sickle
                    
                    temp = pd.DataFrame({
                        'Value': subset[metric],
                        'Label': f"{group_conf['label']}\n{status_str}\n(F{frame_id})",
                        'Type': status_str,
                        'Sickle_Label': target_label
                    })
                    composite_data.append(temp)

        if not composite_data:
            messagebox.showinfo("Info", "No data selected.")
            return

        final_df = pd.concat(composite_data, ignore_index=True)
        width = max(6, len(composite_data) * 1.5 + 2)
        fig, ax = plt.subplots(figsize=(width, 7))
        y_min, y_max = self.get_plotting_params()
        
        base_title = self.comp_title_var.get()
        final_title = f"{base_title}: {metric.replace('_', ' ')}"

        plot_nature_style(
            ax=ax, df=final_df, x_col='Label', y_col='Value',
            palette=self.get_palette(metric), order=final_df['Label'].unique(),
            title=final_title, y_label=metric.replace('_', ' '),
            show_legend=self.show_legend_var.get(),
            hue_col='Sickle_Label', y_min=y_min, y_max=y_max,
            show_stats=self.show_stats_var.get(),
            simplify_ns=self.simplify_ns_var.get(),
            stats_style=self.stats_style_var.get(),
            label_normal=lbl_norm, label_sickle=lbl_sickle
        )
        
        if metric == 'Aspect_Ratio' and self.show_threshold_var.get():
            ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def clear_selections(self):
        for gid in self.selections:
            if self.selections[gid]:
                for frame in self.selections[gid]:
                    self.selections[gid][frame] = {0: False, 1: False}
                self.update_combo_visuals(gid)
                if 'combo_widget' in self.data_store[gid]:
                    raw = self.frame_vars[gid].get()
                    if raw:
                        clean = raw.split()[0]
                        self.frame_vars[gid].set(clean)
                        self.check_vars[f"{gid}_0"].set(False)
                        self.check_vars[f"{gid}_1"].set(False)

    def save_config(self):
        config = {
            'metric': self.metric_var.get(),
            'show_legend': self.show_legend_var.get(),
            'show_threshold': self.show_threshold_var.get(),
            'show_stats': self.show_stats_var.get(),
            'simplify_ns': self.simplify_ns_var.get(),
            'stats_style': self.stats_style_var.get(),
            'y_min': self.y_min_var.get(),
            'y_max': self.y_max_var.get(),
            'comp_title': self.comp_title_var.get(),
            'label_normal': self.lbl_normal_var.get(),
            'label_sickle': self.lbl_sickle_var.get(),
            'groups': {}
        }
        
        for gid in self.selections:
            serializable_selections = {str(k): v for k, v in self.selections[gid].items()}
            config['groups'][gid] = {
                'filepath': self.data_store[gid].get('filepath'),
                'title': self.title_vars[gid].get(),
                'class_states': self.class_states[gid],
                'selections': serializable_selections
            }
            
        try:
            with open('sickle_analysis_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def load_config(self):
        if not os.path.exists('sickle_analysis_config.json'):
            messagebox.showwarning("Error", "No config file found.")
            return
            
        try:
            with open('sickle_analysis_config.json', 'r') as f:
                config = json.load(f)
            
            self.metric_var.set(config.get('metric', 'Aspect_Ratio'))
            self.show_legend_var.set(config.get('show_legend', True))
            self.show_threshold_var.set(config.get('show_threshold', True))
            self.show_stats_var.set(config.get('show_stats', True))
            self.simplify_ns_var.set(config.get('simplify_ns', False))
            self.stats_style_var.set(config.get('stats_style', 'Legend'))
            self.y_min_var.set(config.get('y_min', ''))
            self.y_max_var.set(config.get('y_max', ''))
            self.comp_title_var.set(config.get('comp_title', 'Composite Analysis'))
            self.lbl_normal_var.set(config.get('label_normal', 'Non-sickle'))
            self.lbl_sickle_var.set(config.get('label_sickle', 'Sickle'))
            
            groups_conf = config.get('groups', {})
            for gid, g_data in groups_conf.items():
                self.title_vars[gid].set(g_data.get('title', ''))
                saved_classes = g_data.get('class_states')
                if saved_classes:
                    self.class_states[gid] = saved_classes
                
                path = g_data.get('filepath')
                if path and os.path.exists(path):
                    w = self.data_store[gid]['widgets'] 
                    self.upload_csv(gid, w[5], w[1], w[0], w[2], w[3], filepath=path)
                    saved_sels = g_data.get('selections', {})
                    for s_frame_str, state in saved_sels.items():
                        try:
                            f_int = int(s_frame_str)
                            if f_int in self.selections[gid]:
                                self.selections[gid][f_int] = {0: state['0'], 1: state['1']}
                        except: pass
                    self.update_combo_visuals(gid)
                    self.update_class_combo_visuals(gid) 
                    self.on_frame_changed(gid) 
            
            messagebox.showinfo("Success", "Configuration loaded.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    try: style = ttk.Style(); style.theme_use('clam')
    except: pass
    app = SickleAnalysisApp(root)
    root.mainloop()