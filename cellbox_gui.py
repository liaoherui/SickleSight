import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import platform
import subprocess
import threading
import datetime
import collections
import stat

class SickleAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sickle Cell Analysis Dashboard v2.7 (Selection-Aware Execution)")
        self.root.geometry("1200x900")
        
        self.is_windows = platform.system() == "Windows"
        
        self.bg_color = "#2b2b2b"
        self.fg_color = "#dcdcdc"
        self.accent_color = "#3a3a3a"
        self.button_color = "#4a4a4a"
        self.highlight_color = "#4caf50" 

        self.root.configure(bg=self.bg_color)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Dark.TFrame", background=self.bg_color)
        self.style.configure("Dark.TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("Dark.TButton", background=self.button_color, foreground="white", borderwidth=1)
        self.style.map("Dark.TButton", background=[("active", "#5a5a5a")])
        self.style.configure("Dark.TRadiobutton", background=self.bg_color, foreground=self.fg_color)
        self.style.map("Dark.TRadiobutton", background=[("active", self.bg_color)])
        self.style.configure("Action.TButton", background=self.highlight_color, foreground="white", font=('Helvetica', 12, 'bold'))

        self.selected_folders = [] 
        self.pipeline_dir = ""
        self.output_dir = "" 
        self.selected_pipeline_var = tk.StringVar()
        
        self.script_output_dir = os.path.join(os.getcwd(), "_tmp_scripts")
        if not os.path.exists(self.script_output_dir):
            os.makedirs(self.script_output_dir)

        self.create_layout()

    def create_layout(self):
        main_container = ttk.Frame(self.root, style="Dark.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ==========================================================
        # TOP PANE: File Tree & Video Preview
        # ==========================================================
        top_pane = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg=self.bg_color, sashwidth=4)
        top_pane.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # --- Left: File Tree ---
        left_frame = ttk.Frame(top_pane, style="Dark.TFrame")
        top_pane.add(left_frame, width=500)

        btn_frame = ttk.Frame(left_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(0, 2))
        
        self.btn_add = ttk.Button(btn_frame, text="+ Add Parent Folder (Smart Scan)", style="Dark.TButton", command=self.add_folders)
        self.btn_add.pack(side=tk.LEFT, padx=(0, 2))
        
        self.btn_remove = ttk.Button(btn_frame, text="- Remove Selected", style="Dark.TButton", command=self.remove_selection)
        self.btn_remove.pack(side=tk.LEFT)

        columns = ("path",)
        self.tree = ttk.Treeview(left_frame, columns=columns, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Input Folders / Video Files", anchor=tk.W)
        self.tree.heading("path", text="Full Path")
        self.tree.column("path", width=0, stretch=False)
        
        ysb = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_double_click_file)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_file)

        # --- Right: Preview ---
        right_frame = ttk.Frame(top_pane, style="Dark.TFrame")
        top_pane.add(right_frame)

        lbl_preview = ttk.Label(right_frame, text="Video Preview", font=('Helvetica', 11, 'bold'), style="Dark.TLabel")
        lbl_preview.pack(pady=(0, 5), anchor=tk.W)

        self.preview_text = tk.Text(right_frame, height=8, bg="#1e1e1e", fg=self.fg_color, relief=tk.FLAT)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.preview_text.insert(tk.END, "Select a video file to see details.\nDouble-click to play.")
        self.preview_text.config(state=tk.DISABLED)

        self.btn_play = ttk.Button(right_frame, text="> Play Selected Video", style="Action.TButton", command=self.play_video)
        self.btn_play.pack(pady=5, fill=tk.X)

        # ==========================================================
        # MIDDLE: Configuration
        # ==========================================================
        config_frame = ttk.LabelFrame(main_container, text="Job Configuration", style="Dark.TFrame", padding=5)
        config_frame.pack(fill=tk.X, pady=(0, 5))

        col1 = ttk.Frame(config_frame, style="Dark.TFrame")
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(col1, text="1. Select Pipeline:", style="Dark.TLabel", font=('Helvetica', 9, 'bold')).pack(anchor=tk.W)
        self.radio_frame = ttk.Frame(col1, style="Dark.TFrame")
        self.radio_frame.pack(fill=tk.BOTH, expand=True)

        col2 = ttk.Frame(config_frame, style="Dark.TFrame")
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(col2, text="2. Setup Locations:", style="Dark.TLabel", font=('Helvetica', 9, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        sf_btn = ttk.Button(col2, text="Select Pipeline Script Folder", style="Dark.TButton", command=self.select_pipeline_folder)
        sf_btn.pack(fill=tk.X, pady=2)
        self.lbl_pipe_path = ttk.Label(col2, text="[Script Folder Not Selected]", style="Dark.TLabel", font=('Consolas', 8))
        self.lbl_pipe_path.pack(anchor=tk.W, pady=(0, 10))

        out_btn = ttk.Button(col2, text="Select Output Folder", style="Dark.TButton", command=self.select_output_folder)
        out_btn.pack(fill=tk.X, pady=2)
        self.lbl_out_path = ttk.Label(col2, text="[Output Folder Not Selected]", style="Dark.TLabel", font=('Consolas', 8))
        self.lbl_out_path.pack(anchor=tk.W)

        col3 = ttk.Frame(config_frame, style="Dark.TFrame")
        col3.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        btn_text = "GENERATE BAT\n&\nRUN ANALYSIS" if self.is_windows else "GENERATE SCRIPT\n&\nRUN ANALYSIS"
        self.btn_run = ttk.Button(col3, text=btn_text, style="Action.TButton", command=self.run_analysis)
        self.btn_run.pack(fill=tk.BOTH, expand=True)

        # ==========================================================
        # BOTTOM: Terminal
        # ==========================================================
        self.term_frame = ttk.LabelFrame(main_container, text="Process Output / Terminal", style="Dark.TFrame")
        self.term_frame.pack(fill=tk.BOTH, expand=True)
        
        term_scroll = ttk.Scrollbar(self.term_frame, orient=tk.VERTICAL)
        self.terminal = tk.Text(self.term_frame, bg="black", fg="#00ff00", font=('Consolas', 10), height=12, yscrollcommand=term_scroll.set)
        
        term_scroll.config(command=self.terminal.yview)
        term_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.terminal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.terminal.insert(tk.END, ">>> Ready.\n")
        self.terminal.config(state=tk.DISABLED)

    # --- Actions ---

    def add_folders(self):
        parent_path = filedialog.askdirectory(title="Select Parent Folder (Auto-detects subfolders with .mp4)")
        if not parent_path:
            return

        added_any = False
        
        for root_dir, dirs, files in os.walk(parent_path):
            has_videos = any(f.lower().endswith(".mp4") for f in files)
            if has_videos:
                clean_path = os.path.normpath(root_dir)
                if clean_path not in self.selected_folders:
                    self.selected_folders.append(clean_path)
                    added_any = True

        if added_any:
            self.refresh_tree()
        else:
            messagebox.showinfo("Scan Complete", "No folders containing .mp4 videos were found inside the selected directory.")

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        parents = collections.defaultdict(list)
        for folder in self.selected_folders:
            parent = os.path.dirname(folder)
            dirname = os.path.basename(folder)
            parents[parent].append((dirname, folder))

        for parent_path, children in parents.items():
            p_node = self.tree.insert("", tk.END, text=parent_path, open=True, values=(parent_path,))
            for dirname, full_path in children:
                f_node = self.tree.insert(p_node, tk.END, text=dirname, open=False, values=(full_path,))
                try:
                    files = [f for f in os.listdir(full_path) if f.lower().endswith(".mp4")]
                    for vid in files:
                        self.tree.insert(f_node, tk.END, text=vid, values=(os.path.join(full_path, vid),))
                except Exception as e:
                    print(f"Error: {e}")

    def remove_selection(self):
        selected_items = self.tree.selection()
        for item in selected_items:
            values = self.tree.item(item, "values")
            if not values: continue
            path = values[0]
            if path in self.selected_folders:
                self.selected_folders.remove(path)
        self.refresh_tree()

    def select_pipeline_folder(self):
        path = filedialog.askdirectory(title="Select folder containing pipeline scripts")
        if path:
            self.pipeline_dir = path
            self.lbl_pipe_path.config(text=path)
            
            for widget in self.radio_frame.winfo_children():
                if isinstance(widget, ttk.Radiobutton):
                    widget.destroy()

            found = False
            detected_scripts = []
            try:
                for file_name in os.listdir(path):
                    if file_name.startswith("cellbox_") and file_name.endswith(".py"):
                        if re.search('gui',file_name):continue
                        detected_scripts.append(file_name)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read directory: {e}")
                return

            detected_scripts.sort()

            for script_name in detected_scripts:
                rb = ttk.Radiobutton(self.radio_frame, text=script_name, variable=self.selected_pipeline_var, value=script_name, style="Dark.TRadiobutton")
                rb.pack(anchor=tk.W, padx=5, pady=2)
                
                if not found:
                    self.selected_pipeline_var.set(script_name)
                    found = True
            
            if not found:
                messagebox.showwarning("Warning", "No files starting with 'pipeline_' and ending with '.py' were found in this folder.")

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_dir = path
            self.lbl_out_path.config(text=path)

    def on_select_file(self, event):
        selected = self.tree.selection()
        if not selected: return
        path = self.tree.item(selected[0], "values")[0]
        
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, f"Selected: {os.path.basename(path)}\nPath: {path}\n")
        self.preview_text.config(state=tk.DISABLED)

    def on_double_click_file(self, event):
        self.play_video()

    def play_video(self):
        selected = self.tree.selection()
        if not selected: return
        path = self.tree.item(selected[0], "values")[0]
        if path and path.lower().endswith(".mp4"):
            try:
                if self.is_windows:
                    os.startfile(path)
                elif platform.system() == 'Darwin':  
                    subprocess.call(['open', path])
                else: 
                    subprocess.call(['xdg-open', path])
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # --- Script Generation & Execution ---

    def run_analysis(self):
        if not self.selected_folders:
            messagebox.showwarning("Error", "No input folders in the pool.")
            return
        if not self.pipeline_dir or not self.selected_pipeline_var.get():
            messagebox.showwarning("Error", "Pipeline not configured.")
            return
        if not self.output_dir:
            messagebox.showwarning("Error", "Output folder not selected.")
            return

        # 1. RESOLVE SELECTED FILES: Prioritize highlighted Treeview items
        selected_items = self.tree.selection()
        target_files = set()

        if selected_items:
            for item in selected_items:
                path = self.tree.item(item, "values")[0]
                if path.lower().endswith(".mp4"):
                    target_files.add(path)
                elif os.path.isdir(path):
                    # If a whole folder is highlighted, add all its mp4s
                    for f in os.listdir(path):
                        if f.lower().endswith(".mp4"):
                            target_files.add(os.path.join(path, f))
        else:
            # Fallback: if nothing is highlighted, process everything in the pool
            for folder in self.selected_folders:
                for f in os.listdir(folder):
                    if f.lower().endswith(".mp4"):
                        target_files.add(os.path.join(folder, f))

        if not target_files:
            messagebox.showwarning("Error", "No valid .mp4 videos found in your selection.")
            return

        # 2. GROUP THE TARGET FILES
        grouped_files = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for file_path in target_files:
            file_path = os.path.normpath(file_path)
            subfolder_path = os.path.dirname(file_path)
            parent_path = os.path.dirname(subfolder_path)
            subfolder_name = os.path.basename(subfolder_path)
            file_name = os.path.basename(file_path)
            
            grouped_files[parent_path][subfolder_name].append(file_name)

        # 3. GENERATE SCRIPT
        ext = ".bat" if self.is_windows else ".sh"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_filename = f"analyze_{timestamp}{ext}"
        script_path = os.path.join(self.script_output_dir, script_filename)

        if self.is_windows:
            script_content = self.generate_bat_content(grouped_files)
        else:
            script_content = self.generate_sh_content(grouped_files)
        
        try:
            with open(script_path, "w", encoding='utf-8') as f:
                f.write(script_content)
                
            self.log_to_terminal(f"Generated Script: {script_path}")
            threading.Thread(target=self.execute_script, args=(script_path,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("File Error", f"Could not create script file: {e}")

    def generate_bat_content(self, grouped_files):
        script_name = self.selected_pipeline_var.get()
        lines = ["@echo off", "setlocal EnableDelayedExpansion"]
        lines.append('set "PYTHONIOENCODING=utf-8"')
        lines.append(f'cd /d "{self.pipeline_dir}"')
        lines.append(f'set "USER_OUTPUT_ROOT={self.output_dir}"')

        for parent_path, subfolders in grouped_files.items():
            lines.append(f'\n:: === Group: {parent_path} ===')
            lines.append(f'set "DATA_PATH={parent_path}"')
            lines.append('for %%A in ("%DATA_PATH%") do set "BASE_FOLDER=%%~nxA"')
            
            for subfolder, files in subfolders.items():
                lines.append(f'\nset "FOLDER={subfolder}"')
                lines.append('echo Processing Selection in: "!FOLDER!"')
                lines.append('set "OUTPUT_FOLDER=!BASE_FOLDER!_!FOLDER!"')
                
                # Write explicitly selected files to the temporary list
                lines.append('set "VIDEO_LIST_FILE=_video_list_tmp_!RANDOM!.txt"')
                lines.append('> "!VIDEO_LIST_FILE!" (')
                for f in files:
                    lines.append(f'    echo !FOLDER!\\{f}')
                lines.append(')')
                
                block = f"""
    set "INPUTS="
    for /f "usebackq delims=" %%V in ("!VIDEO_LIST_FILE!") do (
        if defined INPUTS (
            set "INPUTS=!INPUTS!,%DATA_PATH%\\%%V"
        ) else (
            set "INPUTS=%DATA_PATH%\\%%V"
        )
    )

    set "KMP_DUPLICATE_LIB_OK=TRUE"
    set "OMP_NUM_THREADS=1"

    echo Running {script_name}...
    python "{script_name}" -i "!INPUTS!" -o "%USER_OUTPUT_ROOT%\\!OUTPUT_FOLDER!"

    del "!VIDEO_LIST_FILE!"
"""
                lines.append(block)

        lines.append("\nendlocal")
        lines.append("echo Done.")
        return "\n".join(lines)

    def generate_sh_content(self, grouped_files):
        script_name = self.selected_pipeline_var.get()
        lines = ["#!/bin/bash"]
        lines.append('export PYTHONIOENCODING=utf-8')
        lines.append(f'cd "{self.pipeline_dir}" || exit 1')
        lines.append(f'USER_OUTPUT_ROOT="{self.output_dir}"')

        for parent_path, subfolders in grouped_files.items():
            parent_path_unix = parent_path.replace('\\', '/')
            lines.append(f'\n# === Group: {parent_path_unix} ===')
            lines.append(f'DATA_PATH="{parent_path_unix}"')
            lines.append('BASE_FOLDER=$(basename "$DATA_PATH")')

            for subfolder, files in subfolders.items():
                lines.append(f'\necho "Processing Selection in: {subfolder}"')
                lines.append(f'OUTPUT_FOLDER="${{BASE_FOLDER}}_{subfolder}"')
                
                # Directly construct comma-separated string of explicitly selected files
                file_paths = [f"{parent_path_unix}/{subfolder}/{f}" for f in files]
                inputs_str = ",".join(file_paths)

                lines.append(f'INPUTS="{inputs_str}"')
                lines.append('export KMP_DUPLICATE_LIB_OK=TRUE')
                lines.append('export OMP_NUM_THREADS=1')
                lines.append(f'echo "Running {script_name}..."')
                lines.append(f'python3 "{script_name}" -i "$INPUTS" -o "$USER_OUTPUT_ROOT/$OUTPUT_FOLDER"')

        lines.append('\necho "Done."')
        return "\n".join(lines)

    def execute_script(self, script_path):
        abs_path = os.path.abspath(script_path)
        self.root.after(0, lambda: self.log_to_terminal(f"Starting execution: {os.path.basename(abs_path)}..."))

        if not self.is_windows:
            try:
                st = os.stat(abs_path)
                os.chmod(abs_path, st.st_mode | stat.S_IXUSR)
                self.root.after(0, lambda: self.log_to_terminal("System: Executable permissions set."))
            except Exception as err:
                self.root.after(0, lambda err=err: self.log_to_terminal(f"System Warning: Could not set permissions: {err}"))

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
                encoding='utf-8',
                errors='replace',
                creationflags=creation_flags
            )

            if process.stdout:
                for line in process.stdout:
                    clean_line = line.strip()
                    if clean_line:
                        self.root.after(0, lambda l=clean_line: self.log_to_terminal(l))

            process.wait()

            ret_code = process.returncode
            self.root.after(0, lambda r=ret_code: self.log_to_terminal(f"Process finished with exit code: {r}"))
            self.root.after(0, lambda: messagebox.showinfo("Done", "Analysis Complete."))

        except Exception as err:
            error_msg = f"Failed to start process: {str(err)}"
            self.root.after(0, lambda msg=error_msg: self.log_to_terminal(msg))
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))

    def log_to_terminal(self, message):
        def _log():
            self.terminal.config(state=tk.NORMAL)
            self.terminal.insert(tk.END, f"{message}\n")
            self.terminal.see(tk.END)
            self.terminal.config(state=tk.DISABLED)
        self.root.after(0, _log)

def main():
    root = tk.Tk()
    app = SickleAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
