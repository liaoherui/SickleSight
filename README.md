# SickleSight

**SickleSight** is an AI-based toolkit for quantitative analysis of sickle cell disease dynamics from video microscopy data. It combines cell segmentation/tracking, Vision Transformer (ViT) classification, Siamese network state detection, and morphology analysis to produce CSV reports, publication-quality figures, and annotated videos.


---

## Features

- **7-class morphological classification** of sickle cells (classes A–G) at frame 0 using a fine-tuned ViT model
- **Temporal state tracking**: sickled vs. non-sickled detection across all frames via a Siamese ViT comparison network
- **Pocked/non-pocked classification** for an additional morphological dimension
- **Morphological metrics**: aspect ratio (AR), eccentricity (ECC), and circularity at any target frame
- **Publication-ready figures**: Nature-style violin plots, pie charts, and multi-frame trend plots
- **Statistical analysis**: Mann–Whitney U tests comparing sickled vs. non-sickled populations
- **Batch processing**: process multiple videos in one run
- **Cross-platform GUI** (Windows / macOS / Linux) for point-and-click operation
- **Command-line interface**: all three analysis scripts can also be run directly from the terminal
- **Low-resolution mode**: optional YOLO/BoT-SORT tracking backend for videos where Cellpose segmentation is unreliable

---

## Repository Structure

```
SickleSight/
├── CellBox-Models/           # Model folder used by default
├── sicklesight_gui.py        # GUI application — launch this to open the graphical interface
├── sicklesight_part1.py      # Pipeline 1: temporal state-ratio analysis
├── sicklesight_part2.py      # Pipeline 2: multi-frame morphology analysis (AR / ECC / circularity)
├── sicklesight_merged.py     # Pipeline 3: combined — state-ratio + morphology in one pass
├── sicklesight_env.yaml      # Conda environment specification
├── index.html                # Browser-based image labeling tool
├── tool.py                   # Post-processing statistical analysis & visualization GUI
├── Sicklesight_runtest.mp4   # GUI walkthrough video for setup and test runs
└── README.md
```

---

## Supplementary Tools

Two additional standalone tools are included for data preparation and post-processing. Each has its own detailed documentation:

| Tool | Description | Documentation |
|------|-------------|---------------|
| `index.html` | Browser-based image labeler for manually classifying cell crops into morphological classes | [Web Labeler Guide](docs/web-labeler.md) |
| `tool.py` | Desktop GUI for generating publication-quality violin plots with statistical annotations from pipeline CSV outputs | [Analysis Tool Guide](docs/analysis-tool.md) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/liaoherui/sicklesight.git
cd sicklesight
```

### 2. Create the Conda environment

```bash
conda env create -f sicklesight_env.yaml
conda activate sicklesight
```

The environment includes `ultralytics`, which is required for the low-resolution YOLO/BoT-SORT backend.

#### Windows NVIDIA GPU / CUDA setup

On Windows, having an NVIDIA GPU is not enough by itself: PyTorch must be installed as a CUDA-enabled build. If SickleSight prints `Using PyTorch device: cpu` on a GPU laptop, the most common cause is that the environment contains a CPU-only PyTorch build, or the NVIDIA driver is too old for the installed CUDA wheel.

After creating and activating the Conda environment, check what PyTorch sees:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

Choose the PyTorch install command that matches your Windows GPU:

<details>
<summary><strong>Option A — RTX 50-series / Blackwell, e.g. RTX 5090, 5080, 5070</strong></summary>

RTX 50-series GPUs use compute capability 12.0. Use a CUDA 12.8 or newer PyTorch build. Older CUDA wheels can detect the GPU but still fail at runtime with `CUDA error: no kernel image is available for execution on the device`.

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If CUDA still fails, update the NVIDIA driver and choose the newest Windows / Pip / CUDA option from the official PyTorch install selector: <https://pytorch.org/get-started/locally/>.

</details>

<details>
<summary><strong>Option B — RTX 40/30/20 series, GTX 16 series, or other non-RTX-50 NVIDIA GPUs</strong></summary>

These GPUs usually work with the regular CUDA wheels from the PyTorch install selector. CUDA 12.6 is a conservative choice for many non-RTX-50 Windows laptops; CUDA 12.8 is also fine if your NVIDIA driver is new enough.

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If `cu126` fails or your driver is very new, choose the current Windows / Pip / CUDA command from the official PyTorch install selector: <https://pytorch.org/get-started/locally/>.

</details>

<details>
<summary><strong>Option C — No NVIDIA GPU, or CPU-only fallback</strong></summary>

Use this only when the machine has no CUDA-capable NVIDIA GPU, or when you intentionally want CPU execution.

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CPU mode is supported but significantly slower.

</details>

After installing the selected PyTorch build, verify again with the one-line check above. Then run a tiny CUDA convolution smoke test for any NVIDIA GPU setup:

```powershell
python -c "import torch; print('torch:', torch.__version__, 'cuda build:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0)); print('arch list:', torch.cuda.get_arch_list()); x=torch.randn(1,3,32,32,device='cuda'); m=torch.nn.Conv2d(3,8,3).cuda(); y=m(x); torch.cuda.synchronize(); print('CUDA conv smoke test: OK', y.shape)"
```

SickleSight will use CUDA when this smoke test passes. If `torch.cuda.is_available()` is `True` but the smoke test reports `no kernel image is available`, the installed PyTorch wheel is too old for the GPU architecture; install a newer CUDA wheel such as `cu128` or newer.

### 3. Download pre-trained models and test videos

> [!IMPORTANT]
> **Recommended download:** pre-trained models and sample test videos are available together on figshare: **[Download SickleSight models and test videos](https://doi.org/10.6084/m9.figshare.33058646)**.

#### Pre-trained model download

Model files are too large to store directly in GitHub. The Dropbox link below is kept as a separate mirror for the model package. Download **CellBox-Models.zip**, then unzip it into the repository root:

> **[Download CellBox-Models.zip](https://www.dropbox.com/scl/fi/0a4seofjgniqbq4i6x0f3/CellBox-Models.zip?rlkey=6cwzmzxrzs86t19az6eioi89q&st=xail4oqy&dl=0)**

After extraction, the repository should contain:

```text
SickleSight/
└── CellBox-Models/
    ├── cyto3_train0327
    ├── best_model_vit_torch_macos_seven.pth
    ├── best_model_vit_torch_macos_raw_vit_large_binary.pth
    ├── best_model_vit_torch_macos_raw_vit_large_binary_pocked.pth
    ├── direct_vit_D.pt
    ├── direct_vit_E.pt
    ├── direct_vit_G.pt
    ├── siamese_vit_All_Haolin.pt
    ├── yolo/best.pt
    ├── seg/best.pt
    ├── efficientnet/fold1_best.pth ... fold5_best.pth
    ├── siamese/model.pth
    └── configs/botsort_cell.yaml
```

The standard SickleSight backend uses the `.pth/.pt` and `cyto3_train0327` files. The low-resolution backend uses `yolo/`, `seg/`, and `configs/botsort_cell.yaml`.

#### Test videos

Optional sample videos are available for a quick end-to-end check. They are included in the recommended figshare download above, or can be downloaded separately from the Dropbox mirror below:

> **[Download test_video.zip](https://www.dropbox.com/scl/fi/pa5mj69c08vqpy8p0l2ic/test_video.zip?rlkey=saw6x2rihwh7f85p64fgll3qr&st=y4xefc8o&dl=0)**

The archive contains two `.mp4` files: one high-resolution video for the Cellpose workflow and one low-resolution video for the YOLO/BoT-SORT workflow. After unzipping, you can test from the GUI by adding the video file, choosing `sicklesight_merged.py`, selecting the matching backend, setting **Max seconds** to a short value such as `10`, then clicking **Check** and **Run**.

Command-line smoke test examples:

```bash
python sicklesight_merged.py -i path/to/high_resolution_video.mp4 -o outputs/test_high_res --max_time 10 --tracking_backend cellpose
python sicklesight_merged.py -i path/to/low_resolution_video.mp4 -o outputs/test_low_res --max_time 10 --tracking_backend low_res
```

Each run should create a per-video output folder containing files such as `frame_0_segmentation.png`, CSV reports, plots, and an annotated video.

---

## GUI Walkthrough Video

For a visual step-by-step example, watch the included GUI walkthrough video:

> **[Click to play the SickleSight GUI walkthrough video](Sicklesight_runtest.mp4)**

The video demonstrates the recommended first-run workflow: downloading the pre-trained model package and sample test videos, launching `sicklesight_gui.py`, adding a test video, selecting the combined pipeline, choosing the appropriate tracking backend, running **Check**, and starting a short test run.

<video src="Sicklesight_runtest.mp4" controls width="100%">
  Your browser does not support embedded videos. Open Sicklesight_runtest.mp4 instead.
</video>

---

## Usage

### Option A — Graphical User Interface (GUI)

Launch the GUI with:

```bash
conda activate sicklesight
python sicklesight_gui.py
```

**Workflow inside the GUI:**

1. Click **Add Folder** to recursively scan for `.mp4` videos, or **Add Video** to add individual files.
2. Select specific videos or entire folders from the tree.
3. Choose a **Pipeline script**. The GUI defaults to the folder containing `sicklesight_gui.py` and selects `sicklesight_merged.py` when available.
4. Choose **Segmentation / Tracking**:
   - `Cellpose` for standard-resolution videos
   - `Low-resolution YOLO/BoT-SORT` for low-resolution videos
5. Set **Max seconds**, **Frames/sec**, **Full video**, **Target frames**, and **YOLO conf** as needed. Leave `Frames/sec` blank and `YOLO conf` as `auto` unless you want to override them.
6. Confirm the **Output Folder**. The default is `output_default/` beside the GUI script.
7. Click **Check** to run preflight checks, then click **Run** when the setup is ready.

The GUI also includes an inline video preview, a save-current-frame control, and a results preview panel for key output figures.

---

### Option B — Command Line

All three analysis scripts can be used directly from the terminal.

Useful time options:

SickleSight converts `--max_time` to a frame number using the video's own FPS by default. For example, a 20 fps video with `--max_time 120` processes up to about frame `2400`.

To force a custom time definition, set `--analysis_fps`. For example, `--analysis_fps 4` means 4 raw frames are treated as 1 second. Annotated output videos and time-based plots use the same FPS setting.

| Option | Meaning |
|---|---|
| `--max_time 120` | Process the first 120 seconds |
| `--max_time 480` | Process the first 480 seconds |
| `--full_video` | Process the complete video |

Example with a custom time definition:

```bash
python sicklesight_merged.py \
    -i video1.mp4 \
    --max_time 120 \
    --analysis_fps 2
```

This processes up to about frame `240` because `120 × 2 = 240`.

All three pipelines support the same segmentation/tracking choices:

| Video type | Option to use |
|---|---|
| High-resolution / standard-resolution | No extra option, or `--tracking_backend cellpose` |
| Low-resolution | `--tracking_backend low_res` |

---

#### Pipeline 3: Combined Analysis (`sicklesight_merged.py`)

Runs both Pipeline 1 (see below) and Pipeline 2 (see below) in a single pass — more efficient than running them sequentially.

Basic command:

```bash
python sicklesight_merged.py \
    -i video1.mp4,video2.mp4
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | No | `output_default/` | Output directory |
| `--frame_skip` | No | `2` | Process every N-th frame |
| `--max_time` | No | `120` | Maximum seconds to process per video; shorter videos run fully |
| `--analysis_fps` | No | Auto video FPS | Optional custom FPS used for `--max_time`, plots, and output video playback |
| `--full_video` | No | Off | Process the complete video |
| `--max_frame` | No | — | Frame-based limit, used only when `--max_time` is not set |
| `--target_frames` | No | `0` and final processed frame | Comma-separated frame indices for morphology violin plots |
| `--tracking_backend` | No | `cellpose` | Use `cellpose` or `low_res` |
| `--low_res_det_conf` | No | `auto` | YOLO detection confidence for low-resolution mode; accepts `auto` or a number |

If optional arguments are not provided, command-line Pipeline 3 uses the same defaults as the GUI: Cellpose backend, 120 seconds, target frames at 0 and the final processed frame, and `output_default/`.

High-resolution, first 120 seconds:

```bash
python sicklesight_merged.py \
    -i high_resolution_video.mp4 \
    --max_time 120
```

High-resolution, first 480 seconds:

```bash
python sicklesight_merged.py \
    -i high_resolution_video.mp4 \
    --max_time 480
```

High-resolution, complete video:

```bash
python sicklesight_merged.py \
    -i high_resolution_video.mp4 \
    --full_video
```

Low-resolution, first 120 seconds:

```bash
python sicklesight_merged.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 120
```

Low-resolution, first 480 seconds:

```bash
python sicklesight_merged.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 480
```

Low-resolution, complete video:

```bash
python sicklesight_merged.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --full_video
```

**Output files:** all files from Pipeline 1 and Pipeline 2 combined. Each per-video output folder also includes `frame_0_segmentation.png`, a frame-0 image with blue segmentation/detection bounding boxes.

---

#### Pipeline 1: Temporal State-Ratio Analysis (`sicklesight_part1.py`)

Tracks the sickled/non-sickled state of each cell across all frames and produces state-ratio time curves and an annotated output video.

Basic command:

```bash
python sicklesight_part1.py \
    -i video1.mp4,video2.mp4
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | No | `output_default/` | Output directory |
| `--frame_skip` | No | `2` | Process every N-th frame (higher = faster, lower temporal resolution) |
| `--max_time` | No | `120` | Maximum seconds to process per video; shorter videos run fully |
| `--analysis_fps` | No | Auto video FPS | Optional custom FPS used for `--max_time`, plots, and output video playback |
| `--full_video` | No | Off | Process the complete video |
| `--max_frame` | No | — | Frame-based limit, used only when `--max_time` is not set |
| `--tracking_backend` | No | `cellpose` | Use `cellpose` or `low_res` |
| `--low_res_det_conf` | No | `auto` | YOLO detection confidence for low-resolution mode; accepts `auto` or a number |

High-resolution, first 120 seconds:

```bash
python sicklesight_part1.py \
    -i video1.mp4 \
    --max_time 120
```

High-resolution, first 480 seconds:

```bash
python sicklesight_part1.py \
    -i video1.mp4 \
    --max_time 480
```

High-resolution, complete video:

```bash
python sicklesight_part1.py \
    -i video1.mp4 \
    --full_video
```

Low-resolution, first 120 seconds:

```bash
python sicklesight_part1.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 120
```

Low-resolution, first 480 seconds:

```bash
python sicklesight_part1.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 480
```

Low-resolution, complete video:

```bash
python sicklesight_part1.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --full_video
```

**Output files** (written to `<output_dir>/<video_name>/`):

| File | Description |
|---|---|
| `first_frame.png` | Raw frame-0 image used as the analysis baseline |
| `frame_0_segmentation.png` | Frame-0 image with blue segmentation/detection bounding boxes |
| `<video_name>.avi` | Annotated video with cell labels and bounding boxes |
| `state_ratio_report.csv` | 7-class state distribution over time |
| `state_ratio_plot.png` | 7-class ratio curves |
| `state_ratio_plot_binary.png` | Overall sickling fraction over time |
| `state_ratio_report_pock.csv` | Pocked status over time |
| `state_ratio_plot_pocked.png` | Pocked ratio curves |
| `state_ratio_report_14groups.csv` | Combined class × pocked status (14 groups) |
| `state_ratio_plot_14groups.png` | 14-group ratio curves |
| `frame0_class_pie.png` | Pie chart of initial 7-class distribution at frame 0 |

---

#### Pipeline 2: Multi-Frame Morphology Analysis (`sicklesight_part2.py`)

Measures aspect ratio, eccentricity, and circularity at specified frames and generates violin plots comparing sickled vs. non-sickled populations.

Basic command:

```bash
python sicklesight_part2.py \
    -i video1.mp4,video2.mp4
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | No | `output_default/` | Output directory |
| `--max_time` | No | `120` | Analyze frame 0 and the frame at this many seconds |
| `--analysis_fps` | No | Auto video FPS | Optional custom FPS used for `--max_time` and plot time axes |
| `--full_video` | No | Off | Analyze frame 0 and the final frame |
| `--target_frames` | No | `0` and the frame at 120 seconds | Comma-separated frame indices for custom morphology analysis |
| `--tracking_backend` | No | `cellpose` | Use `cellpose` or `low_res` |
| `--low_res_det_conf` | No | `auto` | YOLO detection confidence for low-resolution mode; accepts `auto` or a number |

High-resolution, first 120 seconds:

```bash
python sicklesight_part2.py \
    -i video1.mp4 \
    --max_time 120
```

High-resolution, first 480 seconds:

```bash
python sicklesight_part2.py \
    -i video1.mp4 \
    --max_time 480
```

High-resolution, complete video:

```bash
python sicklesight_part2.py \
    -i video1.mp4 \
    --full_video
```

Low-resolution, first 120 seconds:

```bash
python sicklesight_part2.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 120
```

Low-resolution, first 480 seconds:

```bash
python sicklesight_part2.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --max_time 480
```

Low-resolution, complete video:

```bash
python sicklesight_part2.py \
    -i low_resolution_video.mp4 \
    --tracking_backend low_res \
    --full_video
```

Custom frame indices:

```bash
python sicklesight_part2.py \
    -i video1.mp4 \
    --target_frames 0,480
```

**Output files** (written to `<output_dir>/<video_name>/`):

| File | Description |
|---|---|
| `frame_0.png` | Raw frame-0 image used for morphology analysis |
| `frame_0_segmentation.png` | Frame-0 image with blue segmentation/detection bounding boxes |
| `frame<N>_raw_data.csv` | Per-cell morphology data at frame N |
| `frame<N>_stats_ar.csv` | Aspect ratio summary statistics at frame N |
| `frame<N>_stats_ecc.csv` | Eccentricity summary statistics at frame N |
| `frame<N>_stats_circ.csv` | Circularity summary statistics at frame N |
| `frame<N>_violin_overall_ar.png` | Violin plot: AR across all classes |
| `frame<N>_violin_7class_ar.png` | Violin plot: AR broken down by 7 classes |
| `frame<N>_violin_overall_ecc.png` | Violin plot: eccentricity across all classes |
| `frame<N>_violin_7class_ecc.png` | Violin plot: eccentricity by 7 classes |
| `frame<N>_violin_overall_circ.png` | Violin plot: circularity across all classes |
| `multiframe_comparison_ar.png` | Cross-frame AR comparison |
| `multiframe_trend.png` | Morphology trends across all target frames |

---


## Methods Overview

### Segmentation / Tracking
The default backend uses **Cellpose 3** (`cyto3_train0327`) for segmentation and SickleSight's original matching logic for tracking.

For low-resolution videos, use `--tracking_backend low_res`. This backend uses YOLO/BoT-SORT for detection and tracking, then keeps the original SickleSight classification, Siamese state detection, statistics, and output format. YOLO-seg is used only to estimate morphology from each cell crop; if a mask is unavailable, bbox-based morphology is used as a fallback.

### 7-Class Morphological Classification
A fine-tuned **Vision Transformer (ViT-Base, patch 16×16)** classifies each cell at frame 0 into one of 7 morphological classes (A–G), reflecting shape severity from normal biconcave disc (A) to fully sickled forms (G).

### Temporal State Tracking
A **Siamese ViT network** compares each cell's appearance at frame 0 (reference) to the current frame to detect state transitions (non-sickled → sickled). With the default Cellpose backend, cell identity is maintained across frames using a combination of:

- **Optical flow** (Lucas–Kanade) for bounding-box prediction
- **Intersection-over-Union (IoU)** matching
- **Size and position consistency** checks

Predictions are smoothed with an **Exponential Moving Average (EMA)** filter and confirmed by a minimum-persistence streak counter to reduce false transitions.

With `--tracking_backend low_res`, cell identity is provided by YOLO/BoT-SORT instead; the same Siamese state detection and downstream reporting are then applied.

### Morphological Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Aspect Ratio (AR) | major_axis_length / minor_axis_length | Higher → more elongated / sickled |
| Eccentricity (ECC) | skimage `regionprops` | 0 = perfect circle, 1 = line |
| Circularity | 4π × Area / Perimeter² | Lower → more irregular / sickled |

### Statistical Tests
Mann–Whitney U tests compare morphological distributions between sickled and non-sickled populations at each target frame.

---

## Hardware Requirements

| Hardware | Support |
|---|---|
| NVIDIA GPU (CUDA) | Full acceleration — recommended |
| Apple Silicon (MPS) | Supported via PyTorch MPS backend |
| CPU only | Supported, but significantly slower |

The scripts auto-detect the available device at startup.

---

## Citation

If you use this toolkit in your research, please cite:

> *[Citation to be added upon publication]*

---

## License

*[License to be added]*
