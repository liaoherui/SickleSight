# SickleSight

**SickleSight** is an AI-based toolkit for quantitative analysis of sickle cell disease dynamics from video microscopy data. It combines deep learning segmentation, Vision Transformer (ViT) classification, Siamese network state detection, and optical flow tracking to characterize individual red blood cells across time — producing publication-quality statistics, visualizations, and annotated videos.

> **Name note:** This project was formerly called *CellBox*. We recommend the new name **SickleSight** (alternatives: *SickleScope*, *SickleQuant*, *DrepaVision*) since *CellBox* is already taken.

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

---

## Repository Structure

```
SickleSight/
├── cellbox_gui.py        # GUI application — launch this to open the graphical interface
├── cellbox_part1.py      # Pipeline 1: temporal state-ratio analysis
├── cellbox_part2.py      # Pipeline 2: multi-frame morphology analysis (AR / ECC / circularity)
├── cellbox_merged.py     # Pipeline 3: combined — state-ratio + morphology in one pass
├── cellbox_env.yaml      # Conda environment specification
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/liaoherui/CellBox.git
cd CellBox
```

### 2. Create the Conda environment

```bash
conda env create -f cellbox_env.yaml
conda activate cellbox
```

### 3. Download pre-trained models

Download **CellBox-Models.zip** from the link below and unzip its contents into the **same directory as the scripts**:

> **[Download CellBox-Models.zip](https://www.dropbox.com/scl/fi/f4roqang8cwm6cazw9qiw/CellBox-Models.zip?rlkey=aj4wk4xekau0y4dj56850b18l&st=ccsuhdnh&dl=0)**

After extraction, your directory should contain `.pth` / `.pt` model files alongside the Python scripts.

---

## Usage

### Option A — Graphical User Interface (GUI)

Launch the GUI with:

```bash
conda activate cellbox
python cellbox_gui.py
```

**Workflow inside the GUI:**

1. Click **Add Parent Folder** — the GUI recursively scans for `.mp4` video files and displays them in a file tree.
2. Select specific videos or entire folders from the tree.
3. Choose a **Pipeline script** from the dropdown:
   - `cellbox_part1` — temporal state-ratio analysis
   - `cellbox_part2` — multi-frame morphology analysis
   - `cellbox_merged` — both analyses combined
4. Set the **Pipeline Folder** (directory containing the scripts and model files).
5. Set the **Output Directory** where results will be saved.
6. Click **Generate Script & Run Analysis** — the GUI generates a platform-specific shell script and executes it, streaming live output to the built-in terminal pane.

---

### Option B — Command Line

All three analysis scripts accept the same core arguments and can be used independently from the terminal without the GUI.

---

#### Pipeline 1: Temporal State-Ratio Analysis (`cellbox_part1.py`)

Tracks the sickled/non-sickled state of each cell across all frames and produces state-ratio time curves and an annotated output video.

```bash
python cellbox_part1.py \
    -i video1.mp4,video2.mp4 \
    -o /path/to/output \
    [--frame_skip 2] \
    [--max_frame 480]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | Yes | — | Output directory (created if it does not exist) |
| `--frame_skip` | No | `2` | Process every N-th frame (higher = faster, lower temporal resolution) |
| `--max_frame` | No | `480` | Maximum number of frames to process per video |

**Output files** (written to `<output_dir>/<video_name>/`):

| File | Description |
|---|---|
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

#### Pipeline 2: Multi-Frame Morphology Analysis (`cellbox_part2.py`)

Measures aspect ratio, eccentricity, and circularity at specified frames and generates violin plots comparing sickled vs. non-sickled populations.

```bash
python cellbox_part2.py \
    -i video1.mp4,video2.mp4 \
    -o /path/to/output \
    [--target_frames 0,480]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | Yes | — | Output directory |
| `--target_frames` | No | `0,480` | Comma-separated frame indices at which to run morphology analysis |

**Output files** (written to `<output_dir>/<video_name>/`):

| File | Description |
|---|---|
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

#### Pipeline 3: Combined Analysis (`cellbox_merged.py`)

Runs both Pipeline 1 and Pipeline 2 in a single pass — more efficient than running them sequentially.

```bash
python cellbox_merged.py \
    -i video1.mp4,video2.mp4 \
    -o /path/to/output \
    [--frame_skip 2] \
    [--max_frame 480] \
    [--target_frames 0,480]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i` / `--inputs` | Yes | — | Comma-separated list of input video file paths |
| `-o` / `--output_dir` | Yes | — | Output directory |
| `--frame_skip` | No | `2` | Process every N-th frame |
| `--max_frame` | No | `480` | Maximum number of frames to process per video |
| `--target_frames` | No | `0,480` | Comma-separated frame indices for morphology violin plots |

**Output files:** all files from Pipeline 1 and Pipeline 2 combined.

---

## Methods Overview

### Cell Segmentation
Individual cells are segmented using **Cellpose 3** (`cyto3` pre-trained model). Frames are downscaled to 20% before segmentation for speed, and masks are upscaled back to original resolution.

### 7-Class Morphological Classification
A fine-tuned **Vision Transformer (ViT-Base, patch 16×16)** classifies each cell at frame 0 into one of 7 morphological classes (A–G), reflecting shape severity from normal biconcave disc (A) to fully sickled forms (G).

### Temporal State Tracking
A **Siamese ViT network** compares each cell's appearance at frame 0 (reference) to the current frame to detect state transitions (non-sickled → sickled). Cell identity is maintained across frames using a combination of:

- **Optical flow** (Lucas–Kanade) for bounding-box prediction
- **Intersection-over-Union (IoU)** matching
- **Size and position consistency** checks

Predictions are smoothed with an **Exponential Moving Average (EMA)** filter and confirmed by a minimum-persistence streak counter to reduce false transitions.

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
