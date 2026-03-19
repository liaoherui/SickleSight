# Web Image Labeler (`index.html`)

`index.html` is a fully browser-based image classification tool for manually labeling cell images into morphological classes. It runs entirely in your browser — no server, no installation required.

---

## Overview

The labeler is designed to classify cropped cell images into one of the seven SickleSight morphological classes (A–G), though the class definitions are fully customizable. It uses the [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API) to read from and write to local folders directly.

> **Browser requirement:** A Chromium-based browser (Google Chrome, Microsoft Edge, Brave, etc.) is strongly recommended. Firefox and Safari do not fully support the File System Access API.

---

## Getting Started

### 1. Open the tool

Double-click `index.html` to open it in your browser, or drag it into an open browser window. No web server is needed.

### 2. Select the source folder

Click **Select Source Folder** in the right-hand panel. The browser will ask for permission to access the folder. Once granted, all images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`) inside that folder will appear as thumbnails in the left panel.

### 3. Select the output folder

Click **Select Output Folder**. Classified images will be copied (or moved) into sub-folders named after each class inside this output directory.

> Tip: The output folder can be the same as the source folder, or a completely separate directory.

---

## Interface Layout

```
┌─────────────────┬──────────────────────────────┬─────────────────────┐
│  Thumbnail Grid │       Image Viewer            │  Controls           │
│  (left panel)   │       (center panel)          │  (right panel)      │
│                 │                               │                     │
│  [img1] [img2]  │   [ large image display ]     │  [Setup]            │
│  [img3] [img4]  │                               │  [Classification]   │
│  ...            │                               │  [Configuration]    │
│                 │                               │  [History]          │
└─────────────────┴──────────────────────────────┴─────────────────────┘
```

- **Left panel** — scrollable thumbnail grid; click to select, `Ctrl+Click` or `Shift+Click` for multi-select.
- **Center panel** — displays the selected image(s) at full size.
- **Right panel** — all controls for setup, classification, configuration, and history.
- **Dividers** — drag the vertical dividers to resize panels.

---

## Classifying Images

### Using keyboard shortcuts (recommended)

Each class has a keyboard shortcut. Press the key while one or more images are selected to instantly classify and advance to the next image.

| Key | Default Class |
|-----|--------------|
| `1` | A_Discocyte |
| `2` | B_Cup-shape |
| `3` | C_Stomatocyte |
| `4` | D_Reticulocyte |
| `5` | E_Echinocyte |
| `6` | F_Granular |
| `7` | G_ISC |
| `s` | skip |

### Using buttons

In the **Classification** section of the right panel, click any class button to classify the currently selected image(s).

### Multi-image classification

Select multiple images using `Ctrl+Click` (individual) or `Shift+Click` (range), then press a shortcut key or click a button to classify all selected images at once.

---

## File Handling

When an image is classified:
- A sub-folder named after the class (e.g., `A_Discocyte/`) is created inside the output folder.
- The image is **copied** into that sub-folder by default.
- If **Move files** is toggled on, the image is also deleted from the source folder after copying.

---

## Undo / Redo

- Press `Ctrl+Z` to undo the last classification action.
- Press `Ctrl+Y` (or `Ctrl+Shift+Z`) to redo.
- The **History** section in the right panel shows a log of all actions.

---

## Customizing Classes

In the **Configuration** section of the right panel, you can:

1. Edit existing class names and keyboard shortcuts.
2. Add new classes.
3. Remove classes you don't need.
4. Assign colors to each class for visual distinction.

Changes take effect immediately. The tool remembers your configuration for the current session.

---

## Progress Tracking

A progress bar at the top of the interface shows how many images have been classified out of the total. Toast notifications briefly confirm each action (copy, move, skip, undo).

---

## Typical Workflow

```
1. Open index.html in Chrome/Edge
2. Click "Select Source Folder"  →  grant folder access
3. Click "Select Output Folder"  →  grant folder access
4. Click the first thumbnail in the left panel
5. Press 1–7 or 's' to classify; the tool auto-advances
6. Repeat until all images are done
7. Find labelled images in output_folder/ClassName/ sub-directories
```

---

## Limitations

- Requires a Chromium-based browser (Chrome 86+, Edge 86+).
- Works only with local files — no network upload occurs.
- Session state is not persisted across page reloads; re-select source/output folders after refreshing.
- Very large folders (thousands of images) may load slowly due to browser memory constraints.
