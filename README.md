# NC-AFM Control GUI

This project provides a lightweight GUI for controlling NC-AFM parameters through the SXM software (via DDE). The GUI is modular: each tab of the interface is in its own file, so the code is easier to read and extend.

---

## Getting Started (for first-time users)

This section is for researchers who may not have used Python before. Follow it step by step.

### 1. Install Python
- Download Python (we recommend the [Anaconda distribution](https://www.anaconda.com/download)) since it comes with most scientific packages preinstalled.
- During installation, make sure to check **“Add Anaconda to PATH”** (if asked).

### 2. Open a terminal
- On Windows: open **Anaconda Prompt** (search in Start Menu).
- On macOS/Linux: open **Terminal**.

### 3. Create a new environment
This project works best in its own “sandbox” environment so it won’t interfere with other tools.

```bash
conda create -n sxm-ncafm python=3.11
conda activate sxm-ncafm
```

### 4. Install required packages
There are two options:

**Option A: Using Conda**
```bash
conda env update -f environment.yml
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

This installs:
- PyQt5 (for the GUI)
- PyQtGraph (for plotting)
- NumPy + SciPy (for math)
- pywin32 (Windows-only, for SXM driver/DDE link)

### 5. Get the code
Download or clone this repository:
```bash
git clone https://github.com/YOUR-LAB/sxm_ncafm_control.git
cd sxm_ncafm_control
```

### 6. Run the GUI
From inside the project folder:
```bash
python -m sxm_ncafm_control.app
```

The NC-AFM Control GUI window should appear.

---

## 📦 Project layout
```
sxm_ncafm_control/
│
├── app.py               # Entry point, launches the main window
├── dde_client.py        # Handles communication with SXM (DDE bridge)
├── device_driver.py     # Low-level SXM driver interface
├── io_reader.py         # Optional reader for SXM oscilloscope channels
│
└── gui/                 # The GUI package
    ├── common.py        # Shared constants and helper functions
    ├── params_tab.py    # "Parameters" tab (tuning parameters table)
    ├── step_test_tab.py # "Step Test" tab (square wave testing)
    ├── suggested_tab.py # "Suggested Setup" tab (Ki/Kp suggestions + spectrum import)
    ├── scope_tab.py     # "Scope" tab (one-shot channel capture)
    └── main_window.py   # Combines everything into the full application
```

---
