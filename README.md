# AECD & Feedzai - Benchmarking optimizers and learning rate schedulers 
This repository was developed as part of the AECD – Applications of Data Science and Engineering Master’s course, in collaboration with Feedzai. The project focuses on benchmarking different optimizers and learning rate scheduling strategies in training deep neural networks.

---

## Repository Structure

```
.
├── data/                       # Raw data (read-only) and processed data
│   ├── bank-account-fraud-dataset-neurips-2022/
│   │   ├── Base.csv
│   │   ├── ...
│   └── processed/              # Save cleaned/encoded splits here (never overwrite raw)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── ...
│
├── src/                        # Reusable Python modules
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── utils.py
│
├── experiments/                # Configs, logs, and outputs per experiment
│   └── exp_template/
│
├── results/                    # Figures, tables, metrics exported by notebooks / scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Rule of thumb:** raw files in `data/...` are **read-only**. Write all derived artifacts to `data/processed/`, `results/`, or `experiments/`.

---

## Prerequisites

* **Python 3.10+** (recommended)
* **Git**
* (Optional) **Make** and **pipx** for convenience

---

## Quick Start

### 1) Clone the repo

```bash
git clone <YOUR-REPO-URL>.git
cd <YOUR-REPO-NAME>
```

### 2) Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
```

### 3) Activate / Deactivate

**macOS / Linux**

```bash
# activate
source .venv/bin/activate
# deactivate
deactivate
```

**Windows (PowerShell)**

```powershell
# activate
./.venv/Scripts/Activate.ps1
# deactivate
deactivate
```

### 4) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If `requirements.txt` is empty at first use, install the basics:

```bash
pip install numpy pandas scikit-learn matplotlib jupyter ipykernel seaborn
python -m ipykernel install --user --name aecd-env --display-name "AECD (venv)"
```

Then **freeze** them:

```bash
pip freeze --exclude-editable > requirements.txt
```

### 5) Fetch datasets

Follow the dataset download and extraction instructions in [data/README.md](data/README.md).

---

## Managing Dependencies

### Add a new package

```bash
pip install <package-name>
pip freeze --exclude-editable > requirements.txt
```

### Rebuild from scratch

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

---

## Git Hygiene

* Commit small, logical changes with clear messages.
* Keep notebooks tidy (strip large outputs if not needed).
* Large artifacts (models, CSVs, images) → store under `results/` or `experiments/` and consider Git LFS/DVC if they must be versioned. Prefer not to commit multi-GB files.

---

## Reproducibility Tips

* Set random seeds in notebooks and `src/utils.py`.
* Log package versions (`pip freeze`) with each experiment.
* Save a copy of train/val/test indices used for each run to `experiments/<exp-id>/splits/`.

---

## Contributing / Collaboration

* Keep reusable code in `src/` and import it from notebooks.
* Document functions with docstrings and type hints.
* Use notebook markdown cells to describe goals, decisions, and findings.
* Add a short `README.md` inside each new folder in `experiments/` describing the setup and results.

---
