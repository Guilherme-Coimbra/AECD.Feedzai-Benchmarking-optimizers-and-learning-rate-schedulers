from pathlib import Path

# Resolve project root as the directory containing this file's parent (i.e., repo root with src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "bank-account-fraud-dataset-neurips-2022"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"  
TABLES_DIR = RESULTS_DIR / "tables"     

def ensure_dirs():
    for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
