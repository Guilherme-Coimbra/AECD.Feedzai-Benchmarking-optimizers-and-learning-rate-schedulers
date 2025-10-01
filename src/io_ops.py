from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd


def read_csv_safely(
    path: Path,
    dtype_overrides: Optional[Dict[str, str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Safe CSV reader with sensible defaults for large tabular datasets.
    - low_memory=False to avoid mixed dtypes
    - keep_default_na=True (let pandas infer NaNs)
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    defaults = dict(low_memory=False)
    defaults.update(kwargs or {})
    df = pd.read_csv(path, **defaults)
    if dtype_overrides:
        for col, dt in dtype_overrides.items():
            if col in df.columns:
                df[col] = df[col].astype(dt)
    return df

def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

    """
    Estimate in-memory size (in MB) by reading up to `sample_rows`,
    computing per-row memory usage, then scaling to total_rows.

    Returns:
        (mem_est_mb, sample_n)
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # If total_rows not provided, do a quick count (chunked)
    if total_rows is None:
        total_rows = count_rows_csv(path)

    # Read a sample (first N rows); safe defaults
    n = min(sample_rows, total_rows) if total_rows > 0 else sample_rows
    if n == 0:
        return (0.0, 0)

    df_sample = pd.read_csv(path, nrows=n, low_memory=False)
    per_row_bytes = float(df_sample.memory_usage(deep=True).sum()) / max(len(df_sample), 1)
    mem_est_mb = (per_row_bytes * total_rows) / (1024 ** 2)
    return (mem_est_mb, len(df_sample))