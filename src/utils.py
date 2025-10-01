import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

def set_global_seed(seed: int = 42):
    """Set seeds for Python, NumPy (extend later for torch if needed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_logger(name: str = "aecd") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

@contextmanager
def timeit(msg: str, logger: Optional[logging.Logger] = None):
    """Context manager to log execution time of a code block."""
    _log = logger or get_logger()
    start = time.time()
    _log.info(f"START: {msg}")
    try:
        yield
    finally:
        dt = time.time() - start
        _log.info(f"END:   {msg} (took {dt:.2f}s)")
