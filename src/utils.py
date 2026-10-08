"""
Utility functions for AgriShield Quantum.
Provides path resolution, JSON/CSV cache I/O, formatting, and logging helpers.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
EXPORTS_DIR = PROJECT_ROOT / "exports"
SRC_DIR = PROJECT_ROOT / "src"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that gracefully handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_cache_json(filename: str, data: Any) -> Path:
    """Save data to cache directory as JSON with numpy conversion."""
    path = CACHE_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
    return path


def load_cache_json(filename: str, default: Optional[Any] = None) -> Optional[Any]:
    """Load JSON from cache directory. Returns default if not found or corrupted."""
    path = CACHE_DIR / filename
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache {filename}: {e}")
        return default


def cache_exists(filename: str) -> bool:
    """Check if cache file exists and is non-empty."""
    path = CACHE_DIR / filename
    return path.exists() and path.stat().st_size > 0
