"""
AgriShield Quantum - Synthetic Data Generator (Phase 1)
Generates 10x10 farm grid time-series data (~24 weeks) with realistic spatio-temporal
crop disease spread dynamics, pre-symptomatic Red-Edge (NDRE) drop, and sensor noise.

NOTE: Clearly labeled as "Synthetic data for prototype validation".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
from src.utils import CACHE_DIR, save_cache_json


SYNTHETIC_DATA_CSV = CACHE_DIR / "synthetic_farm_data.csv"
DATA_SOURCE_METADATA = CACHE_DIR / "data_source_metadata.json"


def generate_synthetic_data(
    grid_size: int = 10,
    n_weeks: int = 24,
    start_date: str = "2026-05-15",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time series for a grid_size x grid_size farm over n_weeks.
    
    Physics & Biological Dynamics:
    1. Soil/field heterogeneity: Baseline natural vigor varies slightly per cell.
    2. Weather variation: Weekly cycles with monsoon showers, high humidity, warm temps.
    3. Disease clusters: 2 initial outbreak epicenters seeded around week 5-6.
    4. Spatio-temporal contagion: Spread to 8-neighbors driven by humidity (>75%) & temp (28-34C).
    5. Spectral precursor: Cellular chlorophyll loss causes NDRE to drop 1-2 weeks BEFORE
       visible chlorosis/necrosis triggers the visible stress_label.
    6. Non-trivial separability: Sensor noise and natural variance create realistic class overlap.
    """
    np.random.seed(seed)
    
    # Grid coordinates
    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    n_cells = grid_size * grid_size
    
    # Spatial baseline vigor (healthy baseline variation across field, 0.75 - 0.88)
    base_ndvi = 0.78 + 0.05 * np.sin(np.outer(rows, np.ones(grid_size)) * 0.5) + \
                0.04 * np.cos(np.outer(np.ones(grid_size), cols) * 0.5)
    base_ndre = base_ndvi * 0.76  # NDRE is typically slightly lower than NDVI in healthy canopy
    
    # Latent infection state per cell: 0.0 (pristine) to 1.0 (severe disease)
    latent_infection = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Epicenters (e.g., lower-lying humid spots in field)
    epicenters = [(2, 3), (7, 6)]
    outbreak_week = 5
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    
    # Weather base trends across Krishna district Kharif season (May to October)
    for t in range(n_weeks):
        current_date = (start_dt + timedelta(weeks=t)).strftime("%Y-%m-%d")
        
        # Seasonal weather profile
        # Mid-season monsoon (weeks 7 to 16): higher humidity, heavier rainfall
        monsoon_factor = np.exp(-((t - 11) ** 2) / 25.0)
        temp_base = 32.5 - 2.5 * monsoon_factor + np.random.normal(0, 0.8)
        humidity_base = 65.0 + 22.0 * monsoon_factor + np.random.normal(0, 2.5)
        rainfall_base = max(0.0, 10.0 + 45.0 * monsoon_factor + np.random.normal(0, 12.0))
        
        # Seed initial infections at outbreak_week
        if t == outbreak_week:
            for (r, c) in epicenters:
                latent_infection[r, c] = 0.35
        elif t > outbreak_week:
            # Latent infection growth and spatial diffusion
            new_infection = latent_infection.copy()
            for r in range(grid_size):
                for c in range(grid_size):
                    # Local environmental pathogen favorability
                    weather_boost = 1.0 + (humidity_base - 70.0) / 100.0 + (temp_base - 28.0) / 50.0
                    weather_boost = max(0.6, min(1.6, weather_boost))
                    
                    if latent_infection[r, c] > 0.05:
                        # Existing infection progression (logistic saturation)
                        growth = 0.28 * latent_infection[r, c] * (1.0 - latent_infection[r, c]) * weather_boost
                        new_infection[r, c] += growth
                    
                    # Spread from 8-neighbors
                    r_min, r_max = max(0, r - 1), min(grid_size, r + 2)
                    c_min, c_max = max(0, c - 1), min(grid_size, c + 2)
                    neighbor_slice = latent_infection[r_min:r_max, c_min:c_max]
                    
                    # Mean infection in immediate neighborhood (excluding self)
                    neighbor_pressure = (np.sum(neighbor_slice) - latent_infection[r, c]) / 8.0
                    
                    if neighbor_pressure > 0.03:
                        spillover = 0.32 * neighbor_pressure * weather_boost
                        new_infection[r, c] += spillover
            
            latent_infection = np.clip(new_infection, 0.0, 1.0)
            
        # Record observations for each cell in the grid
        for r in range(grid_size):
            for c in range(grid_size):
                cell_id = f"cell_{r:02d}_{c:02d}"
                inf = latent_infection[r, c]
                
                # Spatial microclimate perturbations
                cell_temp = float(temp_base + np.random.normal(0, 0.4))
                cell_humidity = float(np.clip(humidity_base + np.random.normal(0, 1.8), 40.0, 98.0))
                cell_rainfall = float(max(0.0, rainfall_base + np.random.normal(0, 3.0)))
                
                # Spectral response to latent disease:
                # 1. NDRE drops EARLY (chlorophyll degradation in mesophyll cells)
                # Sensitive drop kicks in even at low latent levels (inf > 0.12)
                ndre_drop = 0.32 * (inf ** 1.1)
                cell_ndre = base_ndre[r, c] - ndre_drop + np.random.normal(0, 0.025)
                cell_ndre = float(np.clip(cell_ndre, 0.15, 0.75))
                
                # 2. NDVI drops LATER (overall biomass and canopy thinning)
                # Noticeable drop occurs primarily once latent infection > 0.30
                ndvi_drop = 0.36 * (max(0.0, inf - 0.15) ** 1.3)
                cell_ndvi = base_ndvi[r, c] - ndvi_drop + np.random.normal(0, 0.025)
                cell_ndvi = float(np.clip(cell_ndvi, 0.20, 0.90))
                
                # 3. Visible ground-truth stress label (field scout visual detection)
                # Visible symptoms appear when infection is substantial (inf >= 0.45)
                # Add slight scout detection noise (+/- 0.05 border ambiguity)
                effective_thresh = 0.45 + np.random.normal(0, 0.04)
                stress_label = int(inf >= effective_thresh)
                
                records.append({
                    "cell_id": cell_id,
                    "row": int(r),
                    "col": int(c),
                    "week_idx": int(t),
                    "date": current_date,
                    "ndvi": round(cell_ndvi, 4),
                    "ndre": round(cell_ndre, 4),
                    "temperature": round(cell_temp, 2),
                    "humidity": round(cell_humidity, 2),
                    "rainfall": round(cell_rainfall, 2),
                    "latent_infection": round(float(inf), 4),
                    "stress_label": stress_label,
                    "is_synthetic": True,
                })
                
    df = pd.DataFrame(records)
    return df


def load_or_generate_synthetic_data(force_recompute: bool = False) -> pd.DataFrame:
    """Load cached synthetic dataset or generate and cache it."""
    if not force_recompute and SYNTHETIC_DATA_CSV.exists():
        try:
            df = pd.read_csv(SYNTHETIC_DATA_CSV)
            return df
        except Exception as e:
            print(f"Warning: Corrupted synthetic cache, regenerating: {e}")
            
    df = generate_synthetic_data()
    df.to_csv(SYNTHETIC_DATA_CSV, index=False)
    
    metadata = {
        "dataset_name": "AgriShield Synthetic Farm Grid (10x10)",
        "source": "Spatio-Temporal Crop Disease Diffusion Simulation",
        "label_type": "Synthetic data for prototype validation",
        "description": "Simulates 100 farm plots over 24 weeks with Red-Edge pre-symptomatic stress and neighbor diffusion.",
        "num_rows": len(df),
        "num_weeks": df["week_idx"].nunique(),
        "positive_rate": float(df["stress_label"].mean()),
        "generated_at": datetime.now().isoformat(),
        "fixed_seed": 42
    }
    save_cache_json("data_source_metadata.json", metadata)
    return df


if __name__ == "__main__":
    df = load_or_generate_synthetic_data(force_recompute=True)
    print(f"Synthetic data generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Stress label distribution:\n{df['stress_label'].value_counts(normalize=True)}")
