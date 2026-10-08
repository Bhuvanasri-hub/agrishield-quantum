"""
AgriShield Quantum - Spatial Risk & Pathogen Diffusion Modeling (Phase 3)
Calculates per-cell stress probability using calibrated quantum model,
then models secondary neighbor transmission influenced by distance and microclimate humidity.

Formula:
  final_risk(i, j) = alpha * own_risk(i, j) + beta * sum_{(r,c) in N(i,j)} w(d, H) * own_risk(r,c)
  where w(d, H) = (1 / d^1.5) * (humidity / 100)^1.8
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

from src.data_synth import load_or_generate_synthetic_data
from src.features import extract_agronomic_features, AgronomicFeaturePipeline
from src.quantum_models import build_qsvc_model, predict_qsvc_proba
from src.utils import save_cache_json, load_cache_json


def classify_risk_tier(score: float) -> str:
    """Classify risk score into tier: Low, Medium, High."""
    if score >= 0.65:
        return "High"
    elif score >= 0.35:
        return "Medium"
    return "Low"


def compute_neighbor_spread(
    own_risk_grid: np.ndarray,
    humidity_grid: np.ndarray,
    alpha: float = 0.55,
    beta: float = 0.45
) -> np.ndarray:
    """
    Simulate airborne spore and fungal diffusion from neighboring farm plots.
    Weights decay with Euclidean distance and amplify super-linearly with humidity.
    """
    rows, cols = own_risk_grid.shape
    final_risk = np.zeros_like(own_risk_grid)
    
    for r in range(rows):
        for c in range(cols):
            spillover = 0.0
            weight_sum = 0.0
            h_val = float(humidity_grid[r, c])
            # Humidity enhancement factor: moist canopies facilitate spore germination
            h_factor = (h_val / 100.0) ** 1.8
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        dist = np.sqrt(dr**2 + dc**2)
                        w = (1.0 / (dist ** 1.5)) * h_factor
                        spillover += w * own_risk_grid[nr, nc]
                        weight_sum += w
                        
            norm_spill = spillover / max(weight_sum, 1e-4) if weight_sum > 0 else 0.0
            combined = alpha * own_risk_grid[r, c] + beta * norm_spill
            final_risk[r, c] = np.clip(combined, 0.0, 1.0)
            
    return final_risk


def precompute_all_risk_maps() -> Dict[str, Any]:
    """
    Precomputes spatial risk maps across all 24 weeks using quantum model predictions
    and neighbor contagion mechanics. Saves to cache/risk_maps.json.
    """
    print("--> Precomputing Spatial Risk Maps for all 24 weeks...")
    df = load_or_generate_synthetic_data()
    grid_size = 10
    n_weeks = df["week_idx"].nunique()
    
    # Train QSVC model for risk probability calibration (using 2-qubit feature mapping for strong fidelity)
    pipeline = AgronomicFeaturePipeline(n_components=2, scale_for_quantum=True)
    X_raw_all = extract_agronomic_features(df)
    X_q_all = pipeline.fit_transform(X_raw_all)
    y_all = df["stress_label"].to_numpy()
    
    # Stratified training subset
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    np.random.seed(42)
    sample_idx = np.concatenate([
        np.random.choice(pos_idx, size=min(50, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, size=min(50, len(neg_idx)), replace=False)
    ])
    
    qsvc = build_qsvc_model(n_qubits=2, reps=1)
    qsvc.fit(X_q_all[sample_idx], y_all[sample_idx])
    
    # Predict probabilities across full dataset
    probs = predict_qsvc_proba(qsvc, X_q_all)[:, 1]
    df["quantum_stress_prob"] = probs
    
    weekly_maps = {}
    
    for w in range(n_weeks):
        week_df = df[df["week_idx"] == w].sort_values(by=["row", "col"]).copy()
        date_str = week_df["date"].iloc[0]
        
        own_grid = week_df["quantum_stress_prob"].to_numpy().reshape((grid_size, grid_size))
        hum_grid = week_df["humidity"].to_numpy().reshape((grid_size, grid_size))
        rain_grid = week_df["rainfall"].to_numpy().reshape((grid_size, grid_size))
        temp_grid = week_df["temperature"].to_numpy().reshape((grid_size, grid_size))
        ndvi_grid = week_df["ndvi"].to_numpy().reshape((grid_size, grid_size))
        ndre_grid = week_df["ndre"].to_numpy().reshape((grid_size, grid_size))
        label_grid = week_df["stress_label"].to_numpy().reshape((grid_size, grid_size))
        
        final_grid = compute_neighbor_spread(own_grid, hum_grid)
        
        # Structure cells for frontend rendering
        cells = []
        for r in range(grid_size):
            for c in range(grid_size):
                cell_id = f"cell_{r:02d}_{c:02d}"
                risk_score = float(final_grid[r, c])
                tier = classify_risk_tier(risk_score)
                cells.append({
                    "cell_id": cell_id,
                    "row": r,
                    "col": c,
                    "quantum_prob": round(float(own_grid[r, c]), 4),
                    "final_risk": round(risk_score, 4),
                    "risk_tier": tier,
                    "ndvi": round(float(ndvi_grid[r, c]), 4),
                    "ndre": round(float(ndre_grid[r, c]), 4),
                    "humidity": round(float(hum_grid[r, c]), 1),
                    "temperature": round(float(temp_grid[r, c]), 1),
                    "rainfall": round(float(rain_grid[r, c]), 1),
                    "stress_label": int(label_grid[r, c])
                })
                
        weekly_maps[str(w)] = {
            "week_idx": w,
            "date": date_str,
            "mean_risk": round(float(np.mean(final_grid)), 4),
            "high_risk_count": int(np.sum(final_grid >= 0.65)),
            "medium_risk_count": int(np.sum((final_grid >= 0.35) & (final_grid < 0.65))),
            "low_risk_count": int(np.sum(final_grid < 0.35)),
            "cells": cells,
            "risk_matrix": final_grid.tolist()
        }
        
    save_cache_json("risk_maps.json", weekly_maps)
    print("--> Successfully generated and saved cache/risk_maps.json")
    return weekly_maps


if __name__ == "__main__":
    precompute_all_risk_maps()
