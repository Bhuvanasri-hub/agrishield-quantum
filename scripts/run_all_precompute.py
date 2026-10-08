"""
AgriShield Quantum - Master Precompute Pipeline (scripts/run_all_precompute.py)
Executes all data generation, quantum modeling, benchmark cross-validation,
spatial contagion, QAOA scouting, and hardware evaluation jobs.
Saves all outputs to /cache (JSON/CSV) so that the Streamlit web application
starts in under 5 seconds without heavy computational overhead.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Fix stdout encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_synth import load_or_generate_synthetic_data
from src.data_real import generate_real_satellite_grid
from src.benchmark import run_benchmarks
from src.risk import precompute_all_risk_maps
from src.qaoa_scout import run_scouting_experiments
from src.early_warning import evaluate_early_warning_horizons
from src.ibm_hardware import run_hardware_or_fallback
from scripts.export_figures import run_all_exports


def main():
    t_start = time.time()
    print("==============================================================")
    print("      AgriShield Quantum: Master Precompute Pipeline          ")
    print("  Problem UC-034: Pest & Crop-Disease Early Warning (Qiskit)   ")
    print("==============================================================\n")
    
    print("[1/8] Generating Synthetic 10x10 Farm Grid Data...")
    df_synth = load_or_generate_synthetic_data(force_recompute=True)
    print(f"      Done. Shape: {df_synth.shape}\n")
    
    print("[2/8] Generating Real Satellite (NASA + Sentinel-2) Data Layer...")
    df_real = generate_real_satellite_grid(force_download=False)
    print(f"      Done. Shape: {df_real.shape}\n")
    
    print("[3/8] Running Rigorous 5-Fold Model Benchmark & Qubit Ablation...")
    run_benchmarks(n_samples=120, n_splits=5)
    print("      Done.\n")
    
    print("[4/8] Precomputing Spatial Contagion Risk Maps (24 Weeks)...")
    precompute_all_risk_maps()
    print("      Done.\n")
    
    print("[5/8] Running QAOA Field Scouting Optimization Experiments...")
    run_scouting_experiments(week_idx=15)
    print("      Done.\n")
    
    print("[6/8] Evaluating Early Warning Lead Times & Multi-Spectral Timeline...")
    evaluate_early_warning_horizons()
    print("      Done.\n")
    
    print("[7/8] Evaluating IBM Quantum Hardware & Realistic Noise Comparison...")
    run_hardware_or_fallback()
    print("      Done.\n")
    
    print("[8/8] Exporting High-Resolution Presentation Graphics to /exports...")
    run_all_exports()
    print("      Done.\n")
    
    elapsed = time.time() - t_start
    print("==============================================================")
    print(f"  All Precomputations Completed Successfully in {elapsed:.1f}s!  ")
    print("  The Streamlit app is ready for zero-latency instant startup. ")
    print("==============================================================")


if __name__ == "__main__":
    main()
