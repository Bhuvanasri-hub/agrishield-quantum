"""
AgriShield Quantum - Figure & Slide Asset Exporter (Phase 9)
Generates high-resolution PNG visualizations for presentation decks, documentation,
and conference posters:
1. quantum_circuits.png (ZZFeatureMap & RealAmplitudes circuits)
2. benchmark_comparison.png (Accuracy & F1 error-bar comparison)
3. risk_heatmap_week15.png (10x10 spatial infection heatmap)
4. early_warning_timeline.png (Focal cell multi-spectral lead time progression)
5. ibm_hardware_comparison.png (Ideal vs Noisy vs Hardware kernel difference matrices)
6. system_architecture.png (AgriShield 4-step pipeline schematic)
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from pathlib import Path

from src.utils import EXPORTS_DIR, load_cache_json
from src.quantum_models import build_zz_feature_map, build_real_amplitudes_ansatz


def export_circuits():
    """Export quantum circuit diagram."""
    fmap = build_zz_feature_map(4, reps=1)
    ansatz = build_real_amplitudes_ansatz(4, reps=1)
    
    # Feature Map
    fig = fmap.draw(output="mpl")
    fig.suptitle("AgriShield ZZFeatureMap (Linear Entanglement, 4 Qubits)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(EXPORTS_DIR / "quantum_feature_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    # Combined VQC
    vqc = fmap.compose(ansatz)
    fig_vqc = vqc.draw(output="mpl")
    fig_vqc.suptitle("AgriShield VQC Architecture (ZZFeatureMap + RealAmplitudes)", fontsize=13, weight="bold")
    fig_vqc.tight_layout()
    fig_vqc.savefig(EXPORTS_DIR / "quantum_circuits.png", dpi=200, bbox_inches="tight")
    plt.close(fig_vqc)
    print("--> Exported quantum circuits to exports/quantum_circuits.png")


def export_benchmark_chart():
    """Export benchmark comparison chart with error bars."""
    bench = load_cache_json("benchmark.json")
    if not bench:
        print("Skipping benchmark chart (cache not found)")
        return
        
    models = list(bench["models"].keys())
    f1_means = [bench["models"][m]["f1_mean"] for m in models]
    f1_stds = [bench["models"][m]["f1_std"] for m in models]
    types = [bench["models"][m]["type"] for m in models]
    
    colors = ["#2b8a3e" if "Classical" in t else "#1971c2" for t in types]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, f1_means, yerr=f1_stds, capsize=5, color=colors, alpha=0.85, edgecolor="#333", width=0.55)
    
    ax.set_ylabel("F1-Score (5-Fold Stratified CV)", fontsize=11, weight="bold")
    ax.set_title("AgriShield Model Benchmark: Classical Baselines vs Quantum Classifiers", fontsize=13, weight="bold", pad=15)
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, mean in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.05, f"{mean:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")
        
    fig.tight_layout()
    fig.savefig(EXPORTS_DIR / "benchmark_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("--> Exported benchmark chart to exports/benchmark_comparison.png")


def export_risk_heatmap():
    """Export spatial risk heatmap for Week 15."""
    risk_maps = load_cache_json("risk_maps.json")
    if not risk_maps or "15" not in risk_maps:
        return
        
    matrix = np.array(risk_maps["15"]["risk_matrix"])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, origin="upper")
    
    # Annotate values
    for r in range(10):
        for c in range(10):
            val = matrix[r, c]
            color = "white" if (val > 0.7 or val < 0.25) else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)
            
    ax.set_title("AgriShield 10x10 Farm Grid: Pathogen Infection Risk (Week 15)", fontsize=12, weight="bold")
    ax.set_xlabel("Field Column Index", fontsize=10)
    ax.set_ylabel("Field Row Index", fontsize=10)
    fig.colorbar(cax, ax=ax, label="Computed Pathogen Risk [0.0 - 1.0]")
    fig.tight_layout()
    fig.savefig(EXPORTS_DIR / "risk_heatmap_week15.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("--> Exported risk heatmap to exports/risk_heatmap_week15.png")


def export_early_warning_timeline():
    """Export focal cell multi-spectral timeline."""
    ew = load_cache_json("early_warning.json")
    if not ew:
        return
        
    series = ew["focal_cell_timeline"]["series"]
    weeks = [s["week_idx"] for s in series]
    ndre = [s["ndre"] for s in series]
    ndvi = [s["ndvi"] for s in series]
    risk = [s["final_risk"] for s in series]
    stress = [s["stress_label"] for s in series]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    ax1.plot(weeks, ndre, color="#d9480f", marker="o", label="NDRE (Red-Edge Chlorophyll)", linewidth=2)
    ax1.plot(weeks, ndvi, color="#2f9e44", marker="s", label="NDVI (Canopy Biomass)", linewidth=2)
    ax1.plot(weeks, risk, color="#e03131", linestyle="--", label="AgriShield Quantum Risk", linewidth=2)
    ax1.axvline(x=5, color="purple", linestyle=":", label="Outbreak Seeding (W5)")
    ax1.axvline(x=7, color="black", linestyle="--", label="Visible Damage Manifested (W7)")
    ax1.axvspan(5, 7, color="yellow", alpha=0.2, label="Pre-Symptomatic Early Warning Window (14 Days)")
    
    ax1.set_ylabel("Spectral Reflectance / Risk", fontsize=10, weight="bold")
    ax1.set_title("Pre-Symptomatic Detection Window: Epicenter Cell (cell_02_03)", fontsize=12, weight="bold")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.step(weeks, stress, color="black", where="mid", linewidth=2, label="Ground Truth Visible Damage")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Healthy", "Symptomatic"])
    ax2.set_xlabel("Time Horizon (Weeks)", fontsize=10, weight="bold")
    ax2.set_ylabel("Visible Stress", fontsize=10, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(EXPORTS_DIR / "early_warning_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("--> Exported timeline to exports/early_warning_timeline.png")


def export_ibm_hardware_comparison():
    """Export Ideal vs Noisy vs Hardware kernel matrix comparisons."""
    ibm = load_cache_json("ibm_results.json")
    if not ibm:
        return
        
    k_ideal = np.array(ibm["kernel_matrices"]["ideal"])
    k_noisy = np.array(ibm["kernel_matrices"]["noisy"])
    k_hw = np.array(ibm["kernel_matrices"]["hardware"])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = axes[0].imshow(k_ideal, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Ideal Simulator (Statevector)", fontsize=10, weight="bold")
    
    im1 = axes[1].imshow(k_noisy, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Realistic Noisy Simulator (T1/T2)", fontsize=10, weight="bold")
    
    im2 = axes[2].imshow(k_hw, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"IBM Hardware Profile\n({ibm['backend_name'].split()[0]})", fontsize=10, weight="bold")
    
    for ax in axes:
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        
    fig.colorbar(im2, ax=axes.ravel().tolist(), label="Quantum Kernel Overlap", fraction=0.03, pad=0.04)
    fig.suptitle("IBM Quantum Kernel Matrix: Ideal vs Noisy vs Hardware", fontsize=12, weight="bold")
    fig.savefig(EXPORTS_DIR / "ibm_hardware_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("--> Exported hardware comparison to exports/ibm_hardware_comparison.png")


def export_architecture_diagram():
    """Export 4-step pipeline schematic (14x5 inch, dark theme, generous spacing)."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0B1A12")
    fig.patch.set_facecolor("#0B1A12")
    ax.set_facecolor("#0B1A12")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    steps = [
        ("1. SENSE", "Sentinel-2 L2A Multispectral\nBands: B4, B5, B8, B8A\nNASA POWER Daily Agro-weather", "#3DDC84", "STAGE 1"),
        ("2. ENCODE", "Agronomic PCA (4D Features)\nZZ-Feature Map (Entangled)\n2^n Hilbert State Space", "#5AB8FF", "STAGE 2"),
        ("3. PREDICT & SPREAD", "QSVC Fidelity Quantum Kernel\nCalibrated Stress Probability\nDistance-Decay Contagion", "#F5B83D", "STAGE 3"),
        ("4. ACTION & DISPATCH", "QAOA Combinatorial QUBO\nSpatial Scout Dispersion\nEnglish & Telugu Advisories", "#FF5A4E", "STAGE 4")
    ]
    
    box_w = 0.20
    box_h = 0.62
    y_pos = 0.18
    
    for idx, (title, desc, accent_color, badge) in enumerate(steps):
        x_pos = 0.04 + idx * 0.245
        rect = patches.FancyBboxPatch(
            (x_pos, y_pos), box_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor="#12261A",
            edgecolor=accent_color,
            linewidth=2.2
        )
        ax.add_patch(rect)
        
        # Header with badge and title
        ax.text(x_pos + box_w / 2, y_pos + box_h - 0.09, title,
                ha="center", va="center", fontsize=12, weight="bold", color=accent_color)
        
        # Divider line
        ax.plot([x_pos + 0.02, x_pos + box_w - 0.02], [y_pos + box_h - 0.16, y_pos + box_h - 0.16],
                color="#8FA89A", alpha=0.35, lw=1)
                
        # Description text
        ax.text(x_pos + box_w / 2, y_pos + box_h / 2 - 0.06, desc,
                ha="center", va="center", fontsize=9.5, color="#EAF3E6", linespacing=1.6)
        
        # Arrow connector to next step
        if idx < 3:
            arrow_start_x = x_pos + box_w + 0.005
            arrow_end_x = x_pos + box_w + 0.040
            ax.annotate(
                "",
                xy=(arrow_end_x, y_pos + box_h / 2),
                xytext=(arrow_start_x, y_pos + box_h / 2),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.5", lw=2.2, color="#3DDC84")
            )
            
    ax.text(0.5, 0.94, "AgriShield Quantum: End-to-End Early Warning Architecture",
            ha="center", va="center", fontsize=15, weight="bold", color="#EAF3E6")
    ax.text(0.5, 0.88, "Sense -> Encode -> Predict & Spread -> Action & Dispatch",
            ha="center", va="center", fontsize=11, color="#8FA89A")
            
    fig.tight_layout()
    fig.savefig(EXPORTS_DIR / "architecture.png", dpi=200, bbox_inches="tight", facecolor="#0B1A12")
    fig.savefig(EXPORTS_DIR / "system_architecture.png", dpi=200, bbox_inches="tight", facecolor="#0B1A12")
    plt.close(fig)
    print("--> Exported redesigned dark-theme architecture to exports/architecture.png and exports/system_architecture.png")


def run_all_exports():
    """Execute all asset export functions."""
    print("=== Generating High-Resolution Presentation Figures ===")
    export_circuits()
    export_benchmark_chart()
    export_risk_heatmap()
    export_early_warning_timeline()
    export_ibm_hardware_comparison()
    export_architecture_diagram()
    print("=== All figures exported successfully to /exports ===")


if __name__ == "__main__":
    run_all_exports()
