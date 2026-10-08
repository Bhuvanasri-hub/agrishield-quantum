# 🌾 AgriShield Quantum

> **"Predict. Protect. Prevent."**  
> *Qiskit Fall Fest 2026 — Problem UC-034: Pest & Crop-Disease Early Warning using Quantum AI/ML*

---

## 📌 Executive Summary
**AgriShield Quantum** is an end-to-end quantum-classical decision support platform designed to detect agricultural pest and foliar crop-disease outbreaks **7 to 14 days before visible leaf symptoms appear**.

In agrarian regions like Andhra Pradesh (e.g. Krishna District paddy belts), fungal blast (*Magnaporthe oryzae*), sheath blight, and hopper-burn infestations spread rapidly across contiguous plots. By the time visual symptoms become conspicuous, cellular degradation has already decimated 30%–70% of potential yield. AgriShield Quantum pairs multispectral remote sensing (Sentinel-2 red-edge reflectances) with **Quantum Support Vector Classifiers (QSVC)** and **Quantum Approximate Optimization Algorithm (QAOA)** scouting dispatch to protect smallholder farmers.

---

## 🔬 Scientific Honesty & Methodology Disclosure
Scientific integrity is the core principle of this project:
1. **No Faked Numbers:** All benchmark metrics (Accuracy, F1, Recall, ROC-AUC, Execution Time) are empirical measurements from 5-fold Stratified Cross-Validation on Qiskit simulators and IBM Quantum hardware profiles.
2. **Transparent Classical Baseline Comparison:** Classical models (Random Forest, RBF SVC) perform exceptionally well on small tabular agronomic vectors with millisecond latencies. We report exactly when classical baselines match or exceed quantum models.
3. **Qubit Ablation Findings:** In our ablation study (2 vs 4 vs 6 qubits), QSVC achieves **87.5% accuracy (0.871 F1)** on 2 qubits with dense non-linear mapping. At 4 and 6 qubits without custom ansatz tuning, kernel concentration in the expanding $2^n$-dimensional Hilbert space reduces generalizability.
4. **Data & Proxy Labels Clearly Labeled:**
   - **Synthetic Layer:** Clearly marked as *"Synthetic data for prototype validation"* (models spatio-temporal contagion and pre-symptomatic red-edge drop).
   - **Real Layer:** Uses real NASA POWER daily agroclimatology for Krishna District ($16.216^\circ\text{N}, 80.985^\circ\text{E}$). Sentinel-2 multispectral bands use synthetic fallback formulas because Copernicus CDSE credentials are not configured in this prototype. Labels are explicitly disclosed as *"Proxy labels derived from >15% NDVI seasonal baseline drop; not microbiological laboratory assay labels."*

---

## 🏗️ System Architecture: The 4-Step Pipeline

```
  +------------------+      +--------------------+      +-----------------------+      +-------------------------+
  |    1. SENSE      | ---> |     2. ENCODE      | ---> |  3. PREDICT & SPREAD  | ---> |   4. ACTION & DISPATCH  |
  | Sentinel-2 L2A   |      | Agronomic PCA (4D) |      | QSVC Fidelity Kernel  |      | QAOA Scout Allocation   |
  | (B4,B5,B8,B8A)   |      | ZZFeatureMap       |      | Calibrated Prob       |      | Bilingual Advisories    |
  | NASA POWER Daily |      | Hilbert State Space|      | Spatial Contagion     |      | (English & Telugu)      |
  +------------------+      +--------------------+      +-----------------------+      +-------------------------+
```

### 1. Sense
Ingests Sentinel-2 multispectral bands (B4 Red 665nm, B5 RedEdge-1 705nm, B8 NIR 842nm, B8A Narrow NIR 865nm) and daily NASA POWER weather (Temperature $T_{2M}$, Relative Humidity $RH_{2M}$, Precipitation $P$).

### 2. Encode
Computes agronomic indices:
- $\text{NDVI} = \frac{B8 - B4}{B8 + B4}$ (Biomass vigor)
- $\text{NDRE} = \frac{B8A - B5}{B8A + B5}$ (Red-edge chlorophyll stress precursor)
- Vapor Pressure Deficit ($\text{VPD}$) & Dew-point proxy for fungal spore germination.
Reduces features to $n$-qubits via PCA and normalizes angles into $[-\pi/2, \pi/2]$. Enters quantum state space using the **$ZZ$-Feature Map**:
$$\Phi(x) = \sum_{i=1}^n x_i Z_i + \sum_{i < j} (\pi - x_i)(\pi - x_j) Z_i Z_j$$

### 3. Predict & Spread
Evaluates quantum state overlap fidelity using `ComputeUncompute` with `FidelityQuantumKernel`:
$$K(x_i, x_j) = |\langle \psi(x_j) | \psi(x_i) \rangle|^2$$
Calibrates stress probabilities via logistic sigmoid scaling, then applies distance-decaying and humidity-amplified neighbor spread:
$$\text{Risk}_{\text{final}}(i, j) = \alpha \cdot \text{Risk}_{\text{own}}(i, j) + \beta \sum_{(r,c) \in \mathcal{N}(i,j)} \frac{1}{d^{1.5}} \left(\frac{H}{100}\right)^{1.8} \text{Risk}_{\text{own}}(r, c)$$

### 4. Action & Dispatch
Builds a combinatorial QUBO for dispatching $K$ field teams across candidate outbreak plots:
$$\max \sum_{i} r_i x_i - \lambda_{\text{adj}} \sum_{\text{adj}(i,j)} x_i x_j \quad \text{s.t.} \quad \sum x_i = K$$
Solved using **QAOA** ($p=1$, COBYLA) via `MinimumEigenOptimizer`. Dispatches bilingual alerts in English and Telugu to local Rythu Bharosa Kendras (RBKs).

---

## 📊 Empirical Results Summary

### Model Benchmark (5-Fold Stratified Cross-Validation)
| Model | Type | Accuracy | Recall | F1-Score | ROC-AUC | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVC (RBF)** | Classical | 96.7% ± 3.1% | 0.983 ± 0.033 | 0.968 ± 0.029 | 0.994 | 0.035s |
| **Random Forest** | Classical | 95.8% ± 2.6% | 0.950 ± 0.041 | 0.959 ± 0.024 | 0.990 | 0.180s |
| **Logistic Regression** | Classical | 95.8% ± 3.7% | 0.967 ± 0.067 | 0.958 ± 0.038 | 0.996 | 0.003s |
| **QSVC (2 Qubits)** | Quantum Kernel | 87.5% ± 3.4% | 0.883 ± 0.038 | 0.871 ± 0.031 | 0.903 | 14.9s |
| **QSVC (4 Qubits)** | Quantum Kernel | 52.5% ± 3.3% | 0.617 ± 0.194 | 0.552 ± 0.075 | 0.590 | 20.3s |
| **VQC (4 Qubits)** | Variational | 48.6% ± 10.9% | 0.500 ± 0.118 | 0.492 ± 0.113 | 0.486 | 10.8s |

### Early Warning Lead Time
- **Mean Warning Lead Time:** **11.48 Days** (Median: 7.0 Days) before visible field damage.
- **5-Day Lead Horizon ($k=5$d):** Recall = **83.8%**, F1 = **0.802**, False Alarm Rate = **9.6%**.
- **10-Day Lead Horizon ($k=10$d):** Recall = **74.9%**, F1 = **0.819**, False Alarm Rate = **3.0%**.
- **15-Day Lead Horizon ($k=15$d):** Recall = **69.6%**, F1 = **0.672**, False Alarm Rate = **15.4%**.

### QAOA Scouting Dispatch
- **Experiment 1 (10 Plots, $K=3$ Teams):**
  - Exact Brute Force Objective: **2.4274** (0.0002s)
  - Classical Greedy Objective: **2.4274** (0.00002s)
  - QAOA Objective ($p=1$, COBYLA): **2.4125** (Approximation Ratio: **99.39%**)
- **Experiment 2 (20 Plots, $K=4$ Teams):**
  - Brute force checks 4,845 combinations in 0.012s (Obj = 3.1843).
  - Greedy evaluates in <0.001s (Obj = 3.1220).
  - Demonstrates why physical QPUs are mandatory: classical statevector simulation of 20-qubit QAOA requires 16 Terabytes of RAM ($2^{20} \times 2^{20}$ operator).

### Hardware Noise Simulation (ibm_brisbane Calibration Model)
- Simulated locally using Qiskit Aer with noise model parameters derived from `ibm_brisbane` (Eagle architecture) calibration data; not run on real physical quantum hardware.
- Transpiled with `optimization_level=3` and simulated readout error mitigation.
- Frobenius norm kernel matrix deviation between Ideal and simulated noise model: **0.2185** (Mean Absolute Error: **0.0256**).

---

## 📁 Repository Structure
```
agrishield-quantum/
├── app.py                      # Main Streamlit web application (9 tabs)
├── requirements.txt            # Pinned reproducible dependencies
├── .env.example                # Template for IBM and Copernicus secrets
├── .gitignore                  # Ignores secrets and large caches
├── README.md                   # Complete documentation
├── cache/                      # Precomputed JSON/CSV artifacts (instant startup)
│   ├── synthetic_farm_data.csv # 10x10 farm grid over 24 weeks
│   ├── benchmark.json          # 5-fold CV & ablation numbers
│   ├── risk_maps.json          # Weekly spatial contagion matrices
│   ├── qaoa_scout.json         # QAOA vs Brute Force vs Greedy results
│   ├── early_warning.json      # Lead time metrics & focal cell series
│   ├── ibm_results.json        # Hardware vs Ideal vs Noisy comparisons
│   ├── real_farm_data.csv      # NASA POWER + Sentinel-2 proxy layer
│   └── real_metadata.json      # Satellite metadata & proxy label disclaimers
├── exports/                    # High-res PNG assets for slide presentations
│   ├── system_architecture.png
│   ├── quantum_circuits.png
│   ├── benchmark_comparison.png
│   ├── risk_heatmap_week15.png
│   ├── early_warning_timeline.png
│   └── ibm_hardware_comparison.png
├── src/
│   ├── __init__.py
│   ├── utils.py                # Path management and cache I/O
│   ├── data_synth.py           # Synthetic farm grid generator
│   ├── data_real.py            # NASA POWER & Sentinel-2 downloader
│   ├── features.py             # Domain feature engineering & PCA
│   ├── quantum_models.py       # QSVC, VQC, ZZFeatureMap, ansatz
│   ├── classical_models.py     # SVC (RBF), Random Forest, Logistic Reg
│   ├── benchmark.py            # 5-fold cross validation runner
│   ├── risk.py                 # Calibrated risk & neighbor spread
│   ├── qaoa_scout.py           # QUBO & QAOA scout allocation
│   ├── early_warning.py        # Lead time evaluation across horizons
│   ├── ibm_hardware.py         # IBM Qiskit Runtime & noise simulation
│   └── alerts.py               # English & Telugu bilingual notifications
└── scripts/
    ├── run_all_precompute.py   # Master pipeline recomputation script
    └── export_figures.py       # PNG visualization generator
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 through 3.14)
- Git

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/your-username/agrishield-quantum.git
cd agrishield-quantum
pip install -r requirements.txt
```

### 3. Configure Secrets (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your credentials if running on physical IBM Quantum hardware or downloading fresh Copernicus Sentinel-2 tiles:
- `IBM_QUANTUM_TOKEN`: from [quantum.ibm.com](https://quantum.ibm.com/)
- `COPERNICUS_USERNAME` / `COPERNICUS_PASSWORD`: from [dataspace.copernicus.eu](https://dataspace.copernicus.eu/)

*(Note: All caches are precomputed and committed, so the app runs instantly without needing external API keys!)*

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. The app starts in **< 2 seconds**.

---

## 🔄 Recomputing Caches
To re-run the entire pipeline from scratch:
```bash
python -m scripts.run_all_precompute
```
Or execute individual phases:
- Synthetic Data: `python -m src.data_synth`
- Real Data (NASA + Sentinel-2): `python -m src.data_real`
- Model Benchmark: `python -m src.benchmark`
- Spatial Risk Maps: `python -m src.risk`
- QAOA Scouting: `python -m src.qaoa_scout`
- Early Warning: `python -m src.early_warning`
- IBM Hardware: `python -m src.ibm_hardware`
- Export Figures: `python -m scripts.export_figures`

---

## 👨‍🌾 Farmer Persona: Krishna District, Andhra Pradesh
AgriShield Quantum includes a localized persona for extension demonstration:
- **Farmer:** Venkata Subba Rao (వెంకట సుబ్బారావు)
- **Location:** Avanigadda / Gudivada, Krishna District, AP (కృష్ణా జిల్లా, ఆంధ్రప్రదేశ్)
- **Crop:** Kharif Paddy (*BPT 5204 Samba Mahsuri*)
- **Landholding:** 4.5 Acres (divided into monitored sub-plots)
- **Extension Center:** Avanigadda Rythu Bharosa Kendra (RBK #14)

---

## 📜 License & Citation
Developed for the **Qiskit Fall Fest 2026 Hackathon**.  
Released under the MIT License.
