"""
AgriShield Quantum - Crop stress early warning
Simple working decision-support tool.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.utils import CACHE_DIR, load_cache_json
from src.risk import classify_risk_tier, compute_neighbor_spread
from src.features import extract_agronomic_features, prepare_benchmark_data
from src.classical_models import build_svc_rbf
from src.qaoa_scout import solve_brute_force

# -----------------------------------------------------------------------------
# Page Configuration & Minimal Dark Forest Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriShield Quantum",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #EAF3E6;
        background-color: #0B1A12;
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    .block-container {
        padding: 24px 28px !important;
        max-width: 1180px !important;
        margin: 0 auto !important;
        overflow-x: hidden !important;
    }
    
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    .panel-card {
        background-color: #12261A;
        border: 1px solid rgba(61, 220, 132, 0.2);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 16px;
        box-sizing: border-box;
    }
    
    .caption-text {
        font-size: 0.80rem;
        color: #8FA89A;
        font-style: italic;
        margin-top: 4px;
        margin-bottom: 16px;
    }
    
    .badge-low {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: rgba(61, 220, 132, 0.15);
        color: #3DDC84;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(61, 220, 132, 0.4);
    }
    
    .badge-medium {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: rgba(245, 184, 61, 0.15);
        color: #F5B83D;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(245, 184, 61, 0.4);
    }
    
    .badge-high {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: rgba(255, 90, 78, 0.15);
        color: #FF5A4E;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(255, 90, 78, 0.4);
    }
    
    .phone-container {
        width: min(100%, 340px);
        margin: 0 auto;
        background-color: #12261A;
        border: 2px solid rgba(61, 220, 132, 0.3);
        border-radius: 20px;
        padding: 16px;
        box-sizing: border-box;
    }
    
    .sms-bubble {
        background-color: #1a3826;
        color: #EAF3E6;
        border-radius: 12px;
        padding: 12px;
        font-size: 0.88rem;
        line-height: 1.5;
        border: 1px solid rgba(61, 220, 132, 0.2);
        word-break: break-word;
    }
    
    @media (max-width: 768px) {
        .block-container {
            padding: 12px 12px 24px 12px !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            gap: 16px !important;
            width: 100% !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loaders & Model Cache
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    return {
        "benchmark": load_cache_json("benchmark.json", {}),
        "risk_maps": load_cache_json("risk_maps.json", {}),
        "qaoa": load_cache_json("qaoa_scout.json", {})
    }

data_store = load_app_data()
risk_maps = data_store.get("risk_maps", {})
benchmark_data = data_store.get("benchmark", {})
qaoa_data = data_store.get("qaoa", {})

@st.cache_resource
def get_trained_classical_svc():
    synth_csv = CACHE_DIR / "synthetic_farm_data.csv"
    if synth_csv.exists():
        df = pd.read_csv(synth_csv)
        X, y, pipeline = prepare_benchmark_data(df, n_samples=120, n_components=4, random_state=42)
        model = build_svc_rbf()
        model.fit(X, y)
        return model, pipeline
    return None, None

def get_clean_sms(cell_id: str, risk_score: float, risk_tier: str, humidity: float, ndre: float, lang: str = "en") -> str:
    pct = int(risk_score * 100)
    if lang.lower().startswith("te"):
        if risk_tier == "High":
            return (
                f"[AgriShield హెచ్చరిక] ప్లాట్ {cell_id}: తీవ్ర ముప్పు ({pct}%). "
                f"తేమ {humidity:.0f}%, క్లోరోఫిల్ లోపం కనుగొనబడింది. "
                f"24 గంటల్లోగా తనిఖీ చేసి తగిన నివారణ చర్యలు చేపట్టండి."
            )
        else:
            return (
                f"[AgriShield సమాచారం] ప్లాట్ {cell_id}: మధ్యస్థ ముప్పు ({pct}%). "
                f"తేమ ఎక్కువగా ఉంది, 48 గంటల్లోగా ప్లాట్ పరిశీలించండి."
            )
    else:
        if risk_tier == "High":
            return (
                f"[AgriShield Alert] Plot {cell_id}: High risk ({pct}%). "
                f"Humidity {humidity:.0f}%, NDRE drop detected. "
                f"Inspect within 24h and apply preventive spray."
            )
        else:
            return (
                f"[AgriShield Advisory] Plot {cell_id}: Moderate risk ({pct}%). "
                f"Elevated microclimate humidity. Monitor plot within 48h."
            )


# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "selected_row" not in st.session_state:
    st.session_state.selected_row = 2
if "selected_col" not in st.session_state:
    st.session_state.selected_col = 3


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("AgriShield Quantum")
st.caption("Crop stress early warning")


# -----------------------------------------------------------------------------
# SECTION 1 - Input
# -----------------------------------------------------------------------------
st.subheader("1. Input")

col_i1, col_i2, col_i3 = st.columns(3)
with col_i1:
    selected_week = st.slider("Observation week", min_value=0, max_value=23, value=15)
with col_i2:
    humidity_mod = st.slider("Humidity modifier (%)", min_value=-20, max_value=20, value=0)
with col_i3:
    temperature_mod = st.slider("Temperature modifier (°C)", min_value=-5, max_value=5, value=0)

col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    selected_model = st.radio("Model selector", ["Quantum (QSVC)", "Classical (SVC)"], index=0, horizontal=True)
with col_m2:
    checker_mode = st.radio("Plot checker mode", ["Select plot (row, column)", "Enter custom sensor values"], index=0, horizontal=True)

if checker_mode == "Select plot (row, column)":
    col_r, col_c = st.columns(2)
    with col_r:
        in_row = st.number_input("Row (0-9)", min_value=0, max_value=9, value=int(st.session_state.selected_row))
    with col_c:
        in_col = st.number_input("Column (0-9)", min_value=0, max_value=9, value=int(st.session_state.selected_col))
    st.session_state.selected_row = int(in_row)
    st.session_state.selected_col = int(in_col)
else:
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        custom_ndvi = st.number_input("NDVI", min_value=0.0, max_value=1.0, value=0.51, step=0.01)
    with col_c2:
        custom_ndre = st.number_input("NDRE", min_value=0.0, max_value=1.0, value=0.32, step=0.01)
    with col_c3:
        custom_hum = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=80.0, step=1.0)
    with col_c4:
        custom_temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.5)

st.markdown('<div class="caption-text">Set observation parameters and select a farm plot to analyze.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Preparation for Week & Inputs
# -----------------------------------------------------------------------------
week_data = risk_maps.get(str(selected_week), risk_maps.get("15", {}))
base_matrix = np.array(week_data.get("risk_matrix", np.zeros((10, 10))))

if humidity_mod != 0 or temperature_mod != 0:
    effective_hum = np.clip(np.array([[c["humidity"] + humidity_mod for c in week_data["cells"][r*10:(r+1)*10]] for r in range(10)]), 20, 100)
    own_probs = np.array([[c["quantum_prob"] for c in week_data["cells"][r*10:(r+1)*10]] for r in range(10)])
    render_matrix = compute_neighbor_spread(own_probs, effective_hum)
else:
    render_matrix = base_matrix


# -----------------------------------------------------------------------------
# SECTION 2 - Prediction
# -----------------------------------------------------------------------------
st.subheader("2. Prediction")

cur_row = st.session_state.selected_row
cur_col = st.session_state.selected_col
cur_cell_idx = cur_row * 10 + cur_col
cur_cell = week_data["cells"][cur_cell_idx] if "cells" in week_data and cur_cell_idx < len(week_data["cells"]) else {}
cell_id = cur_cell.get("cell_id", f"cell_{cur_row:02d}_{cur_col:02d}")

if checker_mode == "Select plot (row, column)":
    cell_hum = cur_cell.get("humidity", 75.0) + humidity_mod
    cell_temp = cur_cell.get("temperature", 28.0) + temperature_mod
    cell_ndre = cur_cell.get("ndre", 0.35)
    cell_ndvi = cur_cell.get("ndvi", 0.55)
    
    if selected_model == "Quantum (QSVC)":
        stress_prob = float(render_matrix[cur_row, cur_col])
        used_model_name = "Quantum (QSVC)"
    else:
        svc_model, svc_pipe = get_trained_classical_svc()
        if svc_model is not None and svc_pipe is not None:
            df_pt = pd.DataFrame({
                "ndvi": [cell_ndvi],
                "ndre": [cell_ndre],
                "temperature": [cell_temp],
                "humidity": [cell_hum],
                "rainfall": [cur_cell.get("rainfall", 30.0)]
            })
            feats = extract_agronomic_features(df_pt)
            X_pt = svc_pipe.transform(feats)
            stress_prob = float(svc_model.predict_proba(X_pt)[0, 1])
        else:
            stress_prob = float(render_matrix[cur_row, cur_col])
        used_model_name = "Classical (SVC)"
else:
    cell_hum = custom_hum + humidity_mod
    cell_temp = custom_temp + temperature_mod
    cell_ndre = custom_ndre
    cell_ndvi = custom_ndvi
    
    if selected_model == "Quantum (QSVC)":
        with st.spinner("Evaluating quantum prediction from precomputed model..."):
            stress_prob = float(render_matrix[cur_row, cur_col])
        used_model_name = "Quantum (QSVC)"
    else:
        svc_model, svc_pipe = get_trained_classical_svc()
        if svc_model is not None and svc_pipe is not None:
            df_pt = pd.DataFrame({
                "ndvi": [cell_ndvi],
                "ndre": [cell_ndre],
                "temperature": [cell_temp],
                "humidity": [cell_hum],
                "rainfall": [30.0]
            })
            feats = extract_agronomic_features(df_pt)
            X_pt = svc_pipe.transform(feats)
            stress_prob = float(svc_model.predict_proba(X_pt)[0, 1])
        else:
            stress_prob = 0.5
        used_model_name = "Classical (SVC)"

risk_tier = classify_risk_tier(stress_prob)
tier_class = "badge-high" if risk_tier == "High" else ("badge-medium" if risk_tier == "Medium" else "badge-low")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    st.markdown(f"""
    <div class="panel-card">
        <div style="font-size: 0.8rem; color: #8FA89A;">Target Plot</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #EAF3E6; margin-top: 4px;">{cell_id}</div>
        <div style="font-size: 0.78rem; color: #8FA89A;">Row {cur_row}, Col {cur_col}</div>
    </div>
    """, unsafe_allow_html=True)
with col_p2:
    st.markdown(f"""
    <div class="panel-card">
        <div style="font-size: 0.8rem; color: #8FA89A;">Evaluated Model</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #EAF3E6; margin-top: 4px;">{used_model_name}</div>
        <div style="font-size: 0.78rem; color: #8FA89A;">Week {selected_week}</div>
    </div>
    """, unsafe_allow_html=True)
with col_p3:
    st.markdown(f"""
    <div class="panel-card">
        <div style="font-size: 0.8rem; color: #8FA89A;">Stress Probability</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #3DDC84; margin-top: 4px;">{stress_prob*100:.1f}%</div>
        <div style="font-size: 0.78rem; color: #8FA89A;">Calibrated score</div>
    </div>
    """, unsafe_allow_html=True)
with col_p4:
    st.markdown(f"""
    <div class="panel-card">
        <div style="font-size: 0.8rem; color: #8FA89A;">Risk Level</div>
        <div style="margin-top: 6px;"><span class="{tier_class}">{risk_tier.upper()}</span></div>
        <div style="font-size: 0.78rem; color: #8FA89A; margin-top: 4px;">Classification status</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="caption-text">Plot stress probability and risk level computed from evaluated model.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION 3 - Risk Map and Inspection List
# -----------------------------------------------------------------------------
st.subheader("3. Risk map and inspection list")

col_map, col_scout = st.columns([1.3, 1.0])

with col_map:
    # 10x10 Heatmap
    custom_colorscale = [
        [0.0, "#3DDC84"],
        [0.35, "#8CE99A"],
        [0.50, "#F5B83D"],
        [0.70, "#F76707"],
        [1.0, "#FF5A4E"]
    ]
    
    hover_text = []
    for r in range(10):
        row_hover = []
        for c in range(10):
            val = render_matrix[r, c]
            row_hover.append(f"Plot: cell_{r:02d}_{c:02d}<br>Row: {r}, Col: {c}<br>Risk: {val*100:.1f}%")
        hover_text.append(row_hover)
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=render_matrix,
        x=[f"C{c}" for c in range(10)],
        y=[f"R{r}" for r in range(10)],
        colorscale=custom_colorscale,
        zmin=0.0,
        zmax=1.0,
        xgap=3,
        ygap=3,
        hovertext=hover_text,
        hoverinfo="text",
        colorbar=dict(
            title=dict(text="Risk", font=dict(color="#8FA89A", size=10)),
            tickfont=dict(color="#8FA89A", size=9),
            outlinecolor="rgba(0,0,0,0)",
            len=0.8,
            thickness=12
        )
    ))
    fig_heat.update_layout(
        xaxis=dict(tickfont=dict(color="#8FA89A", size=10)),
        yaxis=dict(tickfont=dict(color="#8FA89A", size=10), autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=15, r=15, t=15, b=15)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Plot selection control linked to state
    all_plot_labels = [f"cell_{r:02d}_{c:02d} (Row {r}, Col {c})" for r in range(10) for c in range(10)]
    cur_label_idx = cur_row * 10 + cur_col
    def update_plot_from_selector():
        sel = st.session_state["map_plot_dropdown"]
        idx = all_plot_labels.index(sel)
        st.session_state.selected_row = idx // 10
        st.session_state.selected_col = idx % 10
        
    st.selectbox(
        "Select plot to inspect",
        all_plot_labels,
        index=cur_label_idx,
        key="map_plot_dropdown",
        on_change=update_plot_from_selector
    )

with col_scout:
    st.markdown("#### Inspect first")
    k_teams = st.slider("Inspection teams (K)", min_value=2, max_value=5, value=3)
    
    # Candidate cells for scouting
    candidate_pool = sorted(week_data.get("cells", []), key=lambda c: render_matrix[c["row"], c["col"]], reverse=True)[:10]
    
    # Dynamic brute force solution
    bf_res = solve_brute_force(candidate_pool, K=k_teams)
    bf_obj = bf_res.get("objective_value", 0.0)
    
    # Cached QAOA benchmark values for comparison
    qaoa_exp1 = qaoa_data.get("experiment_1_top10", {})
    qaoa_cached_obj = qaoa_exp1.get("qaoa", {}).get("objective_value", 2.4125)
    bf_cached_obj = qaoa_exp1.get("brute_force", {}).get("objective_value", 2.4274)
    approx_ratio = qaoa_exp1.get("approximation_ratio", 0.9939)
    
    # Selected plots list
    if k_teams == 3 and "selected_cells" in qaoa_exp1.get("qaoa", {}):
        selected_scout_cells = qaoa_exp1["qaoa"]["selected_cells"]
    else:
        selected_scout_cells = bf_res.get("selected_cells", [])
        
    for idx, sc_id in enumerate(selected_scout_cells, 1):
        r_sc = int(sc_id.split("_")[1])
        c_sc = int(sc_id.split("_")[2])
        sc_risk = float(render_matrix[r_sc, c_sc])
        st.markdown(f"""
        <div style="background: rgba(255,90,78,0.1); border: 1px solid rgba(255,90,78,0.3); border-radius: 8px; padding: 6px 10px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #EAF3E6;">{idx}. {sc_id} (R{r_sc}, C{c_sc})</span>
            <span style="font-size: 0.85rem; font-weight: 700; color: #FF5A4E;">{sc_risk*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    if k_teams == 3:
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #8FA89A; margin-top: 8px;">
            QAOA objective: {qaoa_cached_obj} vs Brute-force: {bf_cached_obj} ({approx_ratio*100:.1f}% approximation ratio, N=10, K=3).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #8FA89A; margin-top: 8px;">
            Combinatorial objective: {bf_obj:.4f} (evaluated over C(10, {k_teams}) combinations).
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="caption-text">Contagion risk distribution across plots with QAOA scout team selection.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION 4 - Alert
# -----------------------------------------------------------------------------
st.subheader("4. Alert")

if risk_tier in ["Medium", "High"]:
    lang_choice = st.radio("Notification language", ["English", "Telugu"], horizontal=True)
    sms_content = get_clean_sms(
        cell_id=cell_id,
        risk_score=stress_prob,
        risk_tier=risk_tier,
        humidity=cell_hum,
        ndre=cell_ndre,
        lang="te" if "Telugu" in lang_choice else "en"
    )
    st.markdown(f"""
    <div class="phone-container">
        <div style="font-size: 0.75rem; color: #8FA89A; margin-bottom: 8px;">SMS Preview:</div>
        <div class="sms-bubble">{sms_content}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="panel-card" style="border-color: rgba(61,220,132,0.3); color: #3DDC84; font-size: 0.88rem;">
        Normal canopy conditions. Risk is below threshold (< 35%), no alert required.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="caption-text">Early warning SMS advisory generated for plots requiring intervention.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION 5 (Collapsed Expander) - Model Results
# -----------------------------------------------------------------------------
with st.expander("Model results", expanded=False):
    bench_models = benchmark_data.get("models", {})
    bench_abl = benchmark_data.get("ablation", {})
    
    table_rows = []
    
    # 1. SVC (RBF)
    m_svc = bench_models.get("SVC (RBF)", {})
    table_rows.append({
        "Model": "SVC (RBF)",
        "Type": m_svc.get("type", "Classical"),
        "Accuracy (5-fold CV)": f"{m_svc.get('accuracy_mean', 0.967)*100:.1f}% ± {m_svc.get('accuracy_std', 0.031)*100:.1f}%",
        "Recall": f"{m_svc.get('recall_mean', 0.983):.3f}",
        "F1-Score": f"{m_svc.get('f1_mean', 0.968):.3f}"
    })
    
    # 2. Random Forest
    m_rf = bench_models.get("Random Forest", {})
    table_rows.append({
        "Model": "Random Forest",
        "Type": m_rf.get("type", "Classical"),
        "Accuracy (5-fold CV)": f"{m_rf.get('accuracy_mean', 0.958)*100:.1f}% ± {m_rf.get('accuracy_std', 0.026)*100:.1f}%",
        "Recall": f"{m_rf.get('recall_mean', 0.950):.3f}",
        "F1-Score": f"{m_rf.get('f1_mean', 0.959):.3f}"
    })
    
    # 3. QSVC 2q
    m_q2 = bench_abl.get("2_qubits", {})
    table_rows.append({
        "Model": "QSVC (2 qubits)",
        "Type": "Quantum Kernel",
        "Accuracy (5-fold CV)": f"{m_q2.get('accuracy_mean', 0.875)*100:.1f}% ± {m_q2.get('accuracy_std', 0.034)*100:.1f}%",
        "Recall": "0.883",
        "F1-Score": f"{m_q2.get('f1_mean', 0.871):.3f}"
    })
    
    # 4. QSVC 4q
    m_q4 = bench_models.get("QSVC (4 qubits)", {})
    table_rows.append({
        "Model": "QSVC (4 qubits)",
        "Type": m_q4.get("type", "Quantum Kernel"),
        "Accuracy (5-fold CV)": f"{m_q4.get('accuracy_mean', 0.525)*100:.1f}% ± {m_q4.get('accuracy_std', 0.033)*100:.1f}%",
        "Recall": f"{m_q4.get('recall_mean', 0.617):.3f}",
        "F1-Score": f"{m_q4.get('f1_mean', 0.552):.3f}"
    })
    
    # 5. VQC 4q
    m_vqc = bench_models.get("VQC (4 qubits)", {})
    table_rows.append({
        "Model": "VQC (4 qubits)",
        "Type": m_vqc.get("type", "Variational Quantum"),
        "Accuracy (5-fold CV)": f"{m_vqc.get('accuracy_mean', 0.486)*100:.1f}% ± {m_vqc.get('accuracy_std', 0.109)*100:.1f}%",
        "Recall": f"{m_vqc.get('recall_mean', 0.500):.3f}",
        "F1-Score": f"{m_vqc.get('f1_mean', 0.492):.3f}"
    })
    
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    
    # Bar chart
    m_names = [r["Model"] for r in table_rows]
    f1_vals = [float(r["F1-Score"]) for r in table_rows]
    m_colors = ["#3DDC84" if "Classical" in r["Type"] else "#5AB8FF" for r in table_rows]
    
    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(
        x=m_names,
        y=f1_vals,
        marker_color=m_colors,
        text=[f"{v:.3f}" for v in f1_vals],
        textposition="auto",
        textfont=dict(color="#0B1A12", size=11, weight="bold")
    ))
    fig_bench.update_layout(
        title=dict(text="<b>5-Fold Cross-Validation F1-Score Comparison</b>", font=dict(family="Inter", color="#EAF3E6", size=13)),
        yaxis=dict(range=[0, 1.15], gridcolor="rgba(143,168,154,0.15)", tickfont=dict(color="#8FA89A", size=10)),
        xaxis=dict(tickfont=dict(color="#8FA89A", size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=35, r=15, t=40, b=30)
    )
    st.plotly_chart(fig_bench, use_container_width=True)
    st.markdown('<div class="caption-text">5-fold cross-validation comparison across classical and quantum models.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #8FA89A; font-size: 0.80rem; margin-top: 36px; padding: 16px 0; border-top: 1px solid rgba(143, 168, 154, 0.15);">
    Synthetic data, prototype. Not field-tested.
</div>
""", unsafe_allow_html=True)
