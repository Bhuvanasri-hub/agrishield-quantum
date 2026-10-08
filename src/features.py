"""
AgriShield Quantum - Feature Engineering & Preprocessing Pipeline (Phase 2)
Cleans agronomic and weather sensor data, standardizes, applies PCA reduction
to n components (corresponding to n qubits), and scales features into quantum state angles.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


BASE_COLUMNS = ["ndvi", "ndre", "temperature", "humidity", "rainfall"]
FEATURE_COLUMNS = [
    "ndvi",
    "ndre",
    "temperature",
    "humidity",
    "rainfall",
    "ndre_ndvi_ratio",     # Chlorophyll vigor proxy
    "dew_point_proxy",     # Fungal spore condensation index (T - (100-H)/5)
    "vpd_proxy"            # Vapor pressure deficit proxy
]


def extract_agronomic_features(df: pd.DataFrame) -> np.ndarray:
    """Extract raw and domain-engineered agronomic features."""
    t = df["temperature"].to_numpy(dtype=np.float64)
    h = np.clip(df["humidity"].to_numpy(dtype=np.float64), 1.0, 100.0)
    ndvi = df["ndvi"].to_numpy(dtype=np.float64)
    ndre = df["ndre"].to_numpy(dtype=np.float64)
    rain = df["rainfall"].to_numpy(dtype=np.float64)
    
    # Agronomic indices
    ratio = ndre / np.maximum(ndvi, 0.05)
    dew_point = t - ((100.0 - h) / 5.0)
    # Saturated vapor pressure (Tetens formula approx) and VPD
    es = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
    ea = es * (h / 100.0)
    vpd = np.maximum(0.0, es - ea)
    
    feats = np.column_stack([ndvi, ndre, t, h, rain, ratio, dew_point, vpd])
    return feats


class AgronomicFeaturePipeline:
    """
    Preprocessing pipeline for quantum and classical agronomic disease classification.
    1. Extracts raw features: NDVI, NDRE, Temperature, Humidity, Rainfall.
    2. Applies StandardScaler (zero mean, unit variance).
    3. Applies PCA to n_components (matching n_qubits).
    4. Applies MinMaxScaler to [-pi/2, pi/2] to ensure quantum feature map rotation angles
       remain within optimal non-aliasing phase bounds.
    """
    def __init__(self, n_components: int = 4, scale_for_quantum: bool = True):
        self.n_components = n_components
        self.scale_for_quantum = scale_for_quantum
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
        self.is_fitted = False
        self.explained_variance_ratio_: List[float] = []

    def fit(self, X: np.ndarray) -> "AgronomicFeaturePipeline":
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        if self.scale_for_quantum:
            self.angle_scaler.fit(X_pca)
        self.explained_variance_ratio_ = [float(v) for v in self.pca.explained_variance_ratio_]
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        if self.scale_for_quantum:
            return self.angle_scaler.transform(X_pca)
        return X_pca

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def prepare_benchmark_data(
    df: pd.DataFrame,
    n_samples: int = 120,
    n_components: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, AgronomicFeaturePipeline]:
    """
    Prepare a stratified subsample of the dataset suitable for rapid, rigorous
    quantum simulator benchmarking and classical comparison.
    
    Args:
        df: Input DataFrame with sensor columns and 'stress_label'.
        n_samples: Sample size (default 120, balanced across classes).
        n_components: Number of PCA components / qubits (default 4).
        random_state: Seed for reproducibility.
        
    Returns:
        X: Processed features array (n_samples, n_components).
        y: Binary labels array (n_samples,).
        pipeline: Fitted AgronomicFeaturePipeline.
    """
    # Filter rows with complete data
    # Filter rows with complete data
    clean_df = df.dropna(subset=BASE_COLUMNS + ["stress_label"]).copy()
    
    # Stratified balance subsampling
    pos = clean_df[clean_df["stress_label"] == 1]
    neg = clean_df[clean_df["stress_label"] == 0]
    
    half_samples = n_samples // 2
    pos_sample = pos.sample(n=min(half_samples, len(pos)), random_state=random_state)
    neg_sample = neg.sample(n=min(half_samples, len(neg)), random_state=random_state)
    
    sampled_df = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    X_raw = extract_agronomic_features(sampled_df)
    y = sampled_df["stress_label"].to_numpy(dtype=np.int32)
    
    pipeline = AgronomicFeaturePipeline(n_components=n_components, scale_for_quantum=True)
    X = pipeline.fit_transform(X_raw)
    
    return X, y, pipeline
