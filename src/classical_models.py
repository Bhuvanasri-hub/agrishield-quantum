"""
AgriShield Quantum - Classical Baselines (Phase 2)
Implements classical machine learning baselines:
1. Support Vector Classifier (RBF Kernel)
2. Random Forest Classifier
3. Logistic Regression
"""

from __future__ import annotations
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.calibration import CalibratedClassifierCV


def build_svc_rbf(C: float = 1.0, gamma: str = "scale", random_state: int = 42):
    """Build Support Vector Classifier with RBF kernel and calibrated probability estimation."""
    base_svc = SVC(C=C, kernel="rbf", gamma=gamma, random_state=random_state)
    return CalibratedClassifierCV(estimator=base_svc, ensemble=False)


def build_random_forest(n_estimators: int = 100, max_depth: int = 6, random_state: int = 42) -> RandomForestClassifier:
    """Build Random Forest Classifier baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def build_logistic_regression(C: float = 1.0, random_state: int = 42) -> LogisticRegression:
    """Build Logistic Regression baseline."""
    return LogisticRegression(C=C, max_iter=500, random_state=random_state)


def get_all_classical_models() -> Dict[str, Any]:
    """Return dictionary of configured classical baseline models."""
    return {
        "SVC (RBF)": build_svc_rbf(),
        "Random Forest": build_random_forest(),
        "Logistic Regression": build_logistic_regression()
    }
