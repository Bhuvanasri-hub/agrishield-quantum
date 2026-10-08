"""
AgriShield Quantum - Quantum Models (Phase 2)
Implements:
1. QSVC (Quantum Support Vector Classifier) with ZZFeatureMap and FidelityQuantumKernel.
2. VQC (Variational Quantum Classifier) with ZZFeatureMap + RealAmplitudes + COBYLA.
3. Quantum circuit extraction & metadata utilities for visual inspection and export.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC


def build_zz_feature_map(n_qubits: int = 4, reps: int = 1) -> QuantumCircuit:
    """Build ZZFeatureMap circuit for quantum data encoding."""
    return zz_feature_map(n_qubits, reps=reps, entanglement="linear")


def build_real_amplitudes_ansatz(n_qubits: int = 4, reps: int = 1) -> QuantumCircuit:
    """Build RealAmplitudes parameterized ansatz circuit."""
    return real_amplitudes(n_qubits, reps=reps, entanglement="linear")


def get_quantum_circuits(n_qubits: int = 4, reps: int = 1) -> Dict[str, Any]:
    """
    Return feature map and ansatz circuits along with structural quantum metrics
    (qubit count, gate depth, total operations, parameter count).
    """
    fmap = build_zz_feature_map(n_qubits, reps=reps)
    ansatz = build_real_amplitudes_ansatz(n_qubits, reps=reps)
    
    # Combined VQC circuit for visual overview
    vqc_circuit = QuantumCircuit(n_qubits)
    vqc_circuit.compose(fmap, inplace=True)
    vqc_circuit.compose(ansatz, inplace=True)
    
    return {
        "n_qubits": n_qubits,
        "reps": reps,
        "feature_map": fmap,
        "ansatz": ansatz,
        "combined_circuit": vqc_circuit,
        "feature_map_depth": fmap.depth(),
        "ansatz_depth": ansatz.depth(),
        "total_depth": vqc_circuit.depth(),
        "feature_map_ops": dict(fmap.count_ops()),
        "ansatz_ops": dict(ansatz.count_ops()),
        "num_parameters": ansatz.num_parameters
    }


def build_qsvc_model(n_qubits: int = 4, reps: int = 1, C: float = 1.0) -> QSVC:
    """
    Instantiate QSVC with ZZFeatureMap and ComputeUncompute FidelityQuantumKernel.
    Aer StatevectorSampler evaluates high-fidelity overlaps.
    """
    fmap = build_zz_feature_map(n_qubits=n_qubits, reps=reps)
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=fmap)
    model = QSVC(quantum_kernel=kernel, C=C)
    return model


def build_vqc_model(n_qubits: int = 4, reps: int = 1, maxiter: int = 60) -> VQC:
    """
    Instantiate VQC with ZZFeatureMap, RealAmplitudes ansatz, COBYLA optimizer,
    and StatevectorSampler.
    """
    fmap = build_zz_feature_map(n_qubits=n_qubits, reps=reps)
    ansatz = build_real_amplitudes_ansatz(n_qubits=n_qubits, reps=reps)
    optimizer = COBYLA(maxiter=maxiter)
    sampler = StatevectorSampler()
    vqc = VQC(
        feature_map=fmap,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )
    return vqc


def predict_qsvc_proba(model: QSVC, X: np.ndarray) -> np.ndarray:
    """
    Calculate calibrated stress probability [0, 1] from QSVC decision values
    using logistic sigmoid mapping.
    """
    decision = model.decision_function(X)
    # Sigmoid calibration: P(y=1) = 1 / (1 + exp(-decision))
    prob_pos = 1.0 / (1.0 + np.exp(-decision))
    prob_neg = 1.0 - prob_pos
    return np.vstack([prob_neg, prob_pos]).T
