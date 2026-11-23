from __future__ import annotations
import time
from typing import Optional
import numpy as np
from numba import njit
from scipy.optimize import curve_fit

from ml_odmr.functions import due_lorentziane
from ml_odmr.FitResult import FitResult

from iminuit import Minuit

# =========================================================================
# 2. DEFINIZIONE DELLA NLL POISSONIANA (NUMBA + CLOSURE)
# =========================================================================

@njit
def _poisson_nll(f_norm, y_counts, A0, A1, x1, w1, A2, x2, w2):
    """
    Calcola la Negative Log-Likelihood poissoniana (a meno di costanti):
        NLL = sum_i [ lambda_i - y_i * ln(lambda_i) ]
    """
    nll = 0.0
    for i in range(len(f_norm)):
        f = f_norm[i]
        l1 = A1 * (w1 * w1 / ((f - x1) * (f - x1) + w1 * w1))
        l2 = A2 * (w2 * w2 / ((f - x2) * (f - x2) + w2 * w2))
        lam = A0 - l1 - l2
        
        # Protezione numerica: lambda deve essere strettamente positivo
        if lam < 1e-9:
            lam = 1e-9
            
        nll += lam - y_counts[i] * np.log(lam)
    return nll

def create_nll(f_norm: np.ndarray, y_counts: np.ndarray):
    """
    Closure che fissa i dati (f_norm, y_counts) e restituisce la funzione
    costo con la firma esplicita dei 7 parametri per iminuit.
    """
    f_arr = np.ascontiguousarray(f_norm, dtype=np.float64)
    y_arr = np.ascontiguousarray(y_counts, dtype=np.float64)
    
    def nll(A0, A1, x1, w1, A2, x2, w2):
        return _poisson_nll_numba(f_arr, y_arr, A0, A1, x1, w1, A2, x2, w2)
        
    return nll

'''
class FitLikelihood:

    def __init__(self, f_norm: np.ndarray, y_counts: np.ndarray):
        self.f_norm = f_norm
        self.y_counts = y_counts
        self.nll = create_nll(f_norm, y_counts)


    @staticmethod
    def create_nll(f_norm: np.ndarray, y_counts: np.ndarray):
        """
        Closure che fissa i dati (f_norm, y_counts) e restituisce la funzione
        costo con la firma esplicita dei 7 parametri per iminuit.
        """
        f_arr = np.ascontiguousarray(f_norm, dtype=np.float64)
        y_arr = np.ascontiguousarray(y_counts, dtype=np.float64)
        
        def nll(A0, A1, x1, w1, A2, x2, w2):
            return _poisson_nll_numba(f_arr, y_arr, A0, A1, x1, w1, A2, x2, w2)
            
        return nll
    
    def fit(self, p0: Optional[np.ndarray] = None) -> FitResult:
'''

import time
import numpy as np
from numba import njit
from iminuit import Minuit
from ml_odmr.PoissonNoise import PoissonNoise


@njit(fastmath=True)
def _poisson_nll_numba(f_norm, y_counts, A0, A1, x1, w1, A2, x2, w2):
    nll = 0.0
    for i in range(len(f_norm)):
        f = f_norm[i]
        l1 = A1 * (w1 * w1 / ((f - x1) * (f - x1) + w1 * w1))
        l2 = A2 * (w2 * w2 / ((f - x2) * (f - x2) + w2 * w2))
        lam = A0 - l1 - l2
        if lam < 1e-9:
            lam = 1e-9
        nll += lam - y_counts[i] * np.log(lam)
    return nll


@njit
def random_p0_poisson(A0_est, dip_depth_est):
    A0 = np.random.uniform(0.95 * A0_est, 1.05 * A0_est)
    A1 = np.random.uniform(0.1 * dip_depth_est, 1.5 * dip_depth_est)
    x1 = np.random.uniform(0.30, 0.55)
    w1 = np.random.uniform(0.01, 0.05)
    A2 = np.random.uniform(0.1 * dip_depth_est, 1.5 * dip_depth_est)
    x2 = np.random.uniform(0.55, 0.80)
    w2 = np.random.uniform(0.01, 0.05)
    return np.array([A0, A1, x1, w1, A2, x2, w2], dtype=np.float64)


class PoissonNLLCost:
    """Callable riutilizzabile: evita di ricreare closure e oggetti Minuit."""
    def __init__(self, f_norm: np.ndarray):
        self.f_norm = np.ascontiguousarray(f_norm, dtype=np.float64)
        self.y_counts = np.zeros_like(self.f_norm)

    def __call__(self, A0, A1, x1, w1, A2, x2, w2):
        return _poisson_nll_numba(self.f_norm, self.y_counts, A0, A1, x1, w1, A2, x2, w2)


def worker_chunk_nll(args):
    (
        indices, spectra_chunk, c_true_chunk, s_true_chunk,
        f_norm, f_min, delta_f, contrast_val, n_ph_val, m_trials, random_seed
    ) = args

    local_noise = PoissonNoise(seed=random_seed)
    cost = PoissonNLLCost(f_norm)

    # Un solo oggetto Minuit per tutto il chunk (zero allocazioni C++ ripetute)
    m = Minuit(cost, A0=1000.0, A1=100.0, x1=0.40, w1=0.025, A2=100.0, x2=0.60, w2=0.025)
    m.errordef = Minuit.LIKELIHOOD
    m.limits["x1"] = (0.0, 1.0)
    m.limits["x2"] = (0.0, 1.0)
    m.limits["w1"] = (0.005, 0.100)
    m.limits["w2"] = (0.005, 0.100)

    chunk_results = []

    for k in range(len(indices)):
        idx = int(indices[k])
        t0 = time.perf_counter()

        ideal_sig = local_noise.ideal_signal(spectra_chunk[k], contrast=contrast_val)
        y_counts = local_noise.add_noise(ideal_sig, n_tot=n_ph_val, seed_modifier=idx)
        cost.y_counts = np.ascontiguousarray(y_counts, dtype=np.float64)

        A0_est = float(np.percentile(y_counts, 85))
        dip_depth_est = float(max(A0_est - np.min(y_counts), 1.0))

        m.limits["A0"] = (0.5 * A0_est, 1.5 * A0_est)
        m.limits["A1"] = (0.0, A0_est)
        m.limits["A2"] = (0.0, A0_est)

        best_nll = np.inf
        best_x1, best_x2 = np.nan, np.nan
        converged = False

        for _ in range(m_trials):
            m.values = random_p0_poisson(A0_est, dip_depth_est)
            m.migrad(ncall=2500)
            if m.valid and m.fval < best_nll:
                best_nll = float(m.fval)
                best_x1 = float(m.values["x1"])
                best_x2 = float(m.values["x2"])
                converged = True

        duration = time.perf_counter() - t0
        c_true = float(c_true_chunk[k])
        s_true = float(s_true_chunk[k])

        if not converged:
            chunk_results.append({
                "spectrum_id": idx, "converged": False,
                "c_true_norm": c_true, "s_true_norm": s_true,
                "c_fit_norm": np.nan, "s_fit_norm": np.nan,
                "err_c_norm": np.nan, "err_s_norm": np.nan,
                "c_fit_mhz": np.nan, "s_fit_mhz": np.nan,
                "nll_min": np.inf, "fit_time_sec": duration,
            })
        else:
            x1_norm, x2_norm = sorted([best_x1, best_x2])
            c_norm = (x1_norm + x2_norm) / 2.0
            s_norm = x2_norm - x1_norm
            x1_phys = (f_min + x1_norm * delta_f) * 1e3
            x2_phys = (f_min + x2_norm * delta_f) * 1e3

            chunk_results.append({
                "spectrum_id": idx, "converged": True,
                "c_true_norm": c_true, "s_true_norm": s_true,
                "c_fit_norm": c_norm, "s_fit_norm": s_norm,
                "err_c_norm": abs(c_norm - c_true), "err_s_norm": abs(s_norm - s_true),
                "c_fit_mhz": (x1_phys + x2_phys) / 2.0,
                "s_fit_mhz": x2_phys - x1_phys,
                "nll_min": best_nll, "fit_time_sec": duration,
            })

    return chunk_results
