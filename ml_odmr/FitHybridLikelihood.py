import time
import numpy as np
from numba import njit
from iminuit import Minuit


@njit(fastmath=True)
def _poisson_nll_numba(f_norm, y_counts, A0, A1, x1, w1, A2, x2, w2):
    """
    Negative Log-Likelihood poissoniana (a meno di costanti indipendenti dai parametri):
        NLL = sum_i [ lambda_i - y_i * ln(lambda_i) ]
    """
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


class PoissonNLLCost:
    """Callable riutilizzabile: evita di ricreare closure e oggetti Minuit per ogni spettro."""
    def __init__(self, f_norm: np.ndarray):
        self.f_norm = np.ascontiguousarray(f_norm, dtype=np.float64)
        self.y_counts = np.zeros_like(self.f_norm)

    def __call__(self, A0, A1, x1, w1, A2, x2, w2):
        return _poisson_nll_numba(self.f_norm, self.y_counts, A0, A1, x1, w1, A2, x2, w2)


def worker_chunk_hybrid_nll(args) -> list:
    """
    Esegue il fit ibrido (CNN -> iminuit Poisson NLL) su un blocco di spettri rumorosi
    riutilizzando un'unica istanza C++ di Minuit per l'intero chunk.
    """
    (
        indices,
        noisy_chunk,
        nu1_chunk_mhz,
        nu2_chunk_mhz,
        c_true_chunk,
        s_true_chunk,
        f_norm,
        f_min_ghz,
        delta_f_ghz,
        w0_norm,
        pos_window_norm,
        fallback_to_cnn,
    ) = args

    f_min_mhz = f_min_ghz * 1e3
    delta_f_mhz = delta_f_ghz * 1e3
    n_pts = len(f_norm)

    cost = PoissonNLLCost(f_norm)

    # Un solo oggetto Minuit per tutto il chunk
    m = Minuit(cost, A0=1000.0, A1=100.0, x1=0.40, w1=w0_norm, A2=100.0, x2=0.60, w2=w0_norm)
    m.errordef = Minuit.LIKELIHOOD
    m.limits["w1"] = (0.005, 0.100)
    m.limits["w2"] = (0.005, 0.100)

    chunk_results = []

    for k in range(len(indices)):
        idx = int(indices[k])
        t0 = time.perf_counter()

        y_counts = np.ascontiguousarray(noisy_chunk[k], dtype=np.float64)
        cost.y_counts = y_counts

        # 1. Conversione predizioni CNN da MHz a spazio normalizzato [0, 1]
        x1_0 = (float(nu1_chunk_mhz[k]) - f_min_mhz) / delta_f_mhz
        x2_0 = (float(nu2_chunk_mhz[k]) - f_min_mhz) / delta_f_mhz
        x1_0, x2_0 = sorted([x1_0, x2_0])
        x1_0 = float(np.clip(x1_0, 1e-4, 1.0 - 1e-4))
        x2_0 = float(np.clip(x2_0, 1e-4, 1.0 - 1e-4))

        # 2. Stima robusta di fondo e ampiezze (media su ±2 bin attorno ai dip predetti)
        A0_est = float(np.percentile(y_counts, 85))
        if A0_est < 1.0:
            A0_est = 1.0
        dip_depth_est = float(max(A0_est - np.min(y_counts), 1.0))

        i1 = int(np.clip(round(x1_0 * (n_pts - 1)), 0, n_pts - 1))
        i2 = int(np.clip(round(x2_0 * (n_pts - 1)), 0, n_pts - 1))
        y1_loc = float(np.mean(y_counts[max(0, i1 - 2) : min(n_pts, i1 + 3)]))
        y2_loc = float(np.mean(y_counts[max(0, i2 - 2) : min(n_pts, i2 + 3)]))

        amp_min = min(0.1 * dip_depth_est, 0.5 * A0_est)
        amp_max = 0.9 * A0_est
        A1_0 = float(np.clip(A0_est - y1_loc, amp_min, amp_max))
        A2_0 = float(np.clip(A0_est - y2_loc, amp_min, amp_max))

        # 3. Limiti adattivi (finestra locale attorno alla CNN per prevenire outlier a basso SNR)
        m.limits["A0"] = (0.5 * A0_est, 1.5 * A0_est)
        m.limits["A1"] = (0.0, A0_est)
        m.limits["A2"] = (0.0, A0_est)

        if pos_window_norm is not None:
            m.limits["x1"] = (max(0.0, x1_0 - pos_window_norm), min(1.0, x1_0 + pos_window_norm))
            m.limits["x2"] = (max(0.0, x2_0 - pos_window_norm), min(1.0, x2_0 + pos_window_norm))
        else:
            m.limits["x1"] = (0.0, 1.0)
            m.limits["x2"] = (0.0, 1.0)

        # 4. Inizializzazione deterministica da CNN e singola minimizzazione MIGRAD
        m.values = [A0_est, A1_0, x1_0, w0_norm, A2_0, x2_0, w0_norm]
        m.migrad(ncall=2500)

        duration = time.perf_counter() - t0
        c_true = float(c_true_chunk[k])
        s_true = float(s_true_chunk[k])
        minuit_valid = bool(m.valid)

        if minuit_valid:
            x1_norm, x2_norm = sorted([float(m.values["x1"]), float(m.values["x2"])])
            nll_min = float(m.fval)
            converged = True
            used_fallback = False
        elif fallback_to_cnn:
            # Se Minuit non converge, mantiene la predizione della CNN
            x1_norm, x2_norm = x1_0, x2_0
            nll_min = np.nan
            converged = True
            used_fallback = True
        else:
            chunk_results.append({
                "spectrum_id": idx,
                "converged": False,
                "minuit_valid": False,
                "used_fallback": False,
                "c_true_norm": c_true,
                "s_true_norm": s_true,
                "c_fit_norm": np.nan,
                "s_fit_norm": np.nan,
                "err_c_norm": np.nan,
                "err_s_norm": np.nan,
                "c_fit_mhz": np.nan,
                "s_fit_mhz": np.nan,
                "nll_min": np.inf,
                "fit_time_sec": duration,
            })
            continue

        c_norm = (x1_norm + x2_norm) / 2.0
        s_norm = x2_norm - x1_norm
        x1_phys = f_min_mhz + x1_norm * delta_f_mhz
        x2_phys = f_min_mhz + x2_norm * delta_f_mhz

        chunk_results.append({
            "spectrum_id": idx,
            "converged": converged,
            "minuit_valid": minuit_valid,
            "used_fallback": used_fallback,
            "c_true_norm": c_true,
            "s_true_norm": s_true,
            "c_fit_norm": c_norm,
            "s_fit_norm": s_norm,
            "err_c_norm": abs(c_norm - c_true),
            "err_s_norm": abs(s_norm - s_true),
            "c_fit_mhz": (x1_phys + x2_phys) / 2.0,
            "s_fit_mhz": x2_phys - x1_phys,
            "nll_min": nll_min,
            "fit_time_sec": duration,
        })

    return chunk_results