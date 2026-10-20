# vedere fit-monte-carlo.ipynb
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numba import njit
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    converged: bool  # esplicito: True o False
    c_norm: float
    s_norm: float
    c_mhz: float
    s_mhz: float
    chi2_red: float
    duration_sec: float
    popt: Optional[np.ndarray] = None  # tutti i parametri ottimizzati (se servono)


@njit
def due_lorentziane(f, A0, A1, x1, w1, A2, x2, w2):
    """
    Funzione che calcola la somma di due lorentziane.
    f:  array di frequenze
    A0: ampiezza di offset
    A1: ampiezza della prima lorentziana
    x1: centro della prima lorentziana
    w1: semi-larghezza della prima lorentziana (HWHM)
    A2: ampiezza della seconda lorentziana
    x2: centro della seconda lorentziana
    w2: semi-larghezza della seconda lorentziana (HWHM)
    """
    
    lorentziana1 = A1 * (w1**2 / ((f - x1) ** 2 + w1**2))
    lorentziana2 = A2 * (w2**2 / ((f - x2) ** 2 + w2**2))
    
    return A0 - lorentziana1 - lorentziana2


@njit
def random_p0_norm(A0_est, dip_depth_est):
    """
    Genera parametri casuali p0 coerenti con lo spazio [0, 1] e Z-score.
    """
    A0 = np.random.uniform(A0_est - 0.5, A0_est + 0.5)
    A1 = np.random.uniform(0.2, dip_depth_est)
    x1 = np.random.uniform(0.30, 0.55)   # dip sinistro in coordinate [0, 1]
    w1 = np.random.uniform(0.01, 0.05)   # HWHM tipica: 2 - 10 MHz su 200 MHz
    A2 = np.random.uniform(0.2, dip_depth_est)
    x2 = np.random.uniform(0.55, 0.80)   # dip destro in coordinate [0, 1]
    w2 = np.random.uniform(0.01, 0.05)
    return np.array([A0, A1, x1, w1, A2, x2, w2], dtype=np.float64)


class FitMonteCarlo:
    """
    # FitMonteCarlo
    Classe per eseguire il fit Monte Carlo di due lorentziane.
    
    `M`: numero di tentativi di fit Monte Carlo
    """
    
    def __init__(self, M: int = 20):
        """
        Inizializza la classe FitMonteCarlo.
        `M`: numero di tentativi di fit Monte Carlo
        """
        self.M = M
        
    
    def fit_single(
        self, 
        f_original: np.ndarray,
        y_original: np.ndarray
    ) -> FitResult:
        """
        Esegue il fit multi-start Monte Carlo su un singolo spettro rumoroso.
        Inserire le frequenze (f_original) e i valori di intensità (y_original) dello spettro; la funzione si occuperà di normalizzare le frequenze e di calcolare lo z-score dei valori di intensità.
        Restituisce: oggetto FitResult
        """
        t_start_fit = time.perf_counter()
        N_points = len(f_original)
        n_params = 7

        # Normalizzazione frequenze [0, 1]
        f_min, f_max = float(f_original.min()), float(f_original.max())
        delta_f = f_max - f_min
        f_norm = (f_original - f_min) / delta_f

        # Standardizzazione Z-score del segnale in ampiezza
        y_mean = np.mean(y_original)
        y_std = np.std(y_original)
        if y_std < 1e-8:
            y_std = 1e-8
        y_zscore = (y_original - y_mean) / y_std

        A0_est = float(np.percentile(y_zscore, 85))
        dip_depth_est = float(max(A0_est - np.min(y_zscore), 0.5))

        # Bounds nello spazio [0, 1] e Z-score
        lower = [A0_est - 2.5, 0.05, 0.0, 0.005, 0.05, 0.0, 0.005]
        upper = [A0_est + 2.5, dip_depth_est * 2.5, 1.0, 0.100, dip_depth_est * 2.5, 1.0, 0.100]

        best_popt = None
        best_chi2 = np.inf

        for _ in range(self.M):
            p0 = random_p0_norm(A0_est, dip_depth_est)

            try:
                popt, _ = curve_fit(
                    due_lorentziane,
                    f_norm,
                    y_zscore,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=2500
                )
                residuals = y_zscore - due_lorentziane(f_norm, *popt)
                chi2 = np.sum(residuals**2)

                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_popt = popt
            except (RuntimeError, ValueError):
                continue

        duration = time.perf_counter() - t_start_fit

        if best_popt is None:
            return FitResult(
                converged=False,
                c_norm=np.nan,
                s_norm=np.nan,
                c_mhz=np.nan,
                s_mhz=np.nan,
                chi2_red=np.inf,
                duration_sec=duration,
                popt=None
            )

        # Ordinamento dei dip (x1 < x2)
        x1_norm, x2_norm = sorted([best_popt[2], best_popt[5]])

        # Coordinate normalizzate in [0, 1] (per confronto diretto con la 1D-CNN)
        c_norm = (x1_norm + x2_norm) / 2.0
        s_norm = x2_norm - x1_norm

        # Coordinate fisiche riconvertite (in MHz, assumendo f_original in GHz)
        x1_phys = (f_min + x1_norm * delta_f) * 1e3
        x2_phys = (f_min + x2_norm * delta_f) * 1e3
        c_mhz = (x1_phys + x2_phys) / 2.0
        s_mhz = x2_phys - x1_phys

        chi2_red = best_chi2 / (N_points - n_params)
        
        return FitResult(
            converged=True,
            c_norm=c_norm,
            s_norm=s_norm,
            c_mhz=c_mhz,
            s_mhz=s_mhz,
            chi2_red=chi2_red,
            duration_sec=duration,
            popt=best_popt
        )