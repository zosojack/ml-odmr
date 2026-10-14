# vedere fit-monte-carlo.ipynb
import numpy as np
from numba import njit
from scipy.optimize import curve_fit

@njit
def due_lorentziane(f, A0, A1, x1, w1, A2, x2, w2):
    """
    Funzione che calcola la somma di due lorentziane.
    f:  array di frequenze
    A0: ampiezza di offset
    A1: ampiezza della prima lorentziana
    x1: centro della prima lorentziana
    w1: larghezza della prima lorentziana
    A2: ampiezza della seconda lorentziana
    x2: centro della seconda lorentziana
    w2: larghezza della seconda lorentziana
    """
    
    def lorentziana(f, A, x, w):
        return A * (w**2 / ((f - x) ** 2 + w**2))
    
    lorentziana1 = lorentziana(f, A1, x1, w1)
    lorentziana2 = lorentziana(f, A2, x2, w2)
    
    return A0 - lorentziana1 - lorentziana2


@njit
def random_p0(A0_est, max_depth=0.15):
    """
    Genera un vettore di parametri iniziali casuali per la funzione due_lorentziane.
    Restituisce un array con i parametri [A0, A1, x1, w1, A2, x2, w2].
    """
    A0 = np.random.uniform(A0_est - 1, A0_est + 1)  # Ampiezza di offset
    A1 = np.random.uniform(0., 2*max_depth)   # Ampiezza della prima lorentziana
    x1 = np.random.uniform(2.81, 2.86)        # Centro della prima lorentziana
    w1 = np.random.uniform(0., 1.)            # Larghezza della prima lorentziana
    A2 = np.random.uniform(0., 2*max_depth)   # Ampiezza della seconda lorentziana
    x2 = np.random.uniform(2.86, 2.91)        # Centro della seconda lorentziana
    w2 = np.random.uniform(0., 1)             # Larghezza della seconda lorentziana
    
    return np.array([A0, A1, x1, w1, A2, x2, w2], dtype=np.float32)


def fit_odmr_monte_carlo(f_ghz, y_zscore, M=20):
    """
    Esegue il fit multi-start Monte Carlo su un singolo spettro rumoroso.
    Restituisce: (c_fit, s_fit, popt, chi2_red)
    """
    f_min, f_max = float(f_ghz.min()), float(f_ghz.max())
    delta_f = f_max - f_min
    N_points = len(f_ghz)
    n_params = 7

    # Stime preliminari per circoscrivere lo spazio di ricerca
    A0_est = np.percentile(y_zscore, 85)
    dip_depth_est = max(A0_est - np.min(y_zscore), 0.5)

    # Limiti fisici (bounds):
    # w: HWHM tra 1.5 MHz e 15 MHz (0.0015 - 0.015 GHz)
    lower = [A0_est - 2.5, 0.05, f_min, 0.0015, 0.05, f_min, 0.0015]
    upper = [A0_est + 2.5, dip_depth_est * 2.5, f_max, 0.0150, dip_depth_est * 2.5, f_max, 0.0150]

    best_popt = None
    best_chi2 = np.inf

    for _ in range(M):
        # Campionamento Monte Carlo dei parametri di partenza (guess p0)
        p0 = random_p0(A0_est, max_depth=dip_depth_est)

        try:
            popt, _ = curve_fit(
                due_lorentziane,
                f_ghz,
                y_zscore,
                p0=p0,
                bounds=(lower, upper),
                maxfev=3000
            )
            residuals = y_zscore - due_lorentziane(f_ghz, *popt)
            chi2 = np.sum(residuals**2)

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_popt = popt
        except (RuntimeError, ValueError):
            continue

    if best_popt is None:
        return np.nan, np.nan, None, np.inf

    # Ordinamento dei dip (x1 < x2)
    x1_fit, x2_fit = best_popt[2], best_popt[5]
    if x1_fit > x2_fit:
        x1_fit, x2_fit = x2_fit, x1_fit

    # Proiezione delle coordinate stimate nello spazio adimensionale [0, 1]
    x1_norm = (x1_fit - f_min) / delta_f
    x2_norm = (x2_fit - f_min) / delta_f
    c_fit = (x1_norm + x2_norm) / 2.0
    s_fit = x2_norm - x1_norm

    chi2_red = best_chi2 / (N_points - n_params)
    return c_fit, s_fit, best_popt, chi2_red