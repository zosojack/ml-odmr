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
    w1: semi-larghezza della prima lorentziana (HWHM)
    A2: ampiezza della seconda lorentziana
    x2: centro della seconda lorentziana
    w2: semi-larghezza della seconda lorentziana (HWHM)
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
    A1 = np.random.uniform(0., 2*max_depth)   # Ampiezza della lorentziana I
    x1 = np.random.uniform(2.81, 2.86)        # Centro della lorentziana I
    w1 = np.random.uniform(0., 1.)            # Semi-larghezza lorentziana I
    A2 = np.random.uniform(0., 2*max_depth)   # Ampiezza della lorentziana II
    x2 = np.random.uniform(2.86, 2.91)        # Centro della lorentziana II
    w2 = np.random.uniform(0., 1)             # Semi-larghezza della lorentziana II
    
    return np.array([A0, A1, x1, w1, A2, x2, w2], dtype=np.float32)

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
    
    def fit(
        f_original: np.ndarray,
        y_original: np.ndarray
    ) -> tuple:
        # TODO:
        # - aggiungere calcolo degli errori ai parametri stimati
        """
        Esegue il fit multi-start Monte Carlo su un singolo spettro rumoroso.
        Inserire le frequenze (f_original) e i valori di intensità (y_original) dello spettro; la funzione si occuperà di normalizzare le frequenze e di calcolare lo z-score dei valori di intensità.
        Restituisce: (c_fit, s_fit, popt, chi2_red)
        """
        
        # Numero di punti dati e numero di parametri della funzione
        N_points = len(f_original)
        n_params = 7
        
        # Normalizzazione delle frequenze
        f_min, f_max = float(f_original.min()), float(f_original.max())
        delta_f = f_max - f_min
        f_normalized = (f_original - f_min) / delta_f
        
        # z-score dei valori di intensità
        y_mean = np.mean(y_original)
        y_std = np.std(y_original)
        y_zscore = (y_original - y_mean) / y_std

        # Stime preliminari per circoscrivere lo spazio di ricerca
        A0_est = np.percentile(y_zscore, 85)
        dip_depth_est = max(A0_est - np.min(y_zscore), 0.5)

        # Limiti fisici (bounds):
        # - A: minimo 0, massimo 2.5 volte la profondità stimata del dip
        # - f: minimo 0, massimo 0.5 | minimo 0.5, massimo 1 (range normalizzato)
        # - w: minimo 0, massimo 0.5 (metà dell'intero range normalizzato)
        lower = [A0_est - 2.5, 0., 0., 0., 0., 0.5, 0.]
        upper = [A0_est + 2.5, dip_depth_est * 2.5, 0.5, 0.6, dip_depth_est * 2.5, 1., 0.5]

        # Inizializzazione delle variabili per il miglior fit
        best_popt = None
        best_chi2 = np.inf

        for _ in range(M):
            # Campionamento Monte Carlo dei parametri di partenza (guess p0)
            p0 = random_p0(A0_est, max_depth=dip_depth_est)

            try:
                popt, _ = curve_fit(
                    due_lorentziane,
                    f_normalized,
                    y_zscore,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=3000
                )
                residuals = y_zscore - due_lorentziane(f_normalized, *popt)
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

        # Ri-proiezione delle coordinate stimate nello spazio originale
        x1 = (x1_fit * delta_f) + f_min
        x2 = (x2_fit * delta_f) + f_min
        # Calcolo della posizione centrale e della separazione dei dip
        c = (x1 + x2) / 2.0
        s = x2 - x1
        # Chi-quadro ridotto
        chi2_red = best_chi2 / (N_points - n_params)
        
        return c, s, best_popt, chi2_red