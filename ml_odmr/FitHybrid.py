import time
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

from ml_odmr.functions import due_lorentziane
from ml_odmr.FitResult import FitResult


class FitHybrid:
    """
    # FitHybrid
    Classe per eseguire il fit di due lorentziane con un approccio ibrido,
    partendo dalle predizioni di un modello di machine learning (1D-CNN).
    """

    def __init__(self, cnn_predictor=None, w0_norm: float = 0.025):
        """
        Inizializza la classe FitHybrid.
        
        `cnn_predictor`: istanza opzionale di CNNPredictor (usata se non si passano
                         esplicitamente nu1_mhz e nu2_mhz a fit_single).
        `w0_norm`: guess iniziale per la semi-larghezza HWHM normalizzata in [0, 1]
                   (default 0.025 -> 5 MHz su una finestra di 200 MHz).
        """
        self.cnn = cnn_predictor
        self.w0_norm = w0_norm

    def fit_single(
        self,
        f_original: np.ndarray,
        y_original: np.ndarray,
        nu1_mhz: Optional[float] = None,
        nu2_mhz: Optional[float] = None,
    ) -> FitResult:
        """
        Esegue un singolo fit deterministico guidato dalla CNN su uno spettro rumoroso.
        
        Parametri:
        - `f_original`: array delle frequenze (in GHz, es. 2.75 - 2.95)
        - `y_original`: array dei conteggi di fluorescenza dello spettro
        - `nu1_mhz`, `nu2_mhz`: frequenze dei dip in MHz predette dalla CNN.
          Se non fornite (None), vengono calcolate al volo tramite `self.cnn`.
          
        Restituisce: oggetto FitResult
        """
        t_start_fit = time.perf_counter()
        N_points = len(f_original)
        n_params = 7

        # 1. Se nu1_mhz e nu2_mhz non sono passati, esegue l'inferenza CNN al volo
        if nu1_mhz is None or nu2_mhz is None:
            if self.cnn is None:
                raise ValueError(
                    "Devi fornire 'nu1_mhz' e 'nu2_mhz' oppure passare un 'cnn_predictor' al costruttore."
                )
            # La CNN si aspetta shape (1, N_points) e frequenze in MHz
            pred = self.cnn.predict_counts(
                np.atleast_2d(y_original), f_original * 1e3
            )
            nu1_mhz = float(np.squeeze(pred["nu1_MHz"]))
            nu2_mhz = float(np.squeeze(pred["nu2_MHz"]))

        # 2. Normalizzazione frequenze in [0, 1]
        f_min, f_max = float(f_original.min()), float(f_original.max())
        delta_f = f_max - f_min
        f_norm = (f_original - f_min) / delta_f

        # Conversione delle guess di frequenza da MHz allo spazio normalizzato [0, 1]
        f_min_mhz = f_min * 1e3
        delta_f_mhz = delta_f * 1e3
        x1_0 = (nu1_mhz - f_min_mhz) / delta_f_mhz
        x2_0 = (nu2_mhz - f_min_mhz) / delta_f_mhz

        # Ordiniamo e vincoliamo le guess strettamente dentro i bounds (0, 1)
        x1_0, x2_0 = sorted([x1_0, x2_0])
        x1_0 = float(np.clip(x1_0, 1e-4, 1.0 - 1e-4))
        x2_0 = float(np.clip(x2_0, 1e-4, 1.0 - 1e-4))

        # 3. Standardizzazione Z-score del segnale in ampiezza
        y_mean = np.mean(y_original)
        y_std = np.std(y_original)
        if y_std < 1e-8:
            y_std = 1e-8
        y_zscore = (y_original - y_mean) / y_std

        A0_est = float(np.percentile(y_zscore, 85))
        dip_depth_est = float(max(A0_est - np.min(y_zscore), 0.5))

        # Stima locale della profondità dei due dip nei punti predetti dalla CNN
        idx1 = int(np.clip(round(x1_0 * (N_points - 1)), 0, N_points - 1))
        idx2 = int(np.clip(round(x2_0 * (N_points - 1)), 0, N_points - 1))
        A1_0 = float(np.clip(A0_est - y_zscore[idx1], 0.2, dip_depth_est * 2.0))
        A2_0 = float(np.clip(A0_est - y_zscore[idx2], 0.2, dip_depth_est * 2.0))

        # Vettore di inizializzazione deterministico p0 guidato dalla CNN
        p0 = np.array(
            [A0_est, A1_0, x1_0, self.w0_norm, A2_0, x2_0, self.w0_norm],
            dtype=np.float64,
        )

        # Bounds nello spazio [0, 1] e Z-score (identici a FitMonteCarlo)
        lower = [A0_est - 2.5, 0.05, 0.0, 0.005, 0.05, 0.0, 0.005]
        upper = [
            A0_est + 2.5,
            dip_depth_est * 2.5,
            1.0,
            0.100,
            dip_depth_est * 2.5,
            1.0,
            0.100,
        ]

        # 4. Singola esecuzione di curve_fit
        try:
            popt, _ = curve_fit(
                due_lorentziane,
                f_norm,
                y_zscore,
                p0=p0,
                bounds=(lower, upper),
                maxfev=2500,
            )
            residuals = y_zscore - due_lorentziane(f_norm, *popt)
            chi2 = float(np.sum(residuals**2))
        except (RuntimeError, ValueError):
            duration = time.perf_counter() - t_start_fit
            return FitResult(
                converged=False,
                c_norm=np.nan,
                s_norm=np.nan,
                c_mhz=np.nan,
                s_mhz=np.nan,
                chi2_red=np.inf,
                duration_sec=duration,
                popt=None,
            )

        duration = time.perf_counter() - t_start_fit

        # Ordinamento dei dip (x1 < x2)
        x1_norm, x2_norm = sorted([popt[2], popt[5]])

        # Coordinate normalizzate in [0, 1]
        c_norm = (x1_norm + x2_norm) / 2.0
        s_norm = x2_norm - x1_norm

        # Coordinate fisiche riconvertite (in MHz, assumendo f_original in GHz)
        x1_phys = (f_min + x1_norm * delta_f) * 1e3
        x2_phys = (f_min + x2_norm * delta_f) * 1e3
        c_mhz = (x1_phys + x2_phys) / 2.0
        s_mhz = x2_phys - x1_phys

        chi2_red = chi2 / (N_points - n_params)

        return FitResult(
            converged=True,
            c_norm=c_norm,
            s_norm=s_norm,
            c_mhz=c_mhz,
            s_mhz=s_mhz,
            chi2_red=chi2_red,
            duration_sec=duration,
            popt=popt,
        )