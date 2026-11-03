import numpy as np
from dataclasses import dataclass
from typing import Optional

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