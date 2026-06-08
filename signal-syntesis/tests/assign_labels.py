from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import curve_fit

@njit
def lorentziana(
    f: np.ndarray,
    A: float,
    f0: float,
    gamma: float
) -> np.ndarray:
    """
    Calcola la funzione di Lorentziana per un dato array di frequenze f,
    con centro f0 e larghezza gamma.
    """
    return A * gamma / ((f - f0)**2 + gamma**2)

@njit
def multi_lorentziana(
    f: np.ndarray,
    ampiezze: np.ndarray,
    posizioni: np.ndarray,
    gamma: np.ndarray
) -> np.ndarray:
    """
    Calcola la somma di più funzioni di Lorentziana per un array di frequenze f,
    dato un array di posizioni dei picchi e un array di larghezze gamma.
    """
    spettro = np.zeros_like(f, np.float64)
    for i in range(len(posizioni)):
        spettro += lorentziana(f, ampiezze[i], posizioni[i], gamma[i])
    return spettro

@njit
def otto_lorentziane(
    f: np.ndarray,
    ampiezze: np.ndarray,
    posizioni: np.ndarray,
    gamma: np.ndarray
) -> np.ndarray:
    """
    Calcola la somma di otto funzioni di Lorentziana per un array di frequenze f,
    dato un array di posizioni dei picchi e un array di larghezze gamma.
    """
    # I parametri possono non essere lunghi otto, l'importante è che siano pari
    # vengono duplicati; utile se ci sono più lorentziane sovrapposte
    
    spettro = np.zeros_like(f, np.float64)
    for i in range(8):
        spettro += lorentziana(f, ampiezze[i], posizioni[i], gamma[i])
    return spettro

def fit_otto_lorentziane(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue il fitting dello spettro utilizzando la funzione otto_lorentziane.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """
    N = len(posizioni_iniziali)
    if N < 8:
        if N % 2 != 0:
            raise ValueError("Il numero di posizioni deve essere pari per poter duplicare i parametri.")
        ampiezze_iniziali = np.tile(ampiezze_iniziali, 8 // N)
        posizioni_iniziali = np.tile(posizioni_iniziali, 8 // N)
        gamma_iniziali = np.tile(gamma_iniziali, 8 // N)
    
    return fit_spettro(f, spettro, ampiezze_iniziali, posizioni_iniziali, gamma_iniziali, use_otto=True)


def fit_spettro(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray,
    use_otto: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue il fitting dello spettro utilizzando la funzione multi_lorentziana.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """
    
    # Definiamo la funzione da passare a curve_fit
    def modello(f, *parametri):
        ampiezze = parametri[:len(ampiezze_iniziali)]
        posizioni = parametri[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
        gamma = parametri[len(ampiezze_iniziali) + len(posizioni_iniziali):]
        if use_otto:
            return -otto_lorentziane(f, ampiezze, posizioni, gamma)
        else:
            return -multi_lorentziana(f, ampiezze, posizioni, gamma)
    
    # Parametri iniziali per il fitting
    parametri_iniziali = np.concatenate((ampiezze_iniziali, posizioni_iniziali, gamma_iniziali))
    
    # Definiamo i limiti per il fitting
    low = np.concatenate((ampiezze_iniziali * 0.1, posizioni_iniziali - 20., gamma_iniziali * 0.1))  # Limiti inferiori
    upp = np.concatenate((ampiezze_iniziali * 10., posizioni_iniziali + 20., gamma_iniziali * 5.))  # Limiti superiori
    
    # Eseguiamo il fitting
    parametri_ottimizzati, _ = curve_fit(modello, f, spettro, p0=parametri_iniziali, bounds=(low, upp))
    
    # Estraiamo le ampiezze, le posizioni e le larghezze ottimizzate
    ampiezze_ottimizzate = parametri_ottimizzati[:len(ampiezze_iniziali)]
    posizioni_ottimizzate = parametri_ottimizzati[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
    gamma_ottimizzati = parametri_ottimizzati[len(ampiezze_iniziali) + len(posizioni_iniziali):]
    
    return ampiezze_ottimizzate, posizioni_ottimizzate, gamma_ottimizzati


def correggi_fit_spettro(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_pre: np.ndarray,
    posizioni_pre: np.ndarray,
    gamma_pre: np.ndarray,
    ampiezze_aggiunte: np.ndarray,
    posizioni_aggiunte: np.ndarray,
    gamma_aggiunte: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue un secondo fit aggiungendo nuovi picchi al modello precedente.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """
    
    N_pre = len(ampiezze_pre)
    N_aggiunte = len(ampiezze_aggiunte)
    
    ampiezze_iniziali = np.concatenate((ampiezze_pre, ampiezze_aggiunte))
    posizioni_iniziali = np.concatenate((posizioni_pre, posizioni_aggiunte))
    gamma_iniziali = np.concatenate((gamma_pre, gamma_aggiunte))
    
    # Definiamo la funzione da passare a curve_fit
    def modello(f, *parametri):
        ampiezze = parametri[:len(ampiezze_iniziali)]
        posizioni = parametri[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
        gamma = parametri[len(ampiezze_iniziali) + len(posizioni_iniziali):]
        return -multi_lorentziana(f, ampiezze, posizioni, gamma)
    
    # Definiamo i parametri iniziali per il fitting
    parametri_iniziali = np.concatenate((ampiezze_iniziali, posizioni_iniziali, gamma_iniziali))
                                         
    # Definiamo i limiti per il fitting                          
    amp_low = np.concatenate((ampiezze_pre * 0.75, ampiezze_aggiunte * 0.1))
    amp_upp = np.concatenate((ampiezze_pre * 1.5, ampiezze_aggiunte * 5.))
    f0_low = np.concatenate((posizioni_pre - 1., posizioni_aggiunte - 5.))
    f0_upp = np.concatenate((posizioni_pre + 1., posizioni_aggiunte + 5.))
    gamma_low = np.concatenate((gamma_pre * 0.75, gamma_aggiunte * 0.1))
    gamma_upp = np.concatenate((gamma_pre * 1.5, gamma_aggiunte * 5.))
    
    low = np.concatenate((amp_low, f0_low, gamma_low))
    upp = np.concatenate((amp_upp, f0_upp, gamma_upp))
    
    # Eseguiamo il fitting
    parametri_ottimizzati, _ = curve_fit(modello, f, spettro, p0=parametri_iniziali, bounds=(low, upp))
    
    # Estraiamo le ampiezze, le posizioni e le larghezze ottimizzate
    ampiezze_ottimizzate = parametri_ottimizzati[:len(ampiezze_iniziali)]
    posizioni_ottimizzate = parametri_ottimizzati[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
    gamma_ottimizzati = parametri_ottimizzati[len(ampiezze_iniziali) + len(posizioni_iniziali):]

    return ampiezze_ottimizzate, posizioni_ottimizzate, gamma_ottimizzati
