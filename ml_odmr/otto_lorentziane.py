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
    gamma: np.ndarray,
    y0: float = 0.
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
    return spettro + y0

def pareggia_parametri(
    posizioni: np.ndarray,
    gamma: np.ndarray,
    ampiezze: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Se il numero di parametri è dispari, cerca di pareggiarlo duplicando i parametri esistenti.
    Il criterio è che viene duplicato il centro senza un altro speculare.
    """
    
    # come capire quale è il centro da duplicare? faccio una media dei centri e vedo quanto dista dal centro (2870)
    centro_medio = np.mean(posizioni)
    distanza_centro = centro_medio - 2870 # questo mi dice da che lato è il centro solitario
    # il centro da aggiungere è verosimilmente speculare alla stessa distanza
    centro_da_aggiungere = 2870 - distanza_centro
    # gamma e ampiezza vengono aggiunte arbitrarie
    gamma_da_aggiungere = np.mean(gamma)
    ampiezza_da_aggiungere = np.mean(ampiezze)
    # aggiungo i parametri
    posizioni = np.append(posizioni, centro_da_aggiungere)
    gamma = np.append(gamma, gamma_da_aggiungere)
    ampiezze = np.append(ampiezze, ampiezza_da_aggiungere)
    
    return posizioni, gamma, ampiezze


def fit_spettro(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Esegue il fitting dello spettro utilizzando la funzione multi_lorentziana.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """    
    
    # Definiamo la funzione da passare a curve_fit
    def modello(f, *parametri):
        ampiezze = parametri[:len(ampiezze_iniziali)]
        posizioni = parametri[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
        gamma = parametri[len(ampiezze_iniziali) + len(posizioni_iniziali):]
        y0 = parametri[-1]  # L'ultimo parametro è l'offset y0

        return -otto_lorentziane(f, ampiezze, posizioni, gamma, y0)  
    
    # Parametri iniziali per il fitting
    parametri_iniziali = np.concatenate((ampiezze_iniziali, posizioni_iniziali, gamma_iniziali, [0.]))
    
    # Definiamo i limiti per il fitting
    low = np.concatenate((ampiezze_iniziali * 0.1, posizioni_iniziali - 10., gamma_iniziali * 0.1, [-1]))  # Limiti inferiori
    upp = np.concatenate((ampiezze_iniziali * 10., posizioni_iniziali + 10., gamma_iniziali * 5., [1]))  # Limiti superiori
    
    # Eseguiamo il fitting
    parametri_ottimizzati, _ = curve_fit(modello, f, spettro, p0=parametri_iniziali, bounds=(low, upp))
    
    # Estraiamo le ampiezze, le posizioni e le larghezze ottimizzate
    ampiezze_ottimizzate = parametri_ottimizzati[:len(ampiezze_iniziali)]
    posizioni_ottimizzate = parametri_ottimizzati[len(ampiezze_iniziali):len(ampiezze_iniziali) + len(posizioni_iniziali)]
    gamma_ottimizzati = parametri_ottimizzati[len(ampiezze_iniziali) + len(posizioni_iniziali):]
    y0 = parametri_ottimizzati[-1]  # L'ultimo parametro è l'offset y0
    
    return ampiezze_ottimizzate, posizioni_ottimizzate, gamma_ottimizzati, y0


def fit_otto_lorentziane(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Esegue il fitting dello spettro utilizzando la funzione otto_lorentziane.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """
    N = len(posizioni_iniziali)
    if N < 8:
        if N % 2 != 0:
            posizioni_iniziali, gamma_iniziali, ampiezze_iniziali = pareggia_parametri(
                posizioni_iniziali,
                gamma_iniziali,
                ampiezze_iniziali
            )
        ampiezze_iniziali = np.tile(ampiezze_iniziali, 8 // N)
        posizioni_iniziali = np.tile(posizioni_iniziali, 8 // N)
        gamma_iniziali = np.tile(gamma_iniziali, 8 // N)
    
    return fit_spettro(f, spettro, ampiezze_iniziali, posizioni_iniziali, gamma_iniziali)