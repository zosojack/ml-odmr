from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import curve_fit

@njit
def gaussiana(
    f: np.ndarray,
    A: float,
    f0: float,
    sigma: float
) -> np.ndarray:
    """
    Calcola la funzione di Gaussiana per un dato array di frequenze f,
    con centro f0 e deviazione standard sigma.
    """
    return A * np.exp(-0.5 * ((f - f0) / sigma)**2)

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
def pseudo_voigt(
    f: np.ndarray,
    A: float,
    f0: float,
    sigma: float,
    gamma: float,
    eta: float
) -> np.ndarray:
    """
    Calcola la funzione di pseudo-Voigt per un dato array di frequenze f,
    con centro f0, deviazione standard sigma, larghezza gamma e peso eta.
    """
    return eta * lorentziana(f, A, f0, gamma) + (1 - eta) * gaussiana(f, A, f0, sigma)

@njit
def multi_pseudo_voigt(
    f: np.ndarray,
    ampiezze: np.ndarray,
    posizioni: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    """
    Calcola la somma di più funzioni di pseudo-Voigt per un array di frequenze f,
    dato un array di posizioni dei picchi e un array di larghezze gamma.
    """
    spettro = np.zeros_like(f, np.float64)
    for i in range(len(posizioni)):
        spettro += pseudo_voigt(f, ampiezze[i], posizioni[i], sigma[i], gamma[i], eta[i])
    return spettro

@njit
def otto_pseudo_voigt(
    f: np.ndarray,
    ampiezze: np.ndarray,
    posizioni: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    eta: np.ndarray,
    y0: float = 0.
) -> np.ndarray:
    """
    Calcola la somma di otto funzioni di pseudo-Voigt per un array di frequenze f,
    dato un array di posizioni dei picchi e un array di larghezze gamma.
    """
    # I parametri possono non essere lunghi otto, l'importante è che siano pari
    # vengono duplicati; utile se ci sono più pseudo-Voigt sovrapposte
    
    spettro = np.zeros_like(f, np.float64)
    for i in range(8):
        spettro += pseudo_voigt(f, ampiezze[i], posizioni[i], sigma[i], gamma[i], eta[i])
    return spettro + y0


def pareggia_parametri(
    posizioni: np.ndarray,
    gamma: np.ndarray,
    ampiezze: np.ndarray,
    sigma: np.ndarray,
    eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Se il numero di parametri è dispari, cerca di pareggiarlo duplicando i parametri esistenti.
    Il criterio è che viene duplicato il centro senza un altro speculare.
    """
    '''
    if len(posizioni) == 3:
        valore_centrale = 2880
    elif len(posizioni) == 5:
        valore_centrale = 2900
    elif len(posizioni) == 7:
        valore_centrale = 2920
    else:
        valore_centrale = 2870'''
    valore_centrale = 2870
    
    # METODO 1, non bene per B alti #
    # come capire quale è il centro da duplicare? faccio una media dei centri e vedo quanto dista dal centro (2870)
    centro_medio = np.mean(posizioni)
    distanza_centro = centro_medio - valore_centrale # questo mi dice da che lato è il centro solitario
    # il centro da aggiungere è verosimilmente speculare alla stessa distanza
    centro_da_aggiungere = valore_centrale - distanza_centro
    
    # METODO 2 #
    '''punto_medio_estremi = (np.max(posizioni) + np.min(posizioni)) / 2
    distanza_centro = centro_medio - punto_medio_estremi
    centro_da_aggiungere = centro_medio + distanza_centro'''
    
    # gamma e ampiezza vengono aggiunte arbitrarie
    gamma_da_aggiungere = np.mean(gamma)
    ampiezza_da_aggiungere = np.mean(ampiezze)
    sigma_da_aggiungere = np.mean(sigma)
    eta_da_aggiungere = np.mean(eta)
    # aggiungo i parametri
    posizioni = np.append(posizioni, centro_da_aggiungere)
    gamma = np.append(gamma, gamma_da_aggiungere)
    ampiezze = np.append(ampiezze, ampiezza_da_aggiungere)
    sigma = np.append(sigma, sigma_da_aggiungere)
    eta = np.append(eta, eta_da_aggiungere)
    return posizioni, gamma, ampiezze, sigma, eta


def fit_spettro(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray,
    sigma_iniziali: np.ndarray,
    eta_iniziali: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Esegue il fitting dello spettro utilizzando la funzione multi_pseudo_voigt.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """    
    
    '''# Lunghezze per comodità
    La = len(ampiezze_iniziali)
    Lp = len(posizioni_iniziali)
    Lg = len(gamma_iniziali)
    Ls = len(sigma_iniziali)
    Le = len(eta_iniziali)

    def modello(f, *parametri):
        # Convertiamo forzatamente le tuple in numpy array per Numba
        ampiezze = np.array(parametri[:La], dtype=np.float64)
        posizioni = np.array(parametri[La : La + Lp], dtype=np.float64)
        gamma = np.array(parametri[La + Lp : La + Lp + Lg], dtype=np.float64)
        sigma = np.array(parametri[La + Lp + Lg : La + Lp + Lg + Ls], dtype=np.float64)
        eta = np.array(parametri[La + Lp + Lg + Ls : La + Lp + Lg + Ls + Le], dtype=np.float64)
        y0 = float(parametri[-1])'''
        
    def modello(f, *parametri):
        ampiezze = np.array(parametri[0:8], dtype=np.float64)
        posizioni = np.array(parametri[8:16], dtype=np.float64)
        gamma = np.array(parametri[16:24], dtype=np.float64)
        sigma = np.array(parametri[24:32], dtype=np.float64)
        eta = np.array(parametri[32:40], dtype=np.float64)
        y0 = float(parametri[-1])

        return -otto_pseudo_voigt(f, ampiezze, posizioni, gamma, sigma, eta, y0)  
    
    # Parametri iniziali
    parametri_iniziali = np.concatenate((ampiezze_iniziali, posizioni_iniziali, gamma_iniziali, sigma_iniziali, eta_iniziali, [0.]))
    
    # Limiti
    low = np.concatenate((ampiezze_iniziali * 0.1, posizioni_iniziali - 10., gamma_iniziali * 0.1, sigma_iniziali * 0.1, eta_iniziali * 0.1, [-1]))
    upp = np.concatenate((ampiezze_iniziali * 10., posizioni_iniziali + 10., gamma_iniziali * 5., sigma_iniziali * 5., eta_iniziali * 5., [1]))
    
    # Fit
    parametri_ottimizzati, _ = curve_fit(modello, f, spettro, p0=parametri_iniziali, bounds=(low, upp), maxfev=50000)
    
    # Estrazione
    ampiezze_ottimizzate = parametri_ottimizzati[0:8]
    posizioni_ottimizzate = parametri_ottimizzati[8:16]
    gamma_ottimizzati = parametri_ottimizzati[16:24]
    sigma_ottimizzate = parametri_ottimizzati[24:32]
    eta_ottimizzati = parametri_ottimizzati[32:40]
    y0 = parametri_ottimizzati[-1]
    
    return ampiezze_ottimizzate, posizioni_ottimizzate, gamma_ottimizzati, sigma_ottimizzate, eta_ottimizzati, y0



def fit_otto_pseudo_voigt(
    f: np.ndarray,
    spettro: np.ndarray,
    ampiezze_iniziali: np.ndarray,
    posizioni_iniziali: np.ndarray,
    gamma_iniziali: np.ndarray,
    sigma_iniziali: np.ndarray,
    eta_iniziali: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Esegue il fitting dello spettro utilizzando la funzione otto_pseudo_voigt.
    Restituisce le posizioni e le larghezze dei picchi ottimizzati.
    """
    # 1. FORZARE TUTTO A FLOAT64
    # Questo previene l'errore UFuncTypeError e garantisce la compatibilità con Numba/Scipy
    ampiezze = np.asarray(ampiezze_iniziali, dtype=np.float64)
    posizioni = np.asarray(posizioni_iniziali, dtype=np.float64)
    gamma = np.asarray(gamma_iniziali, dtype=np.float64)
    sigma = np.asarray(sigma_iniziali, dtype=np.float64)
    eta = np.asarray(eta_iniziali, dtype=np.float64)
    
    N = len(posizioni)
    
    # 2. GESTIONE SE SONO TROPPI (Troncamento di sicurezza)
    if N > 8:
        ampiezze = ampiezze[:8]
        posizioni = posizioni[:8]
        gamma = gamma[:8]
        sigma = sigma[:8]
        eta = eta[:8]
        N = 8
        
    # 3. GESTIONE SE SONO POCHI (Padding fino a 8)
    if N < 8:
        # Se i picchi sono dispari, chiamiamo la tua funzione per aggiungerne uno sensato
        if N % 2 != 0:
            posizioni, gamma, ampiezze, sigma, eta = pareggia_parametri(
                posizioni, gamma, ampiezze, sigma, eta
            )
            # Forza di nuovo il float64 per sicurezza dopo l'uscita dalla tua funzione
            posizioni = np.asarray(posizioni, dtype=np.float64)
            N = len(posizioni)
        
        # Se ora sono pari ma ancora minori di 8 (es. 4 o 6), riempiamo fino a 8
        if N < 8:
            ampiezze = np.resize(ampiezze, 8)
            posizioni = np.resize(posizioni, 8)
            gamma = np.resize(gamma, 8)
            sigma = np.resize(sigma, 8)
            eta = np.resize(eta, 8)
            
            # Ora che posizioni è sicuramente float64, il += con linspace non crasherà
            posizioni[N:] += np.linspace(1.5, 4.5, 8 - N)
            
    # 4. CHIAMATA AL FIT
    # Ora passiamo alla funzione di fit vettori garantiti di 8 elementi float64
    return fit_spettro(f, spettro, ampiezze, posizioni, gamma, sigma, eta)