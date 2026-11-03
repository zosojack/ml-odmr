from numba import njit

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