import numpy as np
from numba import njit

class PoissonNoise:
    """
    Classe per aggiungere rumore di Poisson agli spettri ODMR.
    """
    
    def __init__(self, seed: int = None):
        """
        Inizializza la classe PoissonNoise.
        `seed`: seme per il generatore di numeri casuali (opzionale)
        """
        if seed is not None:
            np.random.seed(seed)
            
    def ideal_signal(
        self, 
        spectrum: np.ndarray, 
        contrast: float = 0.08
    ) -> np.ndarray:
        """
        Calcola il segnale ideale dato uno spettro e un contrasto.
        `spectrum`: array dello spettro normalizzato
        `contrast`: valore del contrasto
        Restituisce: array del segnale ideale
        """
        # Generazione del segnale ideale (senza rumore)
        ideal_signal = 1.0 - contrast * spectrum
        
        return ideal_signal

    def add_noise(
        self, 
        ideal_signal: np.ndarray, 
        n_tot: float = 3e05,
        seed_modifier: int = 0
    ) -> np.ndarray:
        """
        Aggiunge rumore di Poisson ai dati.
        `ideal_signal`: array del segnale ideale
        `n_tot`: numero totale di conteggi
        `seed_modifier`: modificatore del seme per il generatore di numeri casuali
        Restituisce: array di dati con il rumore di Poisson aggiunto
        """
        expected_counts = n_tot * (ideal_signal / np.sum(ideal_signal))
        
        # Campionamento poissoniano (rumoroso)
        rng = np.random.default_rng(
            seed=self.seed + seed_modifier
            if hasattr(self, 'seed') else None
        )
        noisy_signal = rng.poisson(expected_counts).astype(np.float32)
        
        return noisy_signal