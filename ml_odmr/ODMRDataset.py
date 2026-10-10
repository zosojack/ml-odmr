import numpy as np
import torch
from torch.utils.data import Dataset

class ODMRDataset(Dataset):

    def __init__(self, raw_spectra, targets, is_train=True):
        """
        raw_spectra: np.ndarray [N, 167] con picco normalizzato a 1
        targets: np.ndarray [N, K] contenente i valori di riferimento (es.
        center, splitting) 
        is_train: bool, se False fissa il seed per
        rendere riproducibile validation/test
        """
        self.raw_spectra = raw_spectra.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.is_train = is_train

    def __len__(self):
        return len(self.raw_spectra)

    def __getitem__(self, idx):
        y_ideal = self.raw_spectra[idx]
        
        # 0. Se non è train (quindi validation o test)
        # usiamo un generatore con seed fisso basato sull'indice
        # per avere rumore sempre identico e riproducibile (invece che diverso ad ogni epoca)
        if not self.is_train:
            rng = np.random.default_rng(seed=12345 + idx)
            contrast = rng.uniform(0.012, 0.15)
            n_tot = rng.uniform(3.0e5, 6.0e6)
            expected_counts = n_tot * (spectrum_ideal / np.sum(spectrum_ideal))
            noisy_counts = rng.poisson(expected_counts).astype(np.float32)
        else:
            # Training: rumore sempre nuovo a ogni epoca
            contrast = np.random.uniform(0.012, 0.15)
            n_tot = np.random.uniform(3.0e5, 6.0e6)
            expected_counts = n_tot * (spectrum_ideal / np.sum(spectrum_ideal))
            noisy_counts = np.random.poisson(expected_counts).astype(np.float32)

        # 1. Contrasto ODMR casuale C in [0.012, 0.15]
        contrast = np.random.uniform(0.012, 0.15)
        spectrum_ideal = 1.0 - contrast * y_ideal

        # 2. Shot noise poissoniano:
        # In Yao: N_tot in [180.000, 3.600.000]
        # Noi ri-scaliamo per 167 punti
        n_tot = np.random.uniform(3.0e5, 6.0e6)
        expected_counts = n_tot * (spectrum_ideal / np.sum(spectrum_ideal))
        noisy_counts = np.random.poisson(expected_counts).astype(np.float32)

        # 3. Z-score normalization
        mean_val = np.mean(noisy_counts)
        std_val = np.std(noisy_counts)
        if std_val < 1e-8:
            std_val = 1e-8
        normalized_spectrum = (noisy_counts - mean_val) / std_val

        # Trasformazione in tensori PyTorch (1 canale, 167 campioni)
        x_tensor = torch.from_numpy(normalized_spectrum).unsqueeze(0)
        y_tensor = torch.from_numpy(self.targets[idx])

        return x_tensor, y_tensor