"""Solo preparazione degli input CNN; nessuna estrazione o modifica delle label.

La simulazione Poisson qui serve alla dimostrazione con il MAT pulito.
Per conteggi gia misurati usare direttamente CNNPredictor.predict_counts.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
from cnn_model import validate_frequencies, normalize_fluorescence


def read_clean_mat(path):
    """Legge .mat classici o HDF5 v7.3 in sola lettura.

    Convenzioni del dataset finale: mw_freqs in GHz, assorbimento positivo,
    labels_c/s gia ottenute dal fit pulito sulla finestra di 200 MHz.
    """
    keys = ('spectra_data', 'mw_freqs', 'labels_c', 'labels_s')
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as file:
            data = {k: np.asarray(file[k]) for k in keys if k in file}
    else:
        data = loadmat(path, variable_names=list(keys))
    for key in keys:
        if key not in data:
            raise KeyError(f'Nel MAT manca {key}.')
    freq = validate_frequencies(np.asarray(data['mw_freqs']).squeeze()*1000.)
    clean = np.asarray(data['spectra_data'], dtype=np.float64).squeeze()
    if clean.ndim != 2:
        raise ValueError('spectra_data deve essere una matrice.')
    if clean.shape[1] != 335:
        if clean.shape[0] == 335:
            clean = clean.T
        else:
            raise ValueError('spectra_data deve contenere 335 frequenze.')
    if not np.isfinite(clean).all() or clean.min() < -1e-7 or np.any(np.ptp(clean, axis=1) <= 1e-10):
        raise ValueError('Atteso assorbimento EasySpin positivo, finito e non costante.')
    clean = np.maximum(clean, 0)
    clean /= clean.max(axis=1, keepdims=True)
    target = np.column_stack([np.asarray(data[k]).reshape(-1) for k in ('labels_c', 'labels_s')])
    if target.shape != (len(clean), 2) or not np.isfinite(target).all():
        raise ValueError('Label c/s non valide.')
    return clean.astype(np.float32), freq, target.astype(np.float64)


def reference_observations(clean, rows, split='val'):
    """Replica il rumore a 48M originale: stesse righe e stessi blocchi RNG.

    Richiamare sull'INTERO split originale prima di estrarre il sottoinsieme.
    Restituisce input (N,335,1), conteggi grezzi (N,335), contrasto (N,).
    """
    if split not in ('val', 'test'):
        raise ValueError('split deve essere val o test.')
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(rows) == 0 or rows.min() < 0 or rows.max() >= len(clean):
        raise ValueError('Indici dello split non validi.')
    x = np.empty((len(rows), 335, 1), dtype=np.float32)
    counts_all = np.empty((len(rows), 335), dtype=np.int64)
    contrasts = np.empty(len(rows), dtype=np.float64)
    code = 2001 if split == 'val' else 2002
    for block, start in enumerate(range(0, len(rows), 256)):
        stop = min(start+256, len(rows))
        token = [42, code, 0, 0, block]
        rng_c = np.random.default_rng(np.random.SeedSequence(token+[901]))
        contrast = rng_c.uniform(.012, .15, stop-start)
        ideal = 1.-contrast[:, None]*clean[rows[start:stop]].astype(np.float64)
        prob = ideal/ideal.sum(axis=1, keepdims=True)
        counts = np.zeros_like(prob, dtype=np.int64)
        previous = 0
        # La somma di Poisson indipendenti e ancora Poisson; manteniamo le
        # tre estrazioni originali per riprodurre gli stessi numeri casuali.
        for stage, level in enumerate((6_000_000, 24_000_000, 48_000_000)):
            rng = np.random.default_rng(np.random.SeedSequence(token+[1000+stage]))
            counts += rng.poisson((level-previous)*prob)
            previous = level
        x[start:stop] = normalize_fluorescence(counts)
        counts_all[start:stop] = counts
        contrasts[start:stop] = contrast
    return x, counts_all, contrasts
