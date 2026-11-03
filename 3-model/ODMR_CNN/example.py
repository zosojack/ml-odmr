"""Esempio minimo per richiamare la CNN dal progetto Python del collega.

python example.py --counts "dati_osservati.npz"
Il file contiene counts e freq_MHz. Nessun metodo ibrido e implementato qui.
"""
from pathlib import Path
import argparse
import numpy as np
from cnn_model import CNNPredictor


def estimate_with_cnn(counts, frequency_MHz, model_dir):
    # In un programma reale costruire il predictor UNA volta e riutilizzarlo
    # per tutti gli spettri, per evitare di ricaricare i pesi a ogni chiamata.
    predictor = CNNPredictor(model_dir)
    return predictor.predict_counts(counts, frequency_MHz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', required=True)
    parser.add_argument('--model-dir', default=str(Path(__file__).parent/'artifacts'/'cnn_50k_48M'))
    args = parser.parse_args()
    with np.load(args.counts, allow_pickle=False) as data:
        result = estimate_with_cnn(data['counts'], data['freq_MHz'], args.model_dir)
    print('c:', result['c'])
    print('s:', result['s'])
    print('Frequenze previste [MHz]:', result['nu1_MHz'], result['nu2_MHz'])
