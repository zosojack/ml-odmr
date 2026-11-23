"""Esegue SOLO la CNN finale e salva le sue previsioni.

python predict.py --mat "dati_50k.mat" --samples 20
python predict.py --counts "conteggi.npz"
Il secondo formato contiene counts (N,335) e freq_MHz (335,).
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import platform
import numpy as np
from cnn_model import CNNPredictor, DATA_SHA256, sha256, tf, keras
from preprocessing import read_clean_mat, reference_observations

ROOT = Path(__file__).resolve().parent


def run(args):
    if args.samples <= 0:
        raise ValueError('--samples deve essere positivo.')
    path = Path(args.mat or args.counts).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'File non trovato: {path}')
    model_dir = Path(args.model_dir).expanduser().resolve()
    predictor = CNNPredictor(model_dir, seed=args.seed)
    truth = None
    check = None
    if args.mat:
        # Questo ramo riproduce il benchmark storico, senza ridefinire le label.
        if sha256(path) != DATA_SHA256:
            raise ValueError('Per il controllo storico serve il MAT originale 50k/335 punti. '
                             'Il nome puo cambiare; il contenuto deve restare identico. '
                             'Per altri dati gia osservati usa --counts.')
        clean, freq, targets = read_clean_mat(path)
        with np.load(model_dir/'splits.npz', allow_pickle=False) as z:
            splits = {k: z[k].copy() for k in z.files}
        xv, nv, cv = reference_observations(clean, splits['val'], 'val')
        # Il controllo riproduce l'ensemble, anche se poi si chiede un solo seed.
        checker = predictor if args.seed is None else CNNPredictor(model_dir)
        expected = np.genfromtxt(model_dir/'validation_check.csv', delimiter=',', names=True)
        if not np.array_equal(expected['sample_id'].astype(int), splits['val'][:64]):
            raise ValueError('Ordine validation diverso dal riferimento.')
        if not np.allclose(np.column_stack([expected['c_true'], expected['s_true']]),
                           targets[splits['val'][:64]], atol=1e-10, rtol=0):
            raise ValueError('Label diverse da quelle archiviate.')
        got = checker.predict_normalized(xv[:64], freq)['cs']
        difference = np.abs(got-np.column_stack([expected['c_pred'], expected['s_pred']]))*200.
        check = dict(n_validation=64, tolerance_MHz=.001,
                     max_difference_MHz=difference.max(axis=0).tolist(),
                     passed=bool(difference.max() < .001))
        if not check['passed']:
            raise RuntimeError('Previsioni diverse dal riferimento oltre 1 kHz: '
                               f'{check}. Controlla pesi e versioni prima di procedere.')
        if args.split == 'val':
            x, counts, contrasts = xv, nv, cv
        else:
            x, counts, contrasts = reference_observations(clean, splits['test'], 'test')
        n = min(args.samples, len(x))
        ids = splits[args.split][:n]
        counts, contrasts = counts[:n], contrasts[:n]
        truth = targets[ids]
        print('Riferimento riprodotto entro 1 kHz: controllo di trasferimento superato.')
    else:
        # Conteggi gia osservati: NON simuliamo un secondo rumore Poisson.
        with np.load(path, allow_pickle=False) as file:
            if not {'counts', 'freq_MHz'}.issubset(file.files):
                raise KeyError('Il file NPZ deve contenere counts e freq_MHz.')
            counts = np.asarray(file['counts'])
            freq = np.asarray(file['freq_MHz'])
        if counts.ndim == 1:
            counts = counts[None, :]
        counts = counts[:args.samples]
        ids = np.arange(len(counts))
        contrasts = None
    result = predictor.predict_counts(counts, freq)
    output = Path(args.output or (ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists():
        raise FileExistsError(f'Cartella gia presente: {output}. Usa un nuovo --output oppure omettilo.')
    output.mkdir(parents=True)
    arrays = dict(counts=counts, freq_MHz=freq, sample_id=ids, **result)
    if truth is not None:
        arrays['cs_true'] = truth
        arrays['contrast'] = contrasts
    np.savez_compressed(output/'cnn_predictions.npz', **arrays)
    columns = ['sample_id', 'c', 's', 'center_MHz', 'delta_MHz', 'nu1_MHz', 'nu2_MHz']
    if truth is not None:
        columns += ['c_true', 's_true', 'error_center_MHz', 'error_delta_MHz']
    with open(output/'cnn_predictions.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i, sample_id in enumerate(ids):
            row = dict(sample_id=int(sample_id), **{key: float(result[key][i]) for key in columns[1:7]})
            if truth is not None:
                error = (result['cs'][i]-truth[i])*200.
                row.update(c_true=float(truth[i, 0]), s_true=float(truth[i, 1]),
                           error_center_MHz=float(error[0]), error_delta_MHz=float(error[1]))
            writer.writerow(row)
    metadata = dict(mode='reference_mat' if args.mat else 'observed_counts',
                    samples=len(ids), selected_seeds=list(predictor.seeds), reference_check=check,
                    tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                    python=platform.python_version(), gpu=[d.name for d in tf.config.list_physical_devices('GPU')],
                    model_manifest=predictor.manifest)
    if truth is not None:
        error = (result['cs']-truth)*200_000.
        metadata['rmse_center_and_delta_kHz'] = np.sqrt(np.mean(error**2, axis=0)).tolist()
        print('RMSE centro/splitting [kHz] sul sottoinsieme:', metadata['rmse_center_and_delta_kHz'])
    (output/'run_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Predizioni salvate in:', output)
    print('Prime stime c,s:\n', result['cs'][:5])
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mat', help='MAT originale del riferimento, in assorbimento pulito')
    mode.add_argument('--counts', help='NPZ con conteggi gia osservati e freq_MHz')
    p.add_argument('--model-dir', default=str(ROOT/'artifacts'/'cnn_50k_48M'))
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--split', choices=('val', 'test'), default='val', help='Usato solo con --mat')
    p.add_argument('--seed', type=int, choices=(42, 137, 2026), default=None,
                   help='Omettere per usare la media dei tre checkpoint finali')
    p.add_argument('--output', default='', help='Vuoto: nuova cartella automatica con data/ora')
    run(p.parse_args())
