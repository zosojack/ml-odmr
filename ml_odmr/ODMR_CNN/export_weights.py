"""Raccoglie SOLO i pesi della CNN finale dal PC di Antony.

python export_weights.py --reference "../risultati_grid27_cfit_24h"
Non modifica i risultati originali, non addestra e non esegue fit.
L'output predefinito e artifacts/cnn_50k_48M dentro QUESTO progetto.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
import numpy as np
from cnn_model import (REFERENCE_SIGNATURE, ORIGINAL_SOURCE_SHA256, DATA_SHA256,
                       EXPECTED_WEIGHTS, SEEDS, TARGET_OFFSET, TARGET_SCALE, sha256)

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def get_reference(value):
    if value:
        p = Path(value).expanduser().resolve()
    else:
        import tkinter as tk
        from tkinter.filedialog import askdirectory
        root = tk.Tk(); root.withdraw()
        try:
            selected = askdirectory(title='Scegli la cartella COMPLETA risultati_grid27_cfit_24h')
        finally:
            root.destroy()
        if not selected:
            raise ValueError('Nessuna cartella selezionata. Puoi usare --reference "percorso".')
        p = Path(selected).resolve()
    if not (p/'config.json').is_file():
        raise FileNotFoundError(f'Manca {p / "config.json"}. Serve la cartella con jobs e predictions, '
                               'non la cartella degli script o il solo ZIP analisi.')
    return p


def weight_bytes(reference, seed):
    job = reference/'jobs'/f'N50000_P48M_seed{seed}'
    progress = read_json(job/'progress.json')
    expected = EXPECTED_WEIGHTS[seed]
    if not progress.get('complete') or progress['files_sha256']['best.weights.h5'] != expected:
        raise ValueError(f'Il checkpoint seed {seed} non e quello finale documentato.')
    for p in (job/'best.weights.h5', job/'work'/'best.weights.h5'):
        if p.is_file() and sha256(p) == expected:
            return p.read_bytes(), progress
    for p in sorted(job.glob('checkpoint_*.zip')):
        try:
            with zipfile.ZipFile(p) as z:
                data = z.read('best.weights.h5')
            if hashlib.sha256(data).hexdigest() == expected:
                return data, progress
        except (zipfile.BadZipFile, KeyError):
            continue
    raise FileNotFoundError(f'Pesi seed {seed} assenti o non validi in {job}. '
                           'Lo ZIP analisi non contiene i pesi addestrati.')


def export(reference, output):
    reference = get_reference(reference)
    conf = read_json(reference/'config.json')
    if conf.get('signature') != REFERENCE_SIGNATURE or conf.get('source_sha256') != ORIGINAL_SOURCE_SHA256:
        raise ValueError('La cartella non corrisponde al riferimento finale verificato.')
    if sha256(reference/'executed_code.py') != ORIGINAL_SOURCE_SHA256:
        raise ValueError('Il codice archiviato nel run originale e stato modificato.')
    if read_json(reference/'data_audit.json')['file_sha256'] != DATA_SHA256:
        raise ValueError('Il dataset del riferimento e diverso da quello finale verificato.')
    if not read_json(reference/'status.json').get('complete'):
        raise ValueError('Il run di riferimento non risulta completo.')
    scale = read_json(reference/'target_scaling.json')
    if not np.allclose(scale['offset'], TARGET_OFFSET, rtol=0, atol=1e-12) or not np.allclose(
            scale['scale'], TARGET_SCALE, rtol=0, atol=1e-12):
        raise ValueError('Scala dei target diversa da quella documentata.')
    weights = {seed: weight_bytes(reference, seed) for seed in SEEDS}
    output = Path(output).expanduser().resolve()
    if output == reference or reference in output.parents:
        raise ValueError('Scegli un output esterno ai risultati originali.')
    if output.exists():
        # Ripetere lo stesso comando non crea il precedente errore FileExistsError.
        manifest = output/'manifest.json'
        if manifest.is_file():
            m = read_json(manifest)
            if m.get('protocol') == 'odmr-cnn-only-v1' and all(
                    (output/name).is_file() and sha256(output/name) == expected
                    for name, expected in m['files_sha256'].items()) and all(
                    sha256(output/f'seed{s}.weights.h5') == EXPECTED_WEIGHTS[s] for s in SEEDS):
                print('Pesi CNN gia esportati e verificati:', output)
                return output
        raise FileExistsError('Output presente ma incompleto/diverso. Usa --output con una NUOVA cartella.')
    # Validiamo tutti gli input prima di creare la cartella di consegna.
    if not (reference/'splits.npz').is_file():
        raise FileNotFoundError(reference/'splits.npz')
    with open(reference/'predictions'/'N50000_P48M_ensemble_val.csv', newline='', encoding='utf-8') as f:
        expected = list(csv.DictReader(f))[:64]
    if len(expected) != 64:
        raise ValueError('Predizioni di controllo validation incomplete.')
    output.mkdir(parents=True)
    shutil.copy2(reference/'target_scaling.json', output/'target_scaling.json')
    shutil.copy2(reference/'splits.npz', output/'splits.npz')
    columns = ('sample_id', 'c_true', 's_true', 'c_pred', 's_pred')
    with open(output/'validation_check.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in expected])
    models = []
    for seed, (data, progress) in weights.items():
        (output/f'seed{seed}.weights.h5').write_bytes(data)
        models.append(dict(seed=seed, best_epoch=progress['best_epoch'],
                           best_weights_sha256=EXPECTED_WEIGHTS[seed]))
    manifest = dict(protocol='odmr-cnn-only-v1', reference_signature=REFERENCE_SIGNATURE,
                    original_source_sha256=ORIGINAL_SOURCE_SHA256, data_sha256=DATA_SHA256,
                    dataset_size=50000, train_size=40000, validation_size=5000, test_size=5000,
                    photons=48000000, points=335, window_MHz=[2750, 2950],
                    dropout=0., center_weight=10., models=models,
                    training_environment=conf['environment'],
                    files_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Pesi CNN esportati:', output)
    return output


def make_delivery(model_dir):
    """Archivio con progetto Python e veri pesi; i dati MAT restano separati."""
    archive = ROOT.parent/('ODMR_CNN_con_pesi_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ('*.py', '*.md', '*.txt', '.gitignore', '.gitattributes', 'docs/*.md', 'tests/*.py'):
            for p in sorted(ROOT.glob(pattern)):
                z.write(p, 'ODMR_CNN_GitHub/'+p.relative_to(ROOT).as_posix())
        for p in sorted(model_dir.iterdir()):
            if p.is_file():
                z.write(p, 'ODMR_CNN_GitHub/artifacts/cnn_50k_48M/'+p.name)
    print('ZIP completo da mandare al collega:', archive)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='', help='Vuoto: scegli la cartella dalla finestra')
    parser.add_argument('--output', default=str(ROOT/'artifacts'/'cnn_50k_48M'))
    parser.add_argument('--no-zip', action='store_true', help='Esporta solo la cartella dei pesi')
    args = parser.parse_args()
    folder = export(args.reference, args.output)
    if not args.no_zip:
        make_delivery(folder)
