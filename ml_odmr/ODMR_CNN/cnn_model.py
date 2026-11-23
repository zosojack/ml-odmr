"""CNN B finale ODMR, TensorFlow/Keras. Solo CNN: nessun algoritmo di fitting.

L'architettura, la loss e le metriche sono estratte dal training originale.
La classe CNNPredictor carica i checkpoint appresi e decodifica le due uscite.
Le frequenze passate all'API devono essere espresse in MHz.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Queste identita provengono dal run finale gia analizzato, non da nuovi pesi.
SEEDS = (42, 137, 2026)
REFERENCE_SIGNATURE = '5a708b8af9addc728beba2ea4f5bd1c21bd10bdf0baaa2e5dc6cb93993c05444'
ORIGINAL_SOURCE_SHA256 = '1c03aece0377468e6a8912dd77cfbe182829656388be7d50e4f419b942c0542b'
DATA_SHA256 = '55a0abdc85da2fb783bd71e0edc5851ced8a6bb0de8c2ad0908f9b6497e074d3'
EXPECTED_WEIGHTS = {
    42: 'c52e94a6aefab3d49369e701ad1316c380e73d2483527ce2321c7135adffa4e2',
    137: '6b3f88e10b39f4758642051a5a65fa196b676d7ce9d9805528f12ed7f4379e22',
    2026: 'e6386309eaf31aa02119fc360c65ad0354aa8d0c86fef1ef863eebe8809efe2f',
}
TARGET_OFFSET = np.array([0.46012123978549435, 0.09999820744930771])
TARGET_SCALE = np.array([0.18038035019926157, 0.008374244575435341])
FREQUENCIES_MHZ = np.linspace(2750., 2950., 335)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@keras.utils.register_keras_serializable(package='ODMR')
class WeightedJointMSE(keras.losses.Loss):
    """L = [10*(q_c_true-q_c_pred)^2 + (q_s_true-q_s_pred)^2]/11.

    q e la trasformazione affine dei target; non e direttamente c o s.
    La riduzione media sul batch e eseguita da Keras.
    """
    def __init__(self, center_weight=10., name='weighted_joint_mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.center_weight = float(center_weight)
        if not np.isfinite(self.center_weight) or self.center_weight <= 0:
            raise ValueError('center_weight deve essere positivo e finito.')

    def call(self, y_true, y_pred):
        e2 = tf.square(tf.cast(y_true, tf.float32)-tf.cast(y_pred, tf.float32))
        return (self.center_weight*e2[:, 0]+e2[:, 1])/(self.center_weight+1.)

    def get_config(self):
        return {**super().get_config(), 'center_weight': self.center_weight}


@keras.utils.register_keras_serializable(package='ODMR')
class ParameterRMSE(keras.metrics.Metric):
    """Converte l'errore in q nell'errore sul centro/splitting in MHz."""
    def __init__(self, index, scale_MHz, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index, self.scale_MHz = int(index), float(scale_MHz)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        e = (tf.cast(y_pred[:, self.index], tf.float32)-
             tf.cast(y_true[:, self.index], tf.float32))*self.scale_MHz
        w = tf.ones_like(e) if sample_weight is None else tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        self.sse.assign_add(tf.reduce_sum(w*e*e))
        self.count.assign_add(tf.reduce_sum(w))

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.sse, self.count))

    def reset_state(self):
        self.sse.assign(0.); self.count.assign(0.)

    def get_config(self):
        return {**super().get_config(), 'index': self.index, 'scale_MHz': self.scale_MHz}


def build_cnn():
    """Costruisce ESATTAMENTE la CNN B finale: (335,1) -> 2 uscite lineari q.

    Il modello appena costruito ha pesi casuali: per inferenza scientifica
    usare CNNPredictor, che carica e verifica i checkpoint addestrati.
    """
    def residual(x, filters, kernel, name):
        skip = x
        z = layers.Conv1D(filters, kernel, padding='same', name=name+'_conv1')(x)
        z = layers.LayerNormalization(epsilon=1e-5)(z)
        z = layers.Activation('silu')(z)
        z = layers.Conv1D(filters, kernel, padding='same', name=name+'_conv2')(z)
        z = layers.LayerNormalization(epsilon=1e-5)(z)
        if int(skip.shape[-1]) != filters:
            skip = layers.Conv1D(filters, 1, padding='same', name=name+'_skip')(skip)
        return layers.Activation('silu')(layers.Add()([skip, z]))

    inp = keras.Input((335, 1), name='spectrum_01')
    x = layers.Conv1D(32, 13, padding='same', name='b_stem')(inp)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Activation('silu')(x)
    x = residual(x, 32, 9, 'b_res1')
    x = residual(x, 32, 9, 'b_res2')
    x = layers.AveragePooling1D(2)(x)
    x = residual(x, 64, 5, 'b_res3')
    x = residual(x, 64, 5, 'b_res4')
    x = layers.Conv1D(32, 1, activation='silu')(x)
    x = layers.Flatten()(x)  # 167*32 = 5344; conserva la posizione dei dip
    x = layers.Dense(128, activation='silu')(x)
    # Stessa costruzione del codice storico; rate impostato a zero prima dell'uso.
    dropout = layers.Dropout(.05)
    x = dropout(x)
    x = layers.Dense(64, activation='silu')(x)
    pred = layers.Dense(2, activation='linear', name='prediction_q')(x)
    model = keras.Model(inp, pred, name='ODMR_B_grid27')
    dropout.rate = 0.
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.),
        loss=WeightedJointMSE(10.),
        metrics=[ParameterRMSE(i, TARGET_SCALE[i]*200., name=f'rmse_{key}_MHz')
                 for i, key in enumerate(('c', 's'))],
        jit_compile=False)
    return model


def validate_frequencies(frequency_MHz):
    f = np.asarray(frequency_MHz, dtype=np.float64).squeeze()
    if f.shape != (335,) or not np.isfinite(f).all() or not np.allclose(
            f, FREQUENCIES_MHZ, atol=1e-3, rtol=0):
        raise ValueError('Attesa griglia crescente di 335 frequenze 2750-2950 MHz. '
                         'Non passare GHz, non interpolare automaticamente altre griglie.')
    return f


def normalize_fluorescence(counts):
    """Min-max per spettro. I dip DEVONO essere gia verso il basso.

    Input (335,) o (N,335), output (N,335,1). Non aggiunge rumore.
    Non riceve direttamente l'assorbimento positivo EasySpin.
    """
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 335 or len(x) == 0:
        raise ValueError('Attesa matrice (N,335) o singolo vettore (335,).')
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Attesi conteggi di fluorescenza finiti e non negativi.')
    span = np.ptp(x, axis=1, keepdims=True)
    if np.any(span <= 0):
        raise ValueError('Spettro costante: min-max impossibile.')
    return ((x-x.min(axis=1, keepdims=True))/span).astype(np.float32)[..., None]


def decode(cs):
    """s e la separazione COMPLETA. Nessun fit viene svolto da questa funzione."""
    cs = np.asarray(cs, dtype=np.float64)
    if cs.ndim != 2 or cs.shape[1] != 2 or not np.isfinite(cs).all():
        raise ValueError('Attesa matrice finita (N,2) con colonne c,s.')
    c, s = cs.T
    return dict(cs=cs, c=c, s=s, center_MHz=2750.+200.*c, delta_MHz=200.*s,
                nu1_MHz=2750.+200.*(c-s/2), nu2_MHz=2750.+200.*(c+s/2))


class CNNPredictor:
    """Interfaccia della CNN per essere richiamata dal codice del collega.

    seed=None: media dei 3 modelli (riferimento finale).
    seed=42/137/2026: singola rete gia addestrata.
    Non restituisce sigma e non esegue alcun raffinamento delle previsioni.
    """
    def __init__(self, model_dir, seed=None, batch_size=256):
        folder = Path(model_dir).expanduser().resolve()
        manifest_path = folder/'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Manca {manifest_path}. Esegui export_weights.py sul PC '
                                   'che conserva i checkpoint oppure ricevi i pesi da Antony.')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (self.manifest.get('protocol') != 'odmr-cnn-only-v1' or
                self.manifest.get('reference_signature') != REFERENCE_SIGNATURE):
            raise ValueError('Il pacchetto pesi non corrisponde al riferimento CNN finale.')
        for name, expected in self.manifest['files_sha256'].items():
            p = (folder/name).resolve()
            if folder not in p.parents or not p.is_file() or sha256(p) != expected:
                raise ValueError(f'File modello assente o modificato: {name}')
        scale_record = json.loads((folder/'target_scaling.json').read_text(encoding='utf-8'))
        self.offset = np.asarray(scale_record['offset'], dtype=np.float64)
        self.scale = np.asarray(scale_record['scale'], dtype=np.float64)
        if (self.offset.shape != (2,) or self.scale.shape != (2,) or
                not np.allclose(self.offset, TARGET_OFFSET, rtol=0, atol=1e-12) or
                not np.allclose(self.scale, TARGET_SCALE, rtol=0, atol=1e-12) or
                scale_record['window_MHz'] != [2750., 2950.]):
            raise ValueError('Le scale non coincidono con quelle del training originale.')
        if seed is not None and seed not in SEEDS:
            raise ValueError(f'Seed disponibili: {SEEDS}. None usa tutti e tre.')
        if batch_size <= 0:
            raise ValueError('batch_size deve essere positivo.')
        self.seeds = SEEDS if seed is None else (seed,)
        self.batch_size = int(batch_size)
        self.models = []
        for selected_seed in self.seeds:
            p = folder/f'seed{selected_seed}.weights.h5'
            if sha256(p) != EXPECTED_WEIGHTS[selected_seed]:
                raise ValueError(f'Checkpoint seed {selected_seed} diverso dal riferimento.')
            keras.utils.set_random_seed(selected_seed)
            model = build_cnn()
            model.optimizer.build(model.trainable_variables)
            model.load_weights(p)  # nessuno skip_mismatch
            model.trainable = False
            if model.count_params() != 806946:
                raise ValueError('La struttura della CNN non coincide con il riferimento.')
            self.models.append(model)

    def predict_normalized(self, spectra_01, frequency_MHz):
        """Per input GIA min-max, forma (N,335,1), dip verso il basso."""
        validate_frequencies(frequency_MHz)
        x = np.asarray(spectra_01, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (335, 1) or len(x) == 0:
            raise ValueError('Attesa forma (N,335,1), N positivo.')
        if not np.isfinite(x).all() or x.min() < -1e-6 or x.max() > 1+1e-6:
            raise ValueError('Input CNN non valido: attesi valori finiti in [0,1].')
        if np.any(np.ptp(x[:, :, 0], axis=1) <= 0):
            raise ValueError('Input CNN costante.')
        per_seed = []
        for model in self.models:
            q = np.concatenate([np.asarray(model(x[i:i+self.batch_size], training=False))
                                for i in range(0, len(x), self.batch_size)]).astype(np.float64)
            per_seed.append(self.offset+self.scale*q)
        per_seed = np.stack(per_seed)
        result = decode(per_seed.mean(axis=0))
        result.update(cs_by_seed=per_seed, seeds=np.asarray(self.seeds))
        return result

    def predict_counts(self, counts, frequency_MHz):
        """Input osservato -> min-max -> CNN -> c,s e frequenze. Nessun fit."""
        return self.predict_normalized(normalize_fluorescence(counts), frequency_MHz)
