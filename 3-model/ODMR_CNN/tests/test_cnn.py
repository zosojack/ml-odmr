"""Test tecnici: nessun peso appreso e nessun risultato scientifico simulato."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from scipy.io import savemat
from cnn_model import (build_cnn, WeightedJointMSE, ParameterRMSE, CNNPredictor,
                       normalize_fluorescence, validate_frequencies, decode,
                       FREQUENCIES_MHZ, keras)
from preprocessing import read_clean_mat, reference_observations


class CNNTests(unittest.TestCase):
    def test_architecture_and_weight_roundtrip(self):
        keras.backend.clear_session(); keras.utils.set_random_seed(42)
        original = build_cnn()
        self.assertEqual(original.count_params(), 806946)
        self.assertEqual(original.input_shape, (None, 335, 1))
        self.assertEqual(original.output_shape, (None, 2))
        self.assertTrue(all(layer.rate == 0 for layer in original.layers
                            if isinstance(layer, keras.layers.Dropout)))
        x = np.random.default_rng(1).uniform(size=(2, 335, 1)).astype(np.float32)
        original.optimizer.build(original.trainable_variables)
        expected = np.asarray(original(x, training=False))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'technical.weights.h5'
            original.save_weights(path)
            loaded = build_cnn(); loaded.optimizer.build(loaded.trainable_variables)
            loaded.load_weights(path)
            np.testing.assert_allclose(expected, loaded(x, training=False), atol=1e-7, rtol=1e-6)

    def test_weighted_loss(self):
        loss = WeightedJointMSE(10.)
        got = float(loss(np.zeros((1, 2), np.float32), np.array([[1, 2]], np.float32)))
        self.assertAlmostEqual(got, (10+4)/11, places=6)

    def test_physical_rmse_metric(self):
        metric = ParameterRMSE(1, 2.)
        metric.update_state(np.zeros((2, 2), np.float32), np.array([[0, 1], [0, 2]], np.float32))
        self.assertAlmostEqual(float(metric.result()), np.sqrt(10), places=6)
        metric.reset_state(); self.assertEqual(float(metric.result()), 0.)

    def test_full_splitting_and_units(self):
        decoded = decode(np.array([[.55, .105]]))
        np.testing.assert_allclose(decoded['center_MHz'], [2860.])
        np.testing.assert_allclose(decoded['delta_MHz'], [21.])
        np.testing.assert_allclose(decoded['nu1_MHz'], [2849.5])
        np.testing.assert_allclose(decoded['nu2_MHz'], [2870.5])

    def test_minmax_preserves_dip_direction(self):
        absorption = np.linspace(0, 1, 335)[None, :]
        counts = 100000*(1-.05*absorption)
        x = normalize_fluorescence(counts)
        self.assertEqual(x.shape, (1, 335, 1))
        self.assertEqual(int(x.argmin()), int(absorption.argmax()))
        np.testing.assert_allclose(normalize_fluorescence(3*counts+17), x, atol=1e-7)
        with self.assertRaises(ValueError): normalize_fluorescence(np.ones((1, 335)))

    def test_frequency_errors(self):
        validate_frequencies(FREQUENCIES_MHZ)
        with self.assertRaises(ValueError): validate_frequencies(FREQUENCIES_MHZ/1000)
        with self.assertRaises(ValueError): validate_frequencies(FREQUENCIES_MHZ[::-1])
        with self.assertRaises(ValueError): validate_frequencies(np.linspace(2750, 2950, 167))

    def test_mat_classic_and_hdf5(self):
        y = np.tile(np.linspace(0, 1, 335), (3, 1))
        c, s = np.full(3, .55), np.full(3, .105)
        with tempfile.TemporaryDirectory() as temp:
            classic, hdf = Path(temp)/'classic.mat', Path(temp)/'hdf.mat'
            savemat(classic, dict(spectra_data=y, mw_freqs=FREQUENCIES_MHZ/1000,
                                 labels_c=c, labels_s=s))
            with h5py.File(hdf, 'w') as file:
                file['spectra_data'] = y.T
                file['mw_freqs'] = FREQUENCIES_MHZ[None, :]/1000
                file['labels_c'] = c[None, :]; file['labels_s'] = s[None, :]
            for path in (classic, hdf):
                clean, freq, target = read_clean_mat(path)
                np.testing.assert_allclose(clean, y, atol=1e-7)
                np.testing.assert_allclose(freq, FREQUENCIES_MHZ)
                np.testing.assert_allclose(target, np.column_stack([c, s]))

    def test_missing_weights_are_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError): CNNPredictor(temp)

    def test_reference_noise_is_repeatable(self):
        clean = np.tile(np.linspace(0, 1, 335), (3, 1)).astype(np.float32)
        rows = np.arange(3)
        x, counts, contrast = reference_observations(clean, rows)
        x2, counts2, contrast2 = reference_observations(clean, rows)
        np.testing.assert_array_equal(counts, counts2)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(contrast, contrast2)
        np.testing.assert_array_equal(x, normalize_fluorescence(counts))


if __name__ == '__main__':
    unittest.main(verbosity=2)
