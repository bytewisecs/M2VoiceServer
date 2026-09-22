import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnose_vocal_phase import phase_statistics, summarize_split


class PhaseStatisticsTests(unittest.TestCase):
    def test_static_channel_offsets_have_no_temporal_variation(self):
        phase = np.tile(np.linspace(-3, 3, 256), (10, 1))
        result = phase_statistics(np.exp(1j * phase))
        self.assertGreater(result["phase_std_radians"], 1)
        self.assertLess(result["circular_temporal_rms"], 1e-12)
        self.assertLess(result["adjacent_phase_step_rms_radians"], 1e-12)

    def test_pi_boundary_and_transposed_complex_input(self):
        phase = np.tile(np.deg2rad([179.0, -179.0])[:, None], (1, 256))
        real = phase_statistics(phase)
        complex_result = phase_statistics(np.exp(1j * phase).T)
        self.assertAlmostEqual(real["circular_temporal_rms"], np.sin(np.deg2rad(1)))
        self.assertAlmostEqual(real["adjacent_phase_step_rms_radians"], np.deg2rad(2))
        self.assertAlmostEqual(real["circular_temporal_rms"], complex_result["circular_temporal_rms"])

    def test_nonfinite_complex_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            phase_statistics(np.full((2, 256), complex(np.inf, 1)))

    def test_split_excludes_known_bad_group_and_reports_invalid_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "train/mmVocal"
            folder.mkdir(parents=True)
            np.save(folder / "mmW_20260601_120000_Vib_s1.npy", np.zeros((4, 256)))
            np.save(folder / "mmW_20260411_171730_Vib_s1.npy", np.full((4, 256), np.nan))
            np.save(folder / "mmW_20260601_120001_Vib_s1.npy", np.full((4, 256), np.inf))
            result = summarize_split(root, "train", 0.02)
            self.assertEqual(result["valid_samples"], 1)
            self.assertEqual(result["excluded_samples"], 1)
            self.assertEqual(result["error_count"], 1)
            self.assertEqual(result["samples_temporal_rms_below_noise_chord"], 1)
            self.assertAlmostEqual(result["expected_noise_chord_rms"], 0.02, places=5)
            self.assertEqual(len(list(folder.glob("*.npy"))), 3)


if __name__ == "__main__":
    unittest.main()
