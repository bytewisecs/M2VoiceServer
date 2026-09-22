"""仅在临时目录测试清理操作，不访问真实数据。"""
import tempfile
import unittest
from pathlib import Path

from remove_abnormal_samples import collect_targets, remove_samples


class RemoveSamplesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.targets = []
        for folder in ("Audio", "txt", "txt_calibrated", "mmLip", "mmVocal"):
            (self.root / folder).mkdir()
        for group, suffix in (
            ("20260411_174447", "s1"),
            ("20260411_174614", "s4"),
            ("20260411_174756", "s4"),
            ("20260411_180543", "s6"),
            ("20260411_181349", "s6"),
        ):
            filenames = (
                f"Audio/audio_{group}_{suffix}.wav",
                f"txt/audio_{group}_{suffix}.txt",
                f"txt/audio_{group}_{suffix}.txt.done.json",
                f"txt_calibrated/audio_{group}_{suffix}.txt",
                f"mmLip/mmW_{group}_Lip_{suffix}.npy",
                f"mmVocal/mmW_{group}_Vib_{suffix}.npy",
            )
            for filename in filenames:
                path = self.root / filename
                path.write_text("sample", encoding="utf-8")
                self.targets.append(path)
        self.keep = [
            self.root / "Audio/audio_20260411_174447_s10.wav",
            self.root / "txt_calibrated/audio_20260411_174447_s2.txt",
            self.root / "mmLip/mmW_20260411_174447_Lip_s10.npy",
            self.root / "mmVocal/mmW_20260411_174447_Vib_s10.npy",
            self.root / "gt.txt",
            self.root / "calibration_report.csv",
        ]
        for path in self.keep:
            path.write_text("keep", encoding="utf-8")

    def test_preview_does_not_delete(self):
        self.assertEqual(remove_samples(self.root), len(self.targets))
        self.assertTrue(all(path.exists() for path in self.targets + self.keep))

    def test_apply_deletes_only_exact_targets_and_can_repeat(self):
        self.assertEqual(remove_samples(self.root, apply=True), len(self.targets))
        self.assertTrue(all(not path.exists() for path in self.targets))
        self.assertTrue(all(path.read_text(encoding="utf-8") == "keep" for path in self.keep))
        self.assertEqual(remove_samples(self.root, apply=True), 0)

    def test_uppercase_extension_is_included(self):
        path = self.root / "Audio/audio_20260411_174447_s1.wav"
        renamed = path.with_suffix(".WAV")
        path.rename(renamed)
        self.assertIn(renamed, collect_targets(self.root))

    def test_symlink_is_rejected_before_any_deletion(self):
        path = self.targets[-1]
        path.unlink()
        path.symlink_to(self.keep[-1])
        with self.assertRaises(ValueError):
            remove_samples(self.root, apply=True)
        self.assertTrue(all(path.exists() for path in self.targets + self.keep))

    def test_missing_root_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            remove_samples(self.root / "missing", apply=True)
        self.assertTrue(all(path.exists() for path in self.targets + self.keep))


if __name__ == "__main__":
    unittest.main()
