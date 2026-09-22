"""使用临时数据验证训练数据接入；不读取服务器真实数据。"""
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import trainv4 as training


class TrainV4Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split_root = self.root / "split"
        self.gt = self.root / "gt.txt"
        self.output = self.root / "run"
        self.labels = [f"command {chr(ord('a') + i)}" for i in range(20)]
        self.gt.write_text("\n".join(text + "." for text in self.labels), encoding="utf-8")
        self.rows = []
        for split_index, split in enumerate(("train", "val", "test")):
            for folder in ("Audio", "mmLip", "mmVocal", "txt"):
                (self.split_root / split / folder).mkdir(parents=True)
            for label_id, text in enumerate(self.labels):
                group = f"20260601_{120000 + split_index * 100 + label_id // 10:06d}"
                suffix = f"s{label_id % 10 + 1}"
                sample_id = f"audio_{group}_{suffix}"
                split_path = self.split_root / split
                (split_path / "Audio" / f"{sample_id}.wav").touch()
                (split_path / "txt" / f"{sample_id}.txt").write_text(text + ".", encoding="utf-8")
                length = 16 + label_id % 3
                np.save(split_path / "mmLip" / f"mmW_{group}_Lip_{suffix}.npy",
                        np.linspace(-1, 1, length, dtype=np.float32))
                np.save(split_path / "mmVocal" / f"mmW_{group}_Vib_{suffix}.npy",
                        np.linspace(-1, 1, length * 256, dtype=np.float32).reshape(length, 256))
                self.rows.append({
                    "split": split, "group_id": group, "sample_id": sample_id,
                    "suffix": suffix, "command_set": label_id // 10 + 1,
                    "label_id": label_id, "gt_number": label_id + 1,
                    # 划分脚本旧的 clean_text 会保留句末标点产生的空格。
                    "label_text": text + " ",
                })
        self.write_manifest(self.rows)

    def write_manifest(self, rows):
        with (self.split_root / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def dataset(self, split="train"):
        with redirect_stdout(io.StringIO()):
            return training.M2VoiceTextDataset(str(self.split_root / split), self.labels, split)

    def args(self, *extra):
        argv = ["trainv4", "--split-root", str(self.split_root), "--gt-file", str(self.gt),
                "--output-dir", str(self.output), "--device", "cpu", "--num-workers", "0", *extra]
        with patch("sys.argv", argv):
            return training.parse_args()

    def test_check_data_only_creates_no_model_or_output(self):
        with redirect_stdout(io.StringIO()), patch.object(training, "LipVocalTextClassifier") as model:
            training.train(self.args("--check-data-only"))
        model.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_manifest_is_checked_before_excluding_known_bad_group(self):
        extra_rows = []
        for label_id in range(10):
            group = "20260411_171730"
            suffix = f"s{label_id + 1}"
            sample_id = f"audio_{group}_{suffix}"
            root = self.split_root / "train"
            (root / "Audio" / f"{sample_id}.wav").touch()
            (root / "txt" / f"{sample_id}.txt").write_text(self.labels[label_id], encoding="utf-8")
            np.save(root / "mmLip" / f"mmW_{group}_Lip_{suffix}.npy", np.full(16, np.nan))
            np.save(root / "mmVocal" / f"mmW_{group}_Vib_{suffix}.npy", np.ones((16, 256)))
            extra_rows.append({
                "split": "train", "group_id": group, "sample_id": sample_id,
                "suffix": suffix, "command_set": 1, "label_id": label_id,
                "gt_number": label_id + 1, "label_text": self.labels[label_id],
            })
        self.write_manifest(self.rows + extra_rows)
        output = io.StringIO()
        with redirect_stdout(output):
            training.train(self.args("--check-data-only"))
        self.assertIn("passed (70 samples)", output.getvalue())
        self.assertIn("excluded=10, remaining=20", output.getvalue())
        self.assertFalse(self.output.exists())
        self.assertTrue((root / "Audio/audio_20260411_171730_s1.wav").exists())
        extra_rows[0]["label_id"] = 1
        self.write_manifest(self.rows + extra_rows)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "清单标签映射"):
            training.train(self.args("--check-data-only"))

    def test_exclusion_cannot_remove_a_whole_class_silently(self):
        dataset = self.dataset()
        for sample in dataset.samples:
            if sample["label"] == 0:
                sample["group_id"] = "20260411_171730"
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "缺少类别"):
            training.exclude_known_invalid_groups((dataset,))

    def test_feature_check_reports_all_unknown_invalid_samples(self):
        dataset = self.dataset()
        first, second = dataset.samples[:2]
        np.save(first["lip_path"], np.full(16, np.nan))
        np.save(second["vocal_path"], np.full((16, 256), np.inf))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught:
            training.check_features((dataset,))
        message = str(caught.exception)
        for expected in ("共 2 个样本", first["id"], second["id"], "Lip", "Vocal"):
            self.assertIn(expected, message)
        self.assertEqual(len(dataset), 20)

    def test_missing_modality_is_rejected(self):
        next((self.split_root / "train/mmLip").glob("*.npy")).unlink()
        with self.assertRaisesRegex(ValueError, "缺少文件"):
            self.dataset()

    def test_wrong_suffix_label_is_rejected(self):
        row = self.rows[0]
        path = self.split_root / "train/txt" / (row["sample_id"] + ".txt")
        path.write_text(self.labels[1], encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "后缀不匹配"):
            self.dataset()

    def test_excluded_sample_is_rejected(self):
        (self.split_root / "train/Audio/audio_20260411_174447_s1.wav").touch()
        with self.assertRaisesRegex(ValueError, "已排除样本"):
            self.dataset()

    def test_shared_group_with_distinct_samples_is_rejected(self):
        train = SimpleNamespace(samples=[{"id": "a_s1", "group_id": "same"}])
        val = SimpleNamespace(samples=[{"id": "a_s2", "group_id": "same"}])
        test = SimpleNamespace(samples=[{"id": "b_s1", "group_id": "other"}])
        with self.assertRaisesRegex(ValueError, "重复采集会话"):
            training.validate_split_overlap(train, val, test)

    def test_manifest_mismatch_and_omission_are_rejected(self):
        datasets = tuple(self.dataset(split) for split in ("train", "val", "test"))
        rows = [dict(row) for row in self.rows]
        rows[0]["label_id"] = 1
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "标签映射"):
            training.validate_split_manifest(self.split_root, datasets)
        self.write_manifest(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "未记录"):
            training.validate_split_manifest(self.split_root, datasets)

    def test_nonfinite_and_empty_features_are_rejected(self):
        dataset = self.dataset()
        path = dataset.samples[0]["lip_path"]
        for array in (np.array([0, np.nan]), np.array([0, np.inf]), np.array([])):
            np.save(path, array)
            with self.assertRaises(ValueError):
                dataset[0]

    def test_vocal_orientation_and_complex_values(self):
        dataset = self.dataset()
        path = dataset.samples[0]["vocal_path"]
        original = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
        np.save(path, original + 1j * original)
        first = dataset[0]["vocal"]
        np.save(path, (original + 1j * original).T)
        second = dataset[0]["vocal"]
        self.assertEqual(tuple(first.shape), (16, 256))
        torch.testing.assert_close(first, second)

    def test_model_forward_and_backward_with_variable_lengths(self):
        dataset = self.dataset()
        batch = training.collate_fn([dataset[0], dataset[2]])
        config = SimpleNamespace(hidden_size=8, mlp_dim=16, num_heads=2, num_layers=1, dropout=0.0)
        model = training.LipVocalTextClassifier(config, 20, target_seq_len=8)
        logits = model(batch["lip"], batch["vocal"])
        self.assertEqual(tuple(logits.shape), (2, 20))
        loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))

    def test_invalid_training_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            training.train(self.args("--epochs", "0"))
        self.assertFalse(self.output.exists())

    def test_existing_run_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("keep", encoding="utf-8")
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            training.train(self.args("--epochs", "1"))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
