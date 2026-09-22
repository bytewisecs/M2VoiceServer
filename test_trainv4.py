"""使用临时数据验证训练数据接入；不读取服务器真实数据。"""
import csv
import io
import json
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
        # 已知四象限相位：绝对值实现会丢失这些差异。
        original = np.tile(np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64),
                           16 * 64).reshape(16, 256)
        phases = torch.tensor([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4],
                              dtype=torch.float32).repeat(16 * 64).reshape(16, 256)
        expected = torch.cat((phases.sin(), phases.cos()), dim=1)
        np.save(path, original)
        first = dataset[0]["vocal"]
        np.save(path, original.T)
        second = dataset[0]["vocal"]
        self.assertEqual(tuple(first.shape), (16, 512))
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(first, second)
        # 独立改变幅度，不应改变相位输入。
        amplitudes = np.linspace(0.1, 10, original.size, dtype=np.float32).reshape(original.shape)
        np.save(path, original * amplitudes)
        torch.testing.assert_close(dataset[0]["vocal"], expected)
        # 已提取的实数相位不能再次取 angle（否则会退化成 0/pi）。
        np.save(path, phases.numpy())
        torch.testing.assert_close(dataset[0]["vocal"], expected)
        np.save(path, phases.numpy() + 2 * np.pi)
        torch.testing.assert_close(dataset[0]["vocal"], expected, rtol=1e-5, atol=1e-6)

    def test_zero_real_phase_is_not_standardized_away(self):
        dataset = self.dataset()
        np.save(dataset.samples[0]["vocal_path"], np.zeros((16, 256), dtype=np.float32))
        sine, cosine = dataset[0]["vocal"].chunk(2, dim=1)
        torch.testing.assert_close(sine, torch.zeros_like(sine))
        torch.testing.assert_close(cosine, torch.ones_like(cosine))

    def test_nonfinite_complex_vocal_is_rejected_before_phase_conversion(self):
        dataset = self.dataset()
        path = dataset.samples[0]["vocal_path"]
        for value in (complex(np.inf, 1), complex(1, np.inf), complex(np.nan, 1)):
            with self.subTest(value=value):
                original = np.ones((16, 256), dtype=np.complex64)
                original[0, 0] = value
                np.save(path, original)
                with self.assertRaisesRegex(ValueError, "Vocal 原始复数特征"):
                    dataset[0]

    def test_model_forward_and_backward_with_variable_lengths(self):
        dataset = self.dataset()
        batch = training.collate_fn([dataset[0], dataset[2]])
        config = SimpleNamespace(hidden_size=8, pool_bins=3, vocal_bottleneck=4, num_layers=1, dropout=0.0)
        model = training.LipVocalTextClassifier(config, 20)
        logits = model(batch["lip"], batch["vocal"], batch["lip_lengths"], batch["vocal_lengths"])
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

    def test_compact_training_writes_versioned_checkpoint_and_history(self):
        args = self.args("--epochs", "1", "--batch-size", "20", "--hidden-size", "8",
                         "--num-layers", "1", "--pool-bins", "2", "--vocal-bottleneck", "4",
                         "--dropout", "0", "--no-augmentation")
        with redirect_stdout(io.StringIO()):
            training.train(args)
        checkpoint = torch.load(self.output / "checkpoints/best_model.pth", map_location="cpu")
        self.assertEqual(checkpoint["model_version"], training.MODEL_VERSION)
        self.assertEqual(checkpoint["run_metadata"]["vocal_representation"],
                         "phase_sin_cos_no_standardization")
        self.assertEqual(checkpoint["run_metadata"]["vocal_input_channels"], 512)
        self.assertEqual(checkpoint["config"]["pool_bins"], 2)
        history = json.loads((self.output / "training_history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(history), 1)
        self.assertIn("train_eval_accuracy", history[0])
        self.assertTrue((self.output / "checkpoints/test_results.json").is_file())


class LengthAwareModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        config = SimpleNamespace(hidden_size=8, pool_bins=3, vocal_bottleneck=4, num_layers=1, dropout=0.0)
        self.model = training.LipVocalTextClassifier(config, 20)
        self.lip = torch.randn(1, 1, 17)
        self.vocal = torch.randn(1, 512, 23)

    def test_prediction_is_independent_of_other_sample_lengths(self):
        self.model.eval()
        with torch.no_grad():
            alone = self.model(self.lip, self.vocal, [17], [23])
            # 让第一条样本与更长的另一条样本同批，验证补零不改变预测。
            lip_batch = torch.cat((torch.nn.functional.pad(self.lip, (0, 44)), torch.randn(1, 1, 61)))
            vocal_batch = torch.cat((torch.nn.functional.pad(self.vocal, (0, 50)), torch.randn(1, 512, 73)))
            batched = self.model(lip_batch, vocal_batch, [17, 61], [23, 73])
        torch.testing.assert_close(alone[0], batched[0], rtol=1e-4, atol=1e-5)

    def test_padding_contents_are_ignored_and_receive_no_gradient(self):
        lip = torch.nn.functional.pad(self.lip, (0, 16), value=1000).requires_grad_()
        vocal = torch.nn.functional.pad(self.vocal, (0, 18), value=-1000).requires_grad_()
        padded_logits = self.model(lip, vocal, [17], [23])
        unpadded_logits = self.model(self.lip, self.vocal, [17], [23])
        torch.testing.assert_close(padded_logits, unpadded_logits, rtol=1e-4, atol=1e-5)
        padded_logits.square().sum().backward()
        self.assertEqual(torch.count_nonzero(lip.grad[:, :, 17:]).item(), 0)
        self.assertEqual(torch.count_nonzero(vocal.grad[:, :, 23:]).item(), 0)
        self.assertTrue(torch.isfinite(lip.grad).all().item())

    def test_train_eval_agree_when_dropout_is_disabled(self):
        with torch.no_grad():
            self.model.train()
            train_logits = self.model(self.lip, self.vocal, [17], [23])
            self.model.eval()
            eval_logits = self.model(self.lip, self.vocal, [17], [23])
        torch.testing.assert_close(train_logits, eval_logits, rtol=1e-4, atol=1e-5)

    def test_invalid_lengths_are_rejected(self):
        for lengths in ([0], [18], [17, 17]):
            with self.assertRaises(ValueError):
                self.model(self.lip, self.vocal, lengths, [23])

    def test_collate_tracks_each_modality_length(self):
        batch = training.collate_fn([
            {"id": "a", "text": "a", "label": 0, "lip": torch.ones(17), "vocal": torch.ones(23, 512)},
            {"id": "b", "text": "b", "label": 1, "lip": torch.ones(31), "vocal": torch.ones(19, 512)},
        ])
        self.assertEqual(batch["lip_lengths"].tolist(), [17, 31])
        self.assertEqual(batch["vocal_lengths"].tolist(), [23, 19])


class CompactTCNTests(unittest.TestCase):
    def config(self, modality="fusion"):
        return SimpleNamespace(hidden_size=8, num_layers=1, dropout=0.0, pool_bins=3,
                               vocal_bottleneck=4, modality=modality)

    def test_single_modality_does_not_use_other_input(self):
        lip, vocal = torch.randn(2, 1, 17), torch.randn(2, 512, 23)
        for modality in ("lip", "vocal"):
            model = training.LipVocalTextClassifier(self.config(modality), 20).eval()
            with torch.no_grad():
                expected = model(lip, vocal, [17, 17], [23, 23])
                actual = model(lip if modality == "lip" else torch.full_like(lip, float("nan")),
                               vocal if modality == "vocal" else torch.full_like(vocal, float("nan")),
                               [17, 17], [23, 23])
            torch.testing.assert_close(actual, expected)

    def test_fusion_and_auxiliary_objective(self):
        model = training.LipVocalTextClassifier(self.config(), 20)
        outputs = model(torch.randn(2, 1, 17), torch.randn(2, 512, 23),
                        [17, 15], [23, 19], return_aux=True)
        torch.testing.assert_close(outputs["logits"], (outputs["lip_logits"] + outputs["vocal_logits"]) / 2)
        labels = torch.tensor([0, 1])
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        loss = training.classification_objective(outputs, labels, criterion, 0.2)
        loss.backward()
        for branch in (model.lip_branch, model.vocal_branch):
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0 for p in branch.parameters()))
        self.assertTrue(torch.isfinite(loss).item())

    def test_augmentation_preserves_labels_and_original_arrays(self):
        sample = {"id": "a", "text": "command", "label": 3,
                  "lip": torch.ones(100), "vocal": training.encode_vocal_phase(torch.ones(200, 256))}
        original_vocal = sample["vocal"].clone()
        torch.manual_seed(7)
        batch = training.collate_train_fn([sample], noise_std=0.02, time_scale=0.1, time_mask=0.05)
        self.assertEqual(batch["id"], ["a"])
        self.assertEqual(batch["label"].tolist(), [3])
        self.assertTrue(90 <= batch["lip_lengths"][0].item() <= 110)
        self.assertTrue(180 <= batch["vocal_lengths"][0].item() <= 220)
        torch.testing.assert_close(sample["lip"], torch.ones(100))
        torch.testing.assert_close(sample["vocal"], original_vocal)
        self.assertTrue(torch.isfinite(batch["lip"]).all().item())
        self.assertTrue(torch.isfinite(batch["vocal"]).all().item())
        sine, cosine = batch["vocal"][0].chunk(2, dim=0)
        norms_squared = sine.square() + cosine.square()
        # 遮挡必须同时清空 sin/cos；其他位置保留单位圆。
        self.assertTrue(torch.all((norms_squared == 0) | ((norms_squared - 1).abs() < 1e-5)).item())

    def test_disabled_augmentation_matches_plain_collation(self):
        samples = [{"id": "a", "text": "a", "label": 0,
                    "lip": torch.randn(17), "vocal": training.encode_vocal_phase(torch.randn(23, 256))}]
        actual = training.collate_train_fn(samples, noise_std=0, time_scale=0, time_mask=0)
        expected = training.collate_fn(samples)
        for key in ("lip", "vocal", "lip_lengths", "vocal_lengths", "label"):
            torch.testing.assert_close(actual[key], expected[key])

    def test_phase_resize_crosses_pi_without_passing_through_zero(self):
        phase = torch.deg2rad(torch.tensor([[179.0], [-179.0]])).repeat(1, 256)
        resized = training.resize_vocal_phase(training.encode_vocal_phase(phase), 3)
        sine, cosine = resized.chunk(2, dim=1)
        torch.testing.assert_close(sine[1], torch.zeros(256), atol=1e-6, rtol=0)
        torch.testing.assert_close(cosine[1], -torch.ones(256), atol=1e-6, rtol=0)
        torch.testing.assert_close(sine.square() + cosine.square(), torch.ones(3, 256))

    def test_phase_resize_handles_antipodal_frames(self):
        # 精确反向的单位向量在中点抵消，应确定性回退到最近帧。
        vocal = torch.cat((torch.zeros(2, 256), torch.tensor([[1.0], [-1.0]]).repeat(1, 256)), dim=1)
        resized = training.resize_vocal_phase(vocal, 3)
        self.assertTrue(torch.isfinite(resized).all().item())
        torch.testing.assert_close(resized[1], vocal[0])
        sine, cosine = resized.chunk(2, dim=1)
        torch.testing.assert_close(sine.square() + cosine.square(), torch.ones(3, 256))

    def test_training_resize_uses_circular_representation(self):
        phase = torch.deg2rad(torch.tensor([[179.0], [-179.0]])).repeat(1, 256)
        sample = {"id": "a", "text": "a", "label": 0,
                  "lip": torch.ones(2), "vocal": training.encode_vocal_phase(phase)}
        # 强制 2 -> 3 帧，检查训练增强确实使用了圆周插值路径。
        with patch.object(torch, "rand", return_value=torch.tensor(1.0)):
            batch = training.collate_train_fn([sample], noise_std=0, time_scale=0.5, time_mask=0)
        self.assertEqual(batch["vocal_lengths"].tolist(), [3])
        torch.testing.assert_close(batch["vocal"][0, 256:, 1], -torch.ones(256), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
