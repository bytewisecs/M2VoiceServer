"""服务器执行：python -m unittest -v test_split_dataset_v2。

仅创建临时文件，覆盖删除后 835 个样本的分组、复制和输出校验。
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import split_dataset_v2 as splitter


class SplitDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root, self.output = base / "source", base / "output"
        paths = {
            "ROOT_DIR": self.root,
            "OUT_DIR": self.output,
            "AUDIO_DIR": self.root / "Audio",
            "LIP_DIR": self.root / "mmLip",
            "VOCAL_DIR": self.root / "mmVocal",
            "TXT_DIR": self.root / "txt_calibrated",
            "GT_FILE": self.root / "gt.txt",
        }
        settings = patch.multiple(splitter, **paths)
        settings.start()
        self.addCleanup(settings.stop)
        self.labels = [f"command {chr(ord('a') + i)}" for i in range(20)]
        removed = {
            "20260411_174447": "s1",
            "20260411_174614": "s4",
            "20260411_174756": "s4",
            "20260411_180543": "s6",
            "20260411_181349": "s6",
        }
        group_ids = list(removed) + [f"20260501_{i:06d}" for i in range(79)]
        self.samples = []
        for index, group_id in enumerate(group_ids):
            command_set = index % 2
            for number in range(1, 11):
                suffix = f"s{number}"
                if removed.get(group_id) == suffix:
                    continue
                sample_id = f"audio_{group_id}_{suffix}"
                label_id = command_set * 10 + number - 1
                self.samples.append({
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "suffix": suffix,
                    "command_set": command_set,
                    "label_id": label_id,
                    "label_text": self.labels[label_id],
                    "paths": splitter.build_sample_paths(
                        paths["AUDIO_DIR"] / f"{sample_id}.wav", group_id, suffix
                    ),
                })

    def write_source(self, samples):
        for directory in ("Audio", "mmLip", "mmVocal", "txt_calibrated"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "gt.txt").write_text("\n".join(self.labels), encoding="utf-8")
        for sample in samples:
            for modality, path in sample["paths"].items():
                path.write_text(sample["label_text"] if modality == "txt" else "fixture", encoding="utf-8")

    def test_only_known_five_gaps_are_allowed(self):
        groups = splitter.validate_groups(self.samples)
        self.assertEqual(len(groups), 84)
        self.assertEqual(sum(map(len, groups.values())), 835)
        self.assertEqual(sum(len(rows) == 9 for rows in groups.values()), 5)
        with self.assertRaises(ValueError):
            splitter.validate_groups(self.samples[:-1])

    def test_mixed_command_sets_are_rejected(self):
        self.samples[0]["command_set"] = 1
        with self.assertRaises(ValueError):
            splitter.validate_groups(self.samples)

    def test_split_is_reproducible_and_has_no_group_leakage(self):
        groups = splitter.validate_groups(self.samples)
        plan = splitter.split_groups(groups)
        self.assertEqual(plan, splitter.split_groups(groups))
        splitter.validate_split_plan(groups, plan, 20)
        self.assertEqual({name: len(ids) for name, ids in plan.items()},
                         {"train": 50, "val": 26, "test": 8})
        assigned = [group for ids in plan.values() for group in ids]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(groups))

    def test_missing_classes_are_rejected_before_copy(self):
        groups = splitter.validate_groups(self.samples)
        plan = splitter.split_groups(groups)
        for group_id in plan["test"]:
            groups[group_id] = [s for s in groups[group_id] if s["label_id"] != 0]
        with self.assertRaises(ValueError):
            splitter.validate_split_plan(groups, plan, 20)
        self.assertFalse(self.output.exists())

    def test_removed_sample_must_not_reappear(self):
        self.write_source([])
        (self.root / "Audio/audio_20260411_174447_s1.wav").touch()
        with self.assertRaisesRegex(ValueError, "待排除样本仍存在"):
            splitter.collect_samples(self.labels)

    def test_missing_modality_is_not_silently_skipped(self):
        self.write_source(self.samples[:1])
        self.samples[0]["paths"]["mmVocal"].unlink()
        with self.assertRaisesRegex(ValueError, "缺少文件"):
            splitter.collect_samples(self.labels)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            splitter.prepare_output_dirs()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_check_only_then_full_copy(self):
        self.write_source(self.samples)
        argv = ["split", "--root", str(self.root), "--out-dir", str(self.output)]
        with patch("sys.argv", argv + ["--check-only"]):
            splitter.main()
        self.assertFalse(self.output.exists())
        with patch("sys.argv", argv):
            splitter.main()
        with (self.output / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 835)
        self.assertEqual({row["sample_id"] for row in rows}, {s["sample_id"] for s in self.samples})
        summary = json.loads((self.output / "split_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["sample_count"], 835)
        self.assertEqual(summary["group_count"], 84)
        for sample in self.samples:
            self.assertTrue(all(path.exists() for path in sample["paths"].values()))
        for row in rows:
            for path in next(s["paths"] for s in self.samples if s["sample_id"] == row["sample_id"]).values():
                # 原始标签目录在输出中命名为 txt。
                modality = "txt" if path.parent.name == "txt_calibrated" else path.parent.name
                self.assertEqual((self.output / row["split"] / modality / path.name).read_bytes(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
