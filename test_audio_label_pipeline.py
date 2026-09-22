"""在服务器运行：python -m unittest -v test_audio_label_pipeline。

测试仅使用临时目录和模拟模型，不读取真实数据或加载 Whisper 权重。
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gt_calibration_step2 as calibration
import inference_audio2text as inference


class LabelValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "Audio"
        self.txt = self.root / "txt_calibrated"
        self.audio.mkdir()
        self.txt.mkdir()
        self.gt = self.root / "gt.txt"
        self.gt.write_text("\n".join(f"command {i}" for i in range(1, 21)), encoding="utf-8")
        (self.audio / "sample_s1.wav").touch()
        self.label = self.txt / "sample_s1.TXT"
        self.label.write_text("command 1", encoding="utf-8")

    def errors(self):
        return calibration.validate_labels(self.gt, self.txt, self.audio)

    def test_both_candidates_and_uppercase_extension(self):
        self.assertEqual(self.errors(), [])
        self.label.write_text("command 11", encoding="utf-8")
        self.assertEqual(self.errors(), [])

    def test_gt_from_wrong_suffix_is_rejected(self):
        self.label.write_text("command 2", encoding="utf-8")
        self.assertTrue(any("只允许 GT 1 或 GT 11" in error for error in self.errors()))

    def test_s10_candidate_boundaries(self):
        (self.audio / "sample_s10.wav").touch()
        label = self.txt / "sample_s10.txt"
        for text in ("command 10", "command 20"):
            label.write_text(text, encoding="utf-8")
            self.assertEqual(self.errors(), [])

    def test_duplicate_stems_are_rejected(self):
        (self.audio / "sample_s1.flac").touch()
        self.assertTrue(any("音频同名冲突" in error for error in self.errors()))

    def test_missing_label_and_empty_directory_are_rejected(self):
        self.label.unlink()
        errors = self.errors()
        self.assertTrue(any("缺少标签" in error for error in errors))
        self.assertTrue(any("没有标签文件" in error for error in errors))

    def test_missing_directory_is_rejected(self):
        self.label.unlink()
        self.txt.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.errors()

    def test_empty_audio_directory_is_rejected(self):
        (self.audio / "sample_s1.wav").unlink()
        self.assertTrue(any("音频目录为空" in error for error in self.errors()))

    def test_empty_label_and_invalid_suffix_are_rejected(self):
        self.label.write_text(" ", encoding="utf-8")
        self.assertTrue(self.errors())
        self.label.rename(self.txt / "sample_s20.txt")
        self.assertTrue(any("后缀无效" in error for error in self.errors()))

    def test_cli_exit_status(self):
        argv = ["step2", "--gt-file", str(self.gt), "--txt-dir", str(self.txt),
                "--audio-dir", str(self.audio)]
        with patch("sys.argv", argv):
            self.assertEqual(calibration.main(), 0)
            self.label.write_text("command 2", encoding="utf-8")
            self.assertEqual(calibration.main(), 1)


class InferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "Audio"
        self.output = self.root / "txt"
        self.audio.mkdir()
        self.output.mkdir()
        self.source = self.audio / "sample_s1.wav"
        self.source.write_bytes(b"mock audio")
        self.target = self.output / "sample_s1.txt"
        self.identity = inference.source_identity(self.source, "mock-model")

    def test_legacy_empty_or_partial_output_is_not_skipped(self):
        for text in ("", " ", "partial sentence"):
            self.target.write_text(text, encoding="utf-8")
            self.assertFalse(inference.is_completed(self.target, self.identity))

    def test_completion_requires_intact_text_and_unchanged_source(self):
        inference.save_transcription(self.target, "complete sentence", self.identity)
        self.assertTrue(inference.is_completed(self.target, self.identity))
        self.source.write_bytes(b"changed audio data")
        changed = inference.source_identity(self.source, "mock-model")
        self.assertFalse(inference.is_completed(self.target, changed))
        self.target.write_text("truncated", encoding="utf-8")
        self.assertFalse(inference.is_completed(self.target, self.identity))

    def test_atomic_replace_failure_preserves_previous_text(self):
        self.target.write_text("previous", encoding="utf-8")
        with patch.object(inference.os, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                inference.atomic_write_text(self.target, "new text")
        self.assertEqual(self.target.read_text(encoding="utf-8"), "previous")
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_interruption_before_completion_record_requires_retry(self):
        inference.atomic_write_text(self.target, "complete sentence")
        self.assertFalse(inference.is_completed(self.target, self.identity))
        inference.completion_path(self.target).write_text("{", encoding="utf-8")
        self.assertFalse(inference.is_completed(self.target, self.identity))

    def test_empty_transcription_is_not_committed(self):
        with self.assertRaises(ValueError):
            inference.save_transcription(self.target, " ", self.identity)
        self.assertFalse(self.target.exists())
        self.assertFalse(inference.completion_path(self.target).exists())

    def test_verified_results_skip_model_loading(self):
        inference.save_transcription(self.target, "complete sentence", self.identity)
        with patch.object(inference, "AudioTeacher") as teacher:
            self.assertEqual(inference.run_inference(self.audio, self.output, "mock-model"), 0)
            teacher.assert_not_called()

    def test_failed_file_returns_nonzero_and_other_files_continue(self):
        (self.audio / "second_s2.wav").write_bytes(b"mock audio")
        with patch.object(inference, "AudioTeacher"), patch.object(
            inference.torchaudio, "load", side_effect=OSError("invalid audio")
        ) as load:
            self.assertEqual(inference.run_inference(self.audio, self.output, "mock-model"), 1)
            self.assertEqual(load.call_count, 2)

    def test_successful_inference_then_resume(self):
        waveform = inference.torch.zeros(1, 160)
        with patch.object(inference, "AudioTeacher") as teacher, patch.object(
            inference.torchaudio, "load", return_value=(waveform, 16000)
        ) as load:
            teacher.return_value.to.return_value.processor.batch_decode.return_value = ["hello"]
            self.assertEqual(inference.run_inference(self.audio, self.output, "mock-model"), 0)
            self.assertTrue(inference.is_completed(self.target, self.identity))
            self.assertEqual(inference.run_inference(self.audio, self.output, "mock-model"), 0)
            self.assertEqual(load.call_count, 1)

    def test_cli_propagates_failure_and_rejects_empty_input(self):
        argv = ["inference", "--input-dir", str(self.audio),
                "--output-dir", str(self.output), "--model-name", "mock-model"]
        with patch("sys.argv", argv), patch.object(inference, "AudioTeacher"), patch.object(
            inference.torchaudio, "load", side_effect=OSError("invalid audio")
        ):
            self.assertEqual(inference.main(), 1)
            self.source.unlink()
            self.assertEqual(inference.main(), 1)


if __name__ == "__main__":
    unittest.main()
