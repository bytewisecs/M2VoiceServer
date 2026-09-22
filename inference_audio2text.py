import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ==========================================
# 1. 定义 Teacher 模型 (与训练脚本保持一致)
# ==========================================
class AudioTeacher(torch.nn.Module):
    def __init__(self, model_name="/data2/fanl/M2Voice/openAI_whisper/medium"):
        super().__init__()
        print(f"Loading local Whisper teacher model: {model_name}...")
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
        self.whisper.eval()
        
        # 冻结所有参数
        for param in self.whisper.parameters(): 
            param.requires_grad = False
            
    def generate_ground_truth(self, raw_audio, device):
        if isinstance(raw_audio, torch.Tensor):
            raw_audio = raw_audio.cpu().numpy()
            
        # Whisper 强制要求输入音频的采样率为 16000 Hz
        inputs = self.processor(raw_audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = self.whisper.generate(input_features)
        return predicted_ids

# ==========================================
# 2. 批量推理与文件保存逻辑
# ==========================================
def atomic_write_text(path, text):
    """先写入同目录临时文件，再原子替换，避免留下半个最终文件。"""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def source_identity(input_path, model_name):
    path = Path(input_path)
    stat = path.stat()
    return {
        "audio_path": str(path.resolve()),
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "model_path": str(Path(model_name).resolve()),
    }


def completion_path(output_path):
    return Path(str(output_path) + ".done.json")


def is_completed(output_path, identity):
    """只信任有完成记录且文本校验值一致的非空输出。"""
    try:
        content = Path(output_path).read_bytes()
        if not content.decode("utf-8").strip():
            return False
        record = json.loads(completion_path(output_path).read_text(encoding="utf-8"))
        return record == {
            "version": 1,
            "source": identity,
            "text_sha256": hashlib.sha256(content).hexdigest(),
        }
    except (OSError, UnicodeError, ValueError):
        return False


def save_transcription(output_path, transcription, identity):
    if not transcription.strip():
        raise ValueError("Whisper returned an empty transcription")
    # 必须先提交文本、再提交完成记录；任一步骤失败，下次都可重新处理。
    atomic_write_text(output_path, transcription)
    record = {
        "version": 1,
        "source": identity,
        "text_sha256": hashlib.sha256(transcription.encode("utf-8")).hexdigest(),
    }
    atomic_write_text(completion_path(output_path), json.dumps(record, ensure_ascii=False))


def run_inference(
    input_dir="/data2/fanl/M2Voice/dataset/dt4/Audio",
    output_dir="/data2/fanl/M2Voice/dataset/dt4/txt",
    model_name="/data2/fanl/M2Voice/openAI_whisper/medium",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computing device: {device}")
    
    # 扫描目录下所有的音频文件
    valid_extensions = ('.wav', '.mp3', '.flac')
    audio_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
        and os.path.isfile(os.path.join(input_dir, f))
    )
    total_files = len(audio_files)
    
    if total_files == 0:
        raise ValueError(f"No audio files found in {input_dir}.")

    stems = [os.path.splitext(filename)[0] for filename in audio_files]
    if len(set(stems)) != len(stems):
        raise ValueError("Audio filenames share a stem and would overwrite the same TXT file")
    os.makedirs(output_dir, exist_ok=True)
    teacher = None
    processed_count = skipped_count = failed_count = 0

    print(f"Found {total_files} audio files to process. Starting inference...")
    
    for idx, filename in enumerate(audio_files):
        input_path = os.path.join(input_dir, filename)
        
        # 构建输出的 .txt 文件名
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            identity = source_identity(input_path, model_name)
            if is_completed(output_path, identity):
                skipped_count += 1
                print(f"[{idx+1}/{total_files}] Skipping {filename}, completion verified.")
                continue
            # 先撤销旧完成记录；失败后不能把旧文本误判成成功。
            completion_path(output_path).unlink(missing_ok=True)
            if teacher is None:
                teacher = AudioTeacher(model_name).to(device)

            # 1. 加载音频
            waveform, sr = torchaudio.load(input_path)
            
            # 2. 强制转为单声道 (Whisper 只需要单声道)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # 3. 强制重采样到 16kHz (防止原始数据采样率不一致报错)
            if sr != 16000:
                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                
            # 将形状从 [1, Length] 压缩为 [Length]
            waveform = waveform.squeeze(0)
            
            # 4. 模型推理
            predicted_ids = teacher.generate_ground_truth(waveform, device)
            
            # 5. 解码为纯文本字符串 (跳过全部控制符和填充符)
            transcription = teacher.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            # 6. 确认输入未变化，并提交文本及完成记录
            if source_identity(input_path, model_name) != identity:
                raise RuntimeError("Audio file changed during inference; retry this file")
            save_transcription(output_path, transcription, identity)
            processed_count += 1
                
            print(f"[{idx+1}/{total_files}] Processed: {filename} -> '{transcription}'")
            
        except Exception as e:
            failed_count += 1
            print(f"Error processing {filename}: {str(e)}")

    print(f"Processed: {processed_count}, skipped: {skipped_count}, failed: {failed_count}")
    return 1 if failed_count else 0


def main():
    parser = argparse.ArgumentParser(description="Whisper 音频转文本，支持完整性校验后的断点恢复")
    parser.add_argument("--input-dir", default="/data2/fanl/M2Voice/dataset/dt4/Audio")
    parser.add_argument("--output-dir", default="/data2/fanl/M2Voice/dataset/dt4/txt")
    parser.add_argument("--model-name", default="/data2/fanl/M2Voice/openAI_whisper/medium")
    args = parser.parse_args()
    try:
        return run_inference(args.input_dir, args.output_dir, args.model_name)
    except (OSError, ValueError) as exc:
        print(f"Inference failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
