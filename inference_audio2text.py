import os
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
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computing device: {device}")
    
    # 定义输入输出路径
    input_dir = '/data2/fanl/M2Voice/dataset/dt4/Audio'
    output_dir = '/data2/fanl/M2Voice/dataset/dt4/txt'
    
    # 自动创建目标文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 加载模型
    teacher = AudioTeacher("/data2/fanl/M2Voice/openAI_whisper/medium").to(device)
    
    # 扫描目录下所有的音频文件
    valid_extensions = ('.wav', '.mp3', '.flac')
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    total_files = len(audio_files)
    
    if total_files == 0:
        print(f"Warning: No audio files found in {input_dir}.")
        return
        
    print(f"Found {total_files} audio files to process. Starting inference...")
    
    for idx, filename in enumerate(audio_files):
        input_path = os.path.join(input_dir, filename)
        
        # 构建输出的 .txt 文件名
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # 【断点续传保护】如果该文件已经推理过了，直接跳过，方便意外中断后重新运行
        if os.path.exists(output_path):
            print(f"[{idx+1}/{total_files}] Skipping {filename}, already processed.")
            continue
            
        try:
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
            
            # 6. 保存到 txt 文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
                
            print(f"[{idx+1}/{total_files}] Processed: {filename} -> '{transcription}'")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    run_inference()




