import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 假设你的 `fused_seq` 的特征维度是 384
HIDDEN_SIZE = 384

# ==========================================
# 1. 教师模型：提取 Audio 的文本 Ground Truth
# ==========================================
class AudioTeacher(nn.Module):
    def __init__(self, model_name="openai/whisper-tiny"):
        super(AudioTeacher, self).__init__()
        print(f"正在加载 Whisper 教师模型: {model_name}...")
        # 冻结 Whisper，因为我们只用它来生成标签，不训练它
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper.eval()
        for param in self.whisper.parameters():
            param.requires_grad = False
            
    def generate_ground_truth(self, raw_audio_waveform, sample_rate=16000):
        """
        输入原始音频波形，输出文本 Token 序列 (Ground Truth)
        raw_audio_waveform: [Batch, Length] 的 numpy 数组或 tensor
        """
        # 将音频处理为 Whisper 需要的 Mel 频谱
        inputs = self.processor(
            raw_audio_waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.whisper.device)
        
        # 让 Whisper 生成文本对应的 Token ID
        with torch.no_grad():
            predicted_ids = self.whisper.generate(input_features)
            
        # (可选) 解码为可读文本，用于打印日志查看效果
        # decoded_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return predicted_ids

# ==========================================
# 2. 学生模型：从 fused_seq 到 文本预测
# ==========================================
class FusedSeqToText_Student(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(FusedSeqToText_Student, self).__init__()
        # 在拿到融合后的 fused_seq 后，我们只需要一个分类头把它映射到词表大小
        # vocab_size 通常等于 Whisper 的词表大小 + 1 (用于 CTC 的空白符 Blank)
        self.vocab_size = vocab_size
        
        # 增加几层卷积或线性层作为 ASR Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, fused_seq):
        # fused_seq shape: [Batch, Sequence_Length, Hidden_Size]
        logits = self.decoder(fused_seq) # -> [Batch, Sequence_Length, Vocab_Size]
        
        # PyTorch 的 CTCLoss 要求输入经过 log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs

# ==========================================
# 3. 训练与损失计算流水线
# ==========================================
if __name__ == "__main__":
    # --- 模拟参数 ---
    batch_size = 2
    fused_seq_length = 150  # 融合特征的时间序列长度
    audio_length = 16000 * 3 # 3秒音频
    
    # 模拟输入数据
    # 1. 之前模块融合好的 Lip + Vocal 特征 (学生输入)
    mock_fused_seq = torch.rand(batch_size, fused_seq_length, HIDDEN_SIZE)
    # 2. 原始音频数据 (教师输入)
    mock_raw_audio = [torch.randn(audio_length).numpy() for _ in range(batch_size)] 

    # --- 初始化模型 ---
    # 实例化教师 (为了演示速度这里用 tiny，实际追求“最准”可以换成 medium 或 large-v3)
    teacher = AudioTeacher("openai/whisper-tiny")
    
    # 实例化学生 (Vocab Size 等于 Whisper 的词汇表大小 + 1 个 CTC 空白符)
    # Whisper 词表大小通常为 51865
    VOCAB_SIZE = teacher.processor.tokenizer.vocab_size + 1 
    BLANK_INDEX = VOCAB_SIZE - 1  # 最后一个索引作为 CTC Blank Token
    
    student = FusedSeqToText_Student(hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    
    # --- 定义优化器与 CTC Loss ---
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    # zero_infinity=True 可以防止训练初期出现极端的 loss 导致梯度爆炸
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    # ================= 训练循环 =================
    student.train()
    optimizer.zero_grad()
    
    print("\n--- Step 1: 教师生成 Ground Truth ---")
    # 这部分可以离线做（提前把音频全推一遍保存为文本），也可以在线做（每次 forward 时现推）
    target_ids = teacher.generate_ground_truth(mock_raw_audio)
    print(f"提取到的 Target IDs 维度: {target_ids.shape}") # [Batch, Text_Length]
    
    print("\n--- Step 2: 学生预测 ---")
    log_probs = student(mock_fused_seq)
    print(f"学生输出的 Log Probs 维度: {log_probs.shape}") # [Batch, Fused_Seq_Length, Vocab_Size]
    
    print("\n--- Step 3: 计算 CTC Loss ---")
    # PyTorch CTCLoss 对维度的严格要求：
    # 1. log_probs 必须转置为 [Sequence_Length, Batch, Vocab_Size]
    log_probs_transposed = log_probs.transpose(0, 1) 
    
    # 2. 准备 input_lengths (每个 batch 预测的序列长度)
    input_lengths = torch.full(size=(batch_size,), fill_value=fused_seq_length, dtype=torch.long)
    
    # 3. 准备 target_lengths (每个 batch 真实文本的长度，注意要去掉 padding)
    # Whisper 生成的 target_ids 会自带一些 padding token (id=50257)，需要计算真实长度
    pad_token_id = teacher.processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 50257 # Whisper 默认的 eos/pad
    
    # 过滤掉 target_ids 中的特殊标记，压缩成 1D tensor 供 CTC 使用
    target_lengths = torch.tensor([
        len([idx for idx in seq if idx != pad_token_id]) for seq in target_ids
    ], dtype=torch.long)
    
    # CTC 要求 target 是一个展平的 1D tensor
    targets_flattened = []
    for seq in target_ids:
        targets_flattened.extend([idx for idx in seq if idx != pad_token_id])
    targets_flattened = torch.tensor(targets_flattened, dtype=torch.long)

    # 计算最终 Loss
    loss = ctc_loss_fn(log_probs_transposed, targets_flattened, input_lengths, target_lengths)
    
    print(f"✅ 当前 Batch 的 CTC Loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    optimizer.step()
    print("✅ 参数更新完成！")
    