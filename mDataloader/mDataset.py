import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio  # 用于读取 wav 音频文件，若未安装可通过 pip install torchaudio 安装

import pdb

from torch.nn.utils.rnn import pad_sequence

INVALID_GROUPS = {
    "20260411_171730",
}

def multi_modal_collate_fn(batch):
    """
    自定义的 batch 打包逻辑，用于处理变长序列补齐 (Padding)
    :param batch: 一个列表，包含了当前 batch_size 个 __getitem__ 返回的字典
    """
    ids = [item['id'] for item in batch]
    
    # 1. 处理 Audio
    audios = []
    for item in batch:
        audio = item['audio']
        # torchaudio 读取的通常是 [Channel, Length]，例如 [1, 345]
        # pad_sequence 默认沿着第0维(时间维度)对齐，所以需要先转置为 [Length, Channel]
        if audio.ndim == 2:
            audio = audio.transpose(0, 1) 
        audios.append(audio)
    
    # batch_first=True 让输出变成 [Batch, Length, Channel]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    
    # 如果原先是 2D 的 [Channel, Length]，pad 完再转回 [Batch, Channel, Length]
    if batch[0]['audio'].ndim == 2:
        audios_padded = audios_padded.transpose(1, 2)

    # 2. 处理 Lip
    lips = [item['lip'] for item in batch]
    # 假设 lip 是 [Length] 或 [Length, Feature]
    lips_padded = pad_sequence(lips, batch_first=True, padding_value=0.0)
    
    # 维度变化: [4, 348] -> [4, 1, 348]
    lips_padded = lips_padded.unsqueeze(1)
    
    # 3. 处理 Vocal
    vocals = [item['vocal'] for item in batch]
    # vocals_padded = pad_sequence(vocals, batch_first=True, padding_value=0.0)
    vocals_padded = torch.stack(vocals, dim=0)
    
    audio_lengths = torch.tensor([item['audio'].shape[-1] for item in batch])
    lip_lengths = torch.tensor([item['lip'].shape[0] for item in batch])
    
    return {
        'id': ids,
        'audio': audios_padded,
        'audio_lengths': audio_lengths, # <--- 方便模型生成 Attention Mask
        'lip': lips_padded,
        'lip_lengths': lip_lengths,
        'vocal': vocals_padded
    }
    

class mDatasetBase(Dataset):
    def __init__(self, root_dir):
        """
        多模态数据集的基础类
        :param root_dir: 数据集根目录，例如 'dataset/train/'
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, 'Audio')
        self.lip_dir = os.path.join(root_dir, 'mmLip')
        self.vocal_dir = os.path.join(root_dir, 'mmVocal')
        
        self.data_infos = []
        self._parse_dataset()

    def _parse_dataset(self):
        """
        遍历文件夹，通过文件名匹配三种模态的数据，过滤不完整的样本。
        """
        if not os.path.exists(self.audio_dir):
            raise ValueError(f"路径不存在: {self.audio_dir}")
            
        
        # 我们以 Audio 文件夹的文件列表作为基准进行匹配
        for audio_filename in os.listdir(self.audio_dir):
            if not audio_filename.endswith('.wav'):
                continue
                
            # 样本解析：audio_20260411_183325_s9.wav
            # 剥除 .wav 后按 '_' 分割 -> ['audio', '20260411', '183325', 's9']
            name_parts = audio_filename.replace('.wav', '').split('_')
            if len(name_parts) < 4:
                continue
                
            date_str = name_parts[1]   # 20260411
            time_str = name_parts[2]   # 183325
            suffix = name_parts[3]     # s9
            
            # 唯一时间串标识，例如：20260411_183325
            time_id = f"{date_str}_{time_str}"
            
            if time_id in INVALID_GROUPS:
                print(f"[Skip Invalid Group] {time_id}_{suffix}")
                continue
            
            # 按照命名规则推导出其他两个模态应有的文件名
            lip_filename = f"mmW_{time_id}_Lip_{suffix}.npy"
            vocal_filename = f"mmW_{time_id}_Vib_{suffix}.npy"
            
            audio_path = os.path.join(self.audio_dir, audio_filename)
            lip_path = os.path.join(self.lip_dir, lip_filename)
            vocal_path = os.path.join(self.vocal_dir, vocal_filename)
            
            # 检查三种模态的文件是否齐备
            if os.path.exists(lip_path) and os.path.exists(vocal_path):
                self.data_infos.append({
                    'id': f"{time_id}_{suffix}",
                    'group_id': time_id,
                    'suffix': suffix,
                    'audio_path': audio_path,
                    'lip_path': lip_path,
                    'vocal_path': vocal_path
                })
            else:
                print(f"警告: 样本 {time_id}_{suffix} 数据缺失，已跳过。")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data_infos)

    def __getitem__(self, idx):

        sample_info = self.data_infos[idx]

        # =====================================
        # Audio
        # =====================================
        waveform, sample_rate = torchaudio.load(
            sample_info['audio_path']
        )

        # =====================================
        # Lip
        # =====================================
        lip_data = np.load(sample_info['lip_path'])
        lip_data = np.asarray(lip_data, dtype=np.float32)

        if not np.all(np.isfinite(lip_data)):
            raise ValueError(
                f"[Invalid Lip] {sample_info['id']}"
            )

        lip_tensor = torch.from_numpy(lip_data)

        # =====================================
        # Vocal
        # =====================================
        vocal_data = np.load(sample_info['vocal_path'])

        # complex -> magnitude
        if np.iscomplexobj(vocal_data):
            vocal_data = np.abs(vocal_data)

        vocal_data = np.asarray(
            vocal_data,
            dtype=np.float32
        )

        if not np.all(np.isfinite(vocal_data)):
            raise ValueError(
                f"[Invalid Vocal] {sample_info['id']}"
            )

        if np.std(vocal_data) < 1e-12:
            raise ValueError(
                f"[Zero/constant Vocal] {sample_info['id']}"
            )

        vocal_tensor = torch.from_numpy(vocal_data)

        return {
            'id': sample_info['id'],
            'group_id': sample_info['group_id'],
            'audio': waveform,
            'lip': lip_tensor,
            'vocal': vocal_tensor
        }

# ==========================================
# DataLoader 的实例化与测试流程
# ==========================================
if __name__ == '__main__':
    # 1. 定义数据根目录并实例化 Dataset
    # 假设当前脚本运行环境在 dataset 目录的同级
    dataset_path = '/data2/fanl/M2Voice/dataset/dt3_splitv2/train' 
    
    train_dataset = mDatasetBase(root_dir=dataset_path)

    print(f"\n数据集解析完成，共 {len(train_dataset)} 个样本。")

    invalid_samples = []

    lip_shapes = {}
    vocal_shapes = {}

    print("\n========== Start Dataset Check ==========")

    for idx, info in enumerate(train_dataset.data_infos):

        lip = np.load(info['lip_path'])
        vocal = np.load(info['vocal_path'])

        sample_id = info['group_id']

        # ============================================
        # 1. Shape
        # ============================================

        lip_shapes[lip.shape] = lip_shapes.get(lip.shape, 0) + 1
        vocal_shapes[vocal.shape] = vocal_shapes.get(vocal.shape, 0) + 1

        # ============================================
        # 2. Complex
        # ============================================

        vocal_is_complex = np.iscomplexobj(vocal)

        # ============================================
        # 3. NaN / Inf
        # ============================================

        lip_nan = np.isnan(lip).sum()
        lip_inf = np.isinf(lip).sum()

        vocal_nan = np.isnan(vocal).sum()
        vocal_inf = np.isinf(vocal).sum()

        # ============================================
        # 4. Statistics
        # ============================================

        if np.iscomplexobj(vocal):
            vocal_check = np.abs(vocal)
        else:
            vocal_check = vocal

        lip_min = np.nanmin(lip)
        lip_max = np.nanmax(lip)
        lip_mean = np.nanmean(lip)
        lip_std = np.nanstd(lip)

        vocal_min = np.nanmin(vocal_check)
        vocal_max = np.nanmax(vocal_check)
        vocal_mean = np.nanmean(vocal_check)
        vocal_std = np.nanstd(vocal_check)

        # ============================================
        # 5. Invalid
        # ============================================

        invalid = (
            lip_nan > 0
            or lip_inf > 0
            or vocal_nan > 0
            or vocal_inf > 0
            or not np.isfinite(lip_std)
            or not np.isfinite(vocal_std)
            
            # Lip无有效变化
            or lip_std < 1e-12

            # Vocal无有效信号
            or vocal_std < 1e-12
            or np.max(np.abs(vocal_check)) < 1e-12
        )

        if invalid:

            print("\n===================================")
            print(f"[INVALID SAMPLE] idx={idx}")
            print(f"ID: {sample_id}")

            print("\nLip:")
            print(" path :", info['lip_path'])
            print(" shape:", lip.shape)
            print(" dtype:", lip.dtype)
            print(" min  :", lip_min)
            print(" max  :", lip_max)
            print(" mean :", lip_mean)
            print(" std  :", lip_std)
            print(" NaN  :", lip_nan)
            print(" Inf  :", lip_inf)

            print("\nVocal:")
            print(" path :", info['vocal_path'])
            print(" shape:", vocal.shape)
            print(" dtype:", vocal.dtype)
            print(" complex:", vocal_is_complex)
            print(" min  :", vocal_min)
            print(" max  :", vocal_max)
            print(" mean :", vocal_mean)
            print(" std  :", vocal_std)
            print(" NaN  :", vocal_nan)
            print(" Inf  :", vocal_inf)

            invalid_samples.append(sample_id)

    print("\n\n========== Dataset Summary ==========")

    print("\nLip Shapes:")
    for shape, count in sorted(lip_shapes.items(), key=lambda x: str(x[0])):
        print(f"{shape}: {count}")

    print("\nVocal Shapes:")
    for shape, count in sorted(vocal_shapes.items(), key=lambda x: str(x[0])):
        print(f"{shape}: {count}")

    print("\nInvalid samples:", len(invalid_samples))

    if invalid_samples:
        for sample_id in invalid_samples:
            print(sample_id)
    else:
        print("All samples are finite.")