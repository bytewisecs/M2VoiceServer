import torch
import torch.nn as nn
import ml_collections
from torch.nn import Linear, Dropout, Softmax, LayerNorm
import math
import copy

# ==========================================
# 1. 基础 Transformer 模块 (按依赖顺序定义)
# ==========================================
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.fusion["num_heads"]
        self.attention_head_size = int(config.fusion["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.fusion["hidden_size"], self.all_head_size)
        self.key = Linear(config.fusion["hidden_size"], self.all_head_size)
        self.value = Linear(config.fusion["hidden_size"], self.all_head_size)
        self.out = Linear(config.fusion["hidden_size"], config.fusion["hidden_size"])
        self.attn_dropout = Dropout(config.fusion["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.fusion["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.fusion["hidden_size"], config.fusion["mlp_dim"])
        self.fc2 = Linear(config.fusion["mlp_dim"], config.fusion["hidden_size"])
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.fusion["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.fusion["hidden_size"]
        self.attention_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        self.ffn_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h 

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h 
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        for _ in range(config.fusion["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

# ==========================================
# 2. 适配 1D 序列的 Embedding 模块
# ==========================================
class Conv1dEmbeddings(nn.Module):
    '''
    对 1D 时序数据按时间维度分帧：把序列分割成 Patch
    '''
    def __init__(self, config, in_channels, patch_size, max_seq_len=2000):
        super(Conv1dEmbeddings, self).__init__()
        self.patch_size = patch_size
        
        # 使用 Conv1d 对序列进行 Patch 提取和降维/升维
        self.patch_embeddings_ = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.fusion["hidden_size"],
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 可学习的位置编码，预设一个足够长的最大长度
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_seq_len, config.fusion["hidden_size"])
        )
        self.dropout = nn.Dropout(config.fusion["dropout_rate"])

    def forward(self, x):
        # x shape: [B, C, L] 
        x = self.patch_embeddings_(x)          # -> [B, hidden_size, N_patches]
        x = x.transpose(-1, -2)                # -> [B, N_patches, hidden_size]
        
        # 动态适配 DataLoader 产生的变长序列：获取当前分块后的真实序列长度
        seq_len = x.shape[1]
        
        # 截取对应长度的位置编码并相加
        embeddings = x + self.position_embeddings[:, :seq_len, :]
        embeddings = self.dropout(embeddings)
        
        return embeddings # (bs, n_patch, hidden_size)


# ==========================================
# 3. 三模态特征编码模块 (Three-Stream Encoder)
# ==========================================
class MultiModalFeatureEncoder(nn.Module):
    def __init__(self, config, vis=False):
        super(MultiModalFeatureEncoder, self).__init__()
        
        # --- A. 实例化三个独立的 Embedding 层 ---
        # 1. Audio: 输入维度 [B, 1, ~240000]。
        self.audio_embed = Conv1dEmbeddings(config, in_channels=1, patch_size=400, max_seq_len=1500)
        
        # 2. Lip: 输入维度 [B, 1, ~348]。
        self.lip_embed = Conv1dEmbeddings(config, in_channels=1, patch_size=2, max_seq_len=500)
        
        # 3. Vocal: 输入维度 [B, 256, ~939]。
        self.vocal_embed = Conv1dEmbeddings(config, in_channels=256, patch_size=4, max_seq_len=1000)

        # --- B. 实例化三个独立的 Transformer Encoder ---
        self.audio_encoder = Encoder(config, vis)
        self.lip_encoder = Encoder(config, vis)
        self.vocal_encoder = Encoder(config, vis)
        
    def forward(self, audio, lip, vocal):
        # 1. Audio 流向前传播
        emb_audio = self.audio_embed(audio)
        encoded_audio, weights_audio = self.audio_encoder(emb_audio)
        
        # 2. Lip 流向前传播
        emb_lip = self.lip_embed(lip)
        encoded_lip, weights_lip = self.lip_encoder(emb_lip)
        
        # 3. Vocal 流向前传播
        emb_vocal = self.vocal_embed(vocal)
        encoded_vocal, weights_vocal = self.vocal_encoder(emb_vocal)
        
        # 返回编码后的特征序列和注意力权重
        return (encoded_audio, encoded_lip, encoded_vocal), (weights_audio, weights_lip, weights_vocal)

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    def get_config():
        config = ml_collections.ConfigDict()
        config.fusion = ml_collections.ConfigDict()
        config.fusion.hidden_size = 384
        config.fusion.dropout_rate = 0.1        
        config.fusion.mlp_dim = 3072
        config.fusion.num_heads = 6
        config.fusion.num_layers = 4  # 为加快测试速度，这里设为 4 层
        config.fusion.attention_dropout_rate = 0.0
        return config

    config = get_config()
    
    # 模拟从你 DataLoader 吐出的数据维度 (Batch_size = 4)
    print("正在初始化模拟输入...")
    mock_audio = torch.rand(4, 1, 240000)
    mock_lip = torch.rand(4, 1, 348)
    mock_vocal = torch.rand(4, 256, 939)
    
    # 实例化三模态编码器
    print("正在实例化 MultiModalFeatureEncoder...")
    model = MultiModalFeatureEncoder(config, vis=True)
    
    # 前向传播
    print("执行前向传播...")
    encoded_features, attn_weights = model(mock_audio, mock_lip, mock_vocal)
    
    print("\n✅ 测试成功！模型已通畅。")
    print("--- 各模态编码输出维度 ---")
    print(f"Audio Encoded Shape: {encoded_features[0].shape}") 
    print(f"Lip Encoded Shape:   {encoded_features[1].shape}") 
    print(f"Vocal Encoded Shape: {encoded_features[2].shape}")