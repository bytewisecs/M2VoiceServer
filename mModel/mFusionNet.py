import torch
import torch.nn as nn
import ml_collections
from torch.nn import Linear, Dropout, Softmax, LayerNorm
import math
import copy

# ==========================================
# 1. 基础组件 (Embedding 与 Transformer 基础块)
# ==========================================
class Conv1dEmbeddings(nn.Module):
    def __init__(self, config, in_channels, patch_size, max_seq_len=2000):
        super(Conv1dEmbeddings, self).__init__()
        self.patch_size = patch_size
        self.patch_embeddings_ = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.fusion["hidden_size"],
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, config.fusion["hidden_size"]))
        self.dropout = nn.Dropout(config.fusion["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings_(x)
        x = x.transpose(-1, -2)
        seq_len = x.shape[1]
        embeddings = x + self.position_embeddings[:, :seq_len, :]
        return self.dropout(embeddings)

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

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        self.ffn_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x, _ = self.attn(self.attention_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x))
        return x + h, None

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([Block(config, vis) for _ in range(config.fusion["num_layers"])])
        self.encoder_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states, _ = layer_block(hidden_states)
        return self.encoder_norm(hidden_states), None

# ==========================================
# 2. 双模态融合核心网络 (Dual Attention)
# ==========================================
class Dual_Co_Attention(nn.Module):
    def __init__(self, config, vis):
        super(Dual_Co_Attention, self).__init__()
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

    def _cross_attend(self, q_layer, k_layer, v_layer):
        attention_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.attn_dropout(self.softmax(attention_scores))
        
        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return self.proj_dropout(self.out(context_layer))

    def forward(self, h_l, h_v):
        q_l, k_l, v_l = self.transpose_for_scores(self.query(h_l)), self.transpose_for_scores(self.key(h_l)), self.transpose_for_scores(self.value(h_l))
        q_v, k_v, v_v = self.transpose_for_scores(self.query(h_v)), self.transpose_for_scores(self.key(h_v)), self.transpose_for_scores(self.value(h_v))

        # 1. Lip Cross-Attend Vocal
        out_l = self._cross_attend(q_l, k_v, v_v)
        # 2. Vocal Cross-Attend Lip
        out_v = self._cross_attend(q_v, k_l, v_l)

        return out_l, out_v

class Dual_Merged_Attention(Dual_Co_Attention):
    def forward(self, h_l, h_v):
        q_l, k_l, v_l = self.transpose_for_scores(self.query(h_l)), self.transpose_for_scores(self.key(h_l)), self.transpose_for_scores(self.value(h_l))
        q_v, k_v, v_v = self.transpose_for_scores(self.query(h_v)), self.transpose_for_scores(self.key(h_v)), self.transpose_for_scores(self.value(h_v))

        # 1. 建立全局共享的 K 和 V (拼接 Lip, Vocal)
        share_k = torch.cat((k_l, k_v), dim=2)
        share_v = torch.cat((v_l, v_v), dim=2)

        # 2. 两个模态分别向全局池发起 Query
        out_l = self._cross_attend(q_l, share_k, share_v)
        out_v = self._cross_attend(q_v, share_k, share_v)

        # 3. 在序列长度维度拼接，输出一条融合长序列
        return torch.cat((out_l, out_v), dim=1)

class Dual_FusionBlock(nn.Module):
    def __init__(self, config, vis):
        super(Dual_FusionBlock, self).__init__()
        self.attention_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        self.ffn_norm = LayerNorm(config.fusion["hidden_size"], eps=1e-6)
        self.ffn = Mlp(config)
        
        self.Co_Attn = Dual_Co_Attention(config, vis)
        self.Merged_Attn = Dual_Merged_Attention(config, vis) 

    def forward(self, x_l, x_v):
        # 1. Co-Attention
        norm_l, norm_v = self.attention_norm(x_l), self.attention_norm(x_v)
        co_l, co_v = self.Co_Attn(norm_l, norm_v)
        
        co_l, co_v = co_l + x_l, co_v + x_v
    
        # 2. Merged Attention
        merged_h = torch.cat((co_l, co_v), dim=1) 
        norm_co_l, norm_co_v = self.attention_norm(co_l), self.attention_norm(co_v)
        
        merged_x = self.Merged_Attn(norm_co_l, norm_co_v)
        merged_x = merged_x + merged_h

        # 3. FFN
        h_mlp = merged_x
        merged_x = self.ffn(self.ffn_norm(merged_x))
        return merged_x + h_mlp


class CrossModalAlignment_Model(nn.Module):
    def __init__(self, config, vis=False):
        super(CrossModalAlignment_Model, self).__init__()
        
        # 1. 教师分支 (Audio Ground Truth 提取)
        self.audio_embed = Conv1dEmbeddings(config, in_channels=1, patch_size=400, max_seq_len=1500)
        self.audio_encoder = Encoder(config, vis)
        
        # 2. 学生分支 (Lip + Vocal 特征提取)
        self.lip_embed = Conv1dEmbeddings(config, in_channels=1, patch_size=2, max_seq_len=500)
        self.vocal_embed = Conv1dEmbeddings(config, in_channels=256, patch_size=4, max_seq_len=1000)
        self.lip_encoder = Encoder(config, vis)
        self.vocal_encoder = Encoder(config, vis)
        
        # 3. 双模态融合
        self.fusion_block = Dual_FusionBlock(config, vis)

        # 取消了 cls_token, recognization_encoder 和 head_

    def forward(self, lip, vocal, audio):
        # ==========================================
        # 教师分支 (Teacher): 获取 Audio GT 特征
        # ==========================================
        emb_audio = self.audio_embed(audio)
        audio_seq, _ = self.audio_encoder(emb_audio) # -> [B, L_audio, Hidden]
        
        # ==========================================
        # 学生分支 (Student): 获取 Lip + Vocal 融合特征
        # ==========================================
        emb_lip = self.lip_embed(lip)
        encoded_lip, _ = self.lip_encoder(emb_lip)
        
        emb_vocal = self.vocal_embed(vocal)
        encoded_vocal, _ = self.vocal_encoder(emb_vocal)
        
        # 双模态融合
        fused_seq = self.fusion_block(encoded_lip, encoded_vocal) # -> [B, L_lip + L_vocal, Hidden]
        
        # 直接返回两条特征序列，交由外层的 Loss 函数去处理
        return fused_seq, audio_seq
 
# ==========================================
# 4. 测试代码
# ==========================================
if __name__=="__main__":
    def get_config():
        config = ml_collections.ConfigDict()
        config.fusion = ml_collections.ConfigDict()
        config.fusion.hidden_size = 384
        config.fusion.dropout_rate = 0.1        
        config.fusion.mlp_dim = 3072
        config.fusion.num_heads = 6
        config.fusion.num_layers = 2
        config.fusion.attention_dropout_rate = 0.2
        return config
    
    config = get_config()
    
    # 1. 模拟 DataLoader 吐出的三模态数据 [Batch=4]
    mock_audio = torch.rand(4, 1, 240000)
    mock_lip = torch.rand(4, 1, 348)
    mock_vocal = torch.rand(4, 256, 939)
    
    # 2. 实例化对齐模型
    model = CrossModalAlignment_Model(config, vis=False)
    
    # 定义优化器和损失函数 (MSE Loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_criterion = nn.MSELoss()
    
    # ==========================================
    # 前向传播与 Loss 计算演示
    # ==========================================
    optimizer.zero_grad()
    
    print("正在执行前向传播...")
    fused_seq, audio_seq = model(mock_lip, mock_vocal, mock_audio)
    print(f"  融合序列维度: {fused_seq.shape}") # [4, 408, 384]
    print(f"  音频序列维度: {audio_seq.shape}") # [4, 600, 384]
    
    # 核心步骤：序列长度对齐 (在第 1 维/时间维 度上求平均)
    # [Batch, Length, Hidden] -> [Batch, Hidden]
    fused_pooled = fused_seq.mean(dim=1) 
    audio_pooled = audio_seq.mean(dim=1)
    
    print(f"  Pooling 后融合特征维度: {fused_pooled.shape}") # [4, 384]
    print(f"  Pooling 后音频特征维度: {audio_pooled.shape}") # [4, 384]
    
    # 计算误差 Loss
    loss = mse_criterion(fused_pooled, audio_pooled)
    print(f"\n✅ 计算得到的 MSE Loss: {loss.item():.4f}")
    
    # 反向传播 (更新模型参数)
    loss.backward()
    optimizer.step()