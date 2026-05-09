import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMMultiStep(nn.Module):
    """
    经典 LSTM 直接多步输出模型，可选 LayerNorm。
    (代码保持原样，未改动)
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=144,
                 num_layers=2, dropout=0.2, use_layer_norm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        assert x.dim() == 3, f"[Shape Error] LSTM 期望输入维度为3D, 但接收到了 {x.dim()}D。"
        assert x.size(2) == self.input_size, f"[Shape Error] LSTM 期望的特征维度 input_size={self.input_size}, 但实际为 {x.size(2)}。"

        lstm_out, _ = self.lstm(x)          
        last_out = lstm_out[:, -1, :]        

        if self.use_layer_norm:
            last_out = self.layer_norm(last_out)

        last_out = self.dropout(last_out)
        out = self.fc(last_out)              
        return out


class Attention(nn.Module):
    """
    🎯 新增：Bahdanau Attention (加性注意力机制)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: Decoder当前的隐藏状态 [num_layers, batch, hidden_size]
        # encoder_outputs: Encoder所有时间步的输出 [batch, seq_len, hidden_size]
        
        # 我们取 LSTM 最后一层的隐藏状态来计算注意力
        last_layer_hidden = hidden[-1].unsqueeze(1) # [batch, 1, hidden_size]
        seq_len = encoder_outputs.size(1)
        
        # 将 hidden 扩展到和 seq_len 一样的长度，方便拼接
        hidden_expanded = last_layer_hidden.repeat(1, seq_len, 1) # [batch, seq_len, hidden_size]
        
        # 计算能量值 (Energy)
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2))) # [batch, seq_len, hidden_size]
        
        # 计算注意力权重
        attention = self.v(energy).squeeze(2) # [batch, seq_len]
        return F.softmax(attention, dim=1)


class Seq2SeqLSTM(nn.Module):
    """
    Encoder-Decoder LSTM 结合 Attention 机制，逐步预测未来序列。
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=144, 
                 output_feature_size=1, num_layers=2, dropout=0.2, teacher_forcing_ratio=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_feature_size = output_feature_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 🎯 实例化 Attention 模块
        self.attention = Attention(hidden_size)

        # 🎯 核心修改：因为引入了 Context Vector (维度为 hidden_size)
        # 现在的 Decoder 每次接收的输入不仅是“上一步的预测值 (input_size)”，还要拼上“上下文向量 (hidden_size)”
        self.decoder = nn.LSTM(
            input_size=input_size + hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None):
        assert x.dim() == 3, f"[Shape Error] Seq2Seq Encoder 期望输入为3D, 但收到 {x.dim()}D。"
        assert x.size(2) == self.input_size, f"[Shape Error] Seq2Seq 输入特征维度错误，期望 {self.input_size}, 实际 {x.size(2)}。"

        if target is not None:
            if target.dim() == 2:
                target = target.unsqueeze(2) 

        # Encoder 前向传播
        # 注意：这里我们同时保留了 encoder_outputs（用于注意力计算）和 (hidden, cell)（用于初始化Decoder）
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Decoder 初始输入为输入序列最后一个真实值
        decoder_input = x[:, -1:, :]  # [batch, 1, input_size]

        outputs = []
        for t in range(self.output_size):
            # 🎯 1. 计算注意力权重 a: [batch, seq_len]
            a = self.attention(hidden, encoder_outputs)
            a = a.unsqueeze(1) # 变形为 [batch, 1, seq_len] 方便矩阵乘法
            
            # 🎯 2. 计算上下文向量 context_vector: [batch, 1, hidden_size]
            # 用注意力权重去对 Encoder 的所有输出进行加权求和
            context_vector = torch.bmm(a, encoder_outputs)
            
            # 🎯 3. 将当前输入和上下文向量拼接: [batch, 1, input_size + hidden_size]
            rnn_input = torch.cat((decoder_input, context_vector), dim=2)
            
            # 4. Decoder 步进
            out, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            out = self.dropout(out)
            pred = self.fc(out)  # [batch, 1, output_feature_size]
            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred

        # 拼接所有时间步输出
        out = torch.cat(outputs, dim=1)  # [batch, output_size, output_feature_size]
        
        # 兼容外部单变量评估
        if self.output_feature_size == 1:
            return out.squeeze(-1)       # [batch, output_size]
        return out