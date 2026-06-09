import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMMultiStep(nn.Module):
    """
    经典 LSTM 直接多步输出模型，可选 LayerNorm。
    🆕 v2: 支持残差跳跃连接 — 模型学习预测 Δ = tomorrow - today，
           最终输出 = today + Δ，天然利用天与天之间的强相似性。
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=144,
                 num_layers=2, dropout=0.2, use_layer_norm=False,
                 use_residual=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size        # 🆕 保存供 forward 使用
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual      # 🆕 残差模式开关

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
        out = self.fc(last_out)  # [batch, output_size]

        # 🆕 残差跳跃连接: 预测值 = 今天值 + 模型预测的 Δ
        # 输入 x[:, :, 0] 是 RMS 通道（已统一归一化），取最后 output_size 步作为"今天"
        if self.use_residual:
            today_rms = x[:, -self.output_size:, 0]  # [batch, output_size]
            out = today_rms + out

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
    v2:
      - 修复 Decoder 输入维度 bug(原代码在 input_size != output_feature_size 时崩溃)
      - 支持残差跳跃连接 - 每步预测 = 今天对应时刻 + Delta
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=144,
                 output_feature_size=1, num_layers=2, dropout=0.2, teacher_forcing_ratio=0.5,
                 use_residual=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_feature_size = output_feature_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_residual = use_residual      # 🆕 残差模式开关

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 🎯 Attention 模块
        self.attention = Attention(hidden_size)

        # 🆕 修复: Decoder 的 input_size 应该是 output_feature_size + hidden_size
        # （Decoder 输入 = 上一步预测值 + context vector），而非 input_size + hidden_size
        # 原 bug: 当 input_size(带时间特征)=3 而 output_feature_size=1 时，
        #   第一步 decoder_input shape [b,1,3] 和后续 pred shape [b,1,1] 不一致导致崩溃
        self.decoder = nn.LSTM(
            input_size=output_feature_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None):
        assert x.dim() == 3, f"[Shape Error] Seq2Seq Encoder 期望输入为3D, 但收到 {x.dim()}D."
        assert x.size(2) == self.input_size, f"[Shape Error] Seq2Seq 输入特征维度错误, 期望 {self.input_size}, 实际 {x.size(2)}."

        if target is not None:
            if target.dim() == 2:
                target = target.unsqueeze(2)  # [batch, pred_len, 1]

        # Encoder 前向传播
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # 🆕 修复: Decoder 初始输入只取 RMS 通道 (channel 0)，而非全部 input_size 通道
        # 原因: 时间特征已通过 encoder 充分编码，decoder 只需 RMS 值作为自回归起点
        decoder_input = x[:, -1:, :self.output_feature_size]  # [batch, 1, output_feature_size]

        # 残差基线: 输入中今天的 RMS 值 (最后 output_size 步)
        if self.use_residual:
            today_rms = x[:, -self.output_size:, 0]  # [batch, output_size]

        outputs = []
        for t in range(self.output_size):
            # 1. 计算注意力权重 a: [batch, seq_len]
            a = self.attention(hidden, encoder_outputs)
            a = a.unsqueeze(1)  # [batch, 1, seq_len]

            # 2. 计算上下文向量 context_vector: [batch, 1, hidden_size]
            context_vector = torch.bmm(a, encoder_outputs)

            # 3. 拼接当前输入和上下文向量: [batch, 1, output_feature_size + hidden_size]
            rnn_input = torch.cat((decoder_input, context_vector), dim=2)

            # 4. Decoder 步进
            out, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            out = self.dropout(out)
            pred = self.fc(out)  # [batch, 1, output_feature_size]

            # 🆕 残差跳跃: pred = today_rms[t] + Δ
            if self.use_residual:
                pred = pred + today_rms[:, t:t+1].unsqueeze(1)

            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # 只取 target 的 RMS 值 (第一个特征) 作为下一步输入
                decoder_input = target[:, t:t+1, :self.output_feature_size]
            else:
                decoder_input = pred

        # 拼接所有时间步输出
        out = torch.cat(outputs, dim=1)  # [batch, output_size, output_feature_size]

        # 兼容外部单变量评估
        if self.output_feature_size == 1:
            return out.squeeze(-1)       # [batch, output_size]
        return out