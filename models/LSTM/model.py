import torch
import torch.nn as nn

class LSTMMultiStep(nn.Module):
    """
    经典 LSTM 直接多步输出模型，可选 LayerNorm。
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
        # 【白盒化】维度断言：确保输入 Tensor 形状严格为 [batch, seq_len, input_size]
        assert x.dim() == 3, f"[Shape Error] LSTM 期望输入维度为3D, 但接收到了 {x.dim()}D。"
        assert x.size(2) == self.input_size, f"[Shape Error] LSTM 期望的特征维度 input_size={self.input_size}, 但实际为 {x.size(2)}。"

        lstm_out, _ = self.lstm(x)          # [batch, seq_len, hidden]
        last_out = lstm_out[:, -1, :]        # 取最后一个时间步 [batch, hidden]

        if self.use_layer_norm:
            last_out = self.layer_norm(last_out)

        last_out = self.dropout(last_out)
        out = self.fc(last_out)              # [batch, output_size]
        return out


class Seq2SeqLSTM(nn.Module):
    """
    Encoder-Decoder LSTM，逐步预测未来序列。
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

        self.decoder = nn.LSTM(
            input_size=input_size, # Decoder接收的输入应与Encoder一致(上一时刻的预测值)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解除原代码中对 1 的硬编码，改为可配置的输出特征维度
        self.fc = nn.Linear(hidden_size, output_feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None):
        """
        训练时可通过 target 进行 Teacher Forcing。
        x: [batch, seq_len, input_size]
        """
        # 【白盒化】针对 x 维度的断言
        assert x.dim() == 3, f"[Shape Error] Seq2Seq Encoder 期望输入为3D, 但收到 {x.dim()}D。"
        assert x.size(2) == self.input_size, f"[Shape Error] Seq2Seq 输入特征维度错误，期望 {self.input_size}, 实际 {x.size(2)}。"

        batch_size = x.size(0)
        
        # 【白盒化】针对 target 维度的预处理与断言
        if target is not None:
            if target.dim() == 2:
                target = target.unsqueeze(2) # 补充特征维度变为 [batch, output_size, 1]
            assert target.dim() == 3, f"[Shape Error] target 应该被转换为3D，但当前为 {target.dim()}D。"
            assert target.size(1) == self.output_size, f"[Shape Error] target 时间步长 {target.size(1)} 与 output_size {self.output_size} 不匹配。"
            assert target.size(2) == self.output_feature_size, f"[Shape Error] target 特征维度 {target.size(2)} 与模型设定的 {self.output_feature_size} 不匹配。"

        # Encoder
        _, (hidden, cell) = self.encoder(x)

        # Decoder 初始输入为输入序列最后一个值
        decoder_input = x[:, -1:, :]  # [batch, 1, input_size]

        outputs = []
        for t in range(self.output_size):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
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
        
        # 如果是单变量预测，主动 squeeze 掉最后一维，保持与外部评估兼容
        if self.output_feature_size == 1:
            return out.squeeze(-1)       # [batch, output_size]
        return out