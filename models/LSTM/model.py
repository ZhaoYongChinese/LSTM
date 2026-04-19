import torch
import torch.nn as nn


class LSTMMultiStep(nn.Module):
    """
    经典 LSTM 直接多步输出模型，可选 LayerNorm。
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=144,
                 num_layers=2, dropout=0.2, use_layer_norm=False):
        super().__init__()
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
        # x: [batch, seq_len, 1]
        lstm_out, _ = self.lstm(x)          # [batch, seq_len, hidden]
        last_out = lstm_out[:, -1, :]        # [batch, hidden]

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
                 num_layers=2, dropout=0.2, teacher_forcing_ratio=0.5):
        super().__init__()
        self.output_size = output_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None):
        """
        训练时可通过 target 进行 Teacher Forcing。
        x: [batch, seq_len, 1]
        target: [batch, output_size, 1] 或 [batch, output_size]（会自动reshape）
        """
        batch_size = x.size(0)
        # Encoder
        _, (hidden, cell) = self.encoder(x)

        # Decoder 初始输入为输入序列最后一个值
        decoder_input = x[:, -1:, :]  # [batch, 1, 1]

        outputs = []
        for t in range(self.output_size):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.dropout(out)
            pred = self.fc(out)  # [batch, 1, 1]
            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # 确保 target 形状正确
                if target.dim() == 2:
                    target_t = target[:, t].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                else:
                    target_t = target[:, t:t+1, :]
                decoder_input = target_t
            else:
                decoder_input = pred

        # 拼接所有时间步输出
        out = torch.cat(outputs, dim=1)  # [batch, output_size, 1]
        return out.squeeze(-1)           # [batch, output_size]