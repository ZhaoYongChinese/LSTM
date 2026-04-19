import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler

from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM


def load_model(checkpoint_path, device='cpu', default_seq_len=None, default_output_size=None):
    """
    加载训练好的模型、归一化器及相关参数。
    自动检测模型类型与层数，兼容各种旧版checkpoint。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = checkpoint.get('params', {})
    state_dict = checkpoint['model_state_dict']

    # ---------- 1. 自动检测模型类型 ----------
    if 'encoder.weight_ih_l0' in state_dict or 'decoder.weight_ih_l0' in state_dict:
        model_type = 'Seq2SeqLSTM'
    else:
        model_type = 'LSTM'
    print(f"检测到模型类型: {model_type}")

    # ---------- 2. 推断 hidden_size ----------
    hidden_size = params.get('hidden_size') or params.get('hidden')
    if hidden_size is None:
        if 'lstm.weight_ih_l0' in state_dict:
            hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        elif 'encoder.weight_ih_l0' in state_dict:
            hidden_size = state_dict['encoder.weight_ih_l0'].shape[0] // 4
        else:
            hidden_size = 64

    # ---------- 3. 精确推断 num_layers（通过最大层索引）----------
    num_layers = params.get('num_layers') or params.get('layers')
    if num_layers is None:
        max_idx = -1
        for key in state_dict.keys():
            if '_l' in key:
                parts = key.split('_l')
                if len(parts) > 1:
                    num_str = ''
                    for ch in parts[1]:
                        if ch.isdigit():
                            num_str += ch
                        else:
                            break
                    if num_str:
                        idx = int(num_str)
                        if idx > max_idx:
                            max_idx = idx
        num_layers = max_idx + 1 if max_idx >= 0 else 1

    # ---------- 4. dropout ----------
    dropout = params.get('dropout') or params.get('drop') or 0.2

    # ---------- 5. seq_len ----------
    seq_len = params.get('seq_len') or params.get('seq_length') or default_seq_len
    if seq_len is None:
        raise ValueError("模型文件中缺少 seq_len，请在 test.py 中设置 DEFAULT_SEQ_LENGTH。")

    # ---------- 6. output_size ----------
    # 对于 Seq2Seq，输出长度无法从权重推断，必须从 params 获取或使用默认值
    output_size = params.get('output_size') or params.get('pred_len') or default_output_size
    if output_size is None:
        # 仅对 Vanilla LSTM 尝试从 fc/linear 权重推断
        if model_type == 'LSTM':
            if 'fc.weight' in state_dict:
                output_size = state_dict['fc.weight'].shape[0]
            elif 'linear.weight' in state_dict:
                output_size = state_dict['linear.weight'].shape[0]
        if output_size is None:
            raise ValueError("模型文件中缺少 output_size，请在 test.py 中设置 DEFAULT_OUTPUT_SIZE。")

    # ---------- 6. 实例化模型 ----------
    if model_type == 'Seq2SeqLSTM':
        model = Seq2SeqLSTM(
            input_size=1,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        # 推断是否使用 LayerNorm（优先级：params > state_dict 检测 > 默认值）
        use_layer_norm = params.get('use_layer_norm')
        if use_layer_norm is None:
            # 如果 state_dict 中有 "layer_norm" 开头的键，说明启用了 LayerNorm
            use_layer_norm = any(k.startswith('layer_norm') for k in state_dict.keys())
        model = LSTMMultiStep(
            input_size=1,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )

    # ---------- 8. 加载权重 ----------
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    print(f"✅ 模型加载成功：{model_type} | hidden={hidden_size}, layers={num_layers}, seq_len={seq_len}, out_len={output_size}")
    return model, scaler_X, scaler_y, seq_len, output_size


def compute_mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def predict_sequence(model, scaler_X, scaler_y, input_seq, device='cpu'):
    input_norm = scaler_X.transform(input_seq.reshape(-1, 1)).reshape(1, -1, 1)
    input_tensor = torch.FloatTensor(input_norm).to(device)
    with torch.no_grad():
        pred_norm = model(input_tensor)  # 形状为 (1, output_size)
    pred_np = pred_norm.cpu().numpy()
    # 确保形状为 (1, output_size)
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(1, -1)
    pred = scaler_y.inverse_transform(pred_np).flatten()
    return pred


def create_animation(model, scaler_X, scaler_y, seq_len, output_size,
                     data_series, save_path='prediction.gif', fps=2):
    device = next(model.parameters()).device
    total_len = len(data_series)
    max_start_idx = total_len - seq_len - output_size

    if max_start_idx < 0:
        raise ValueError(f"数据长度 {total_len} 不足，至少需要 {seq_len + output_size} 个点")

    all_true_accum = []
    all_pred_accum = []
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.85)
    overall_acc = 0.0

    def animate(i):
        nonlocal overall_acc
        ax.clear()
        start_idx = i
        hist_true = data_series[start_idx : start_idx + seq_len]
        future_true = data_series[start_idx + seq_len : start_idx + seq_len + output_size]
        pred = predict_sequence(model, scaler_X, scaler_y, hist_true, device)

        all_true_accum.append(future_true)
        all_pred_accum.append(pred)

        window_mape = compute_mape(future_true, pred)
        window_acc = 100 - window_mape

        overall_true = np.concatenate(all_true_accum)
        overall_pred = np.concatenate(all_pred_accum)
        overall_mape = compute_mape(overall_true, overall_pred)
        overall_acc = 100 - overall_mape

        x_hist = np.arange(start_idx, start_idx + seq_len)
        x_future = np.arange(start_idx + seq_len, start_idx + seq_len + output_size)
        x_all = np.arange(start_idx, start_idx + seq_len + output_size)
        y_all_true = np.concatenate([hist_true, future_true])

        ax.plot(x_all, y_all_true, 'b-', label='Actual', linewidth=2)
        ax.plot(x_future, pred, 'r--', label='Predicted', linewidth=2)
        ax.axvline(x=start_idx + seq_len - 1, color='gray', linestyle=':', alpha=0.7)

        ax.legend(loc='upper left')
        ax.set_xlabel('Time Step (index)')
        # ax.set_ylabel('Value')
        # ax.set_title(f'Window {i+1}/{max_start_idx+1} (Start index = {start_idx})')
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.95, f'Window Acc: {window_acc:.2f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 1.02, f'Overall Accuracy: {overall_acc:.2f}%',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center',
                color='green')

    ani = animation.FuncAnimation(fig, animate, frames=max_start_idx+1,
                                  interval=1000//fps, repeat=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    print(f"动画已保存至: {save_path}")

    final_acc = overall_acc if all_true_accum else 0.0
    return final_acc


def main():
    # ==================== 用户配置区域 ====================
    MODEL_PATH = r"汇报/测试/LSTM_h32_l2_drop0.5_lr0.0005_mse_mape1.29.pth"
    TEST_CSV = r"data/show/测试.csv"
    TARGET_COLUMN = 'RMS_Value'
    OUTPUT_GIF = "prediction_animation.gif"
    FPS = 2

    # 旧模型后备参数（当checkpoint中缺失时使用）
    DEFAULT_SEQ_LENGTH = 144
    DEFAULT_OUTPUT_SIZE = 72
    # =====================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("加载模型中...")
    model, scaler_X, scaler_y, seq_len, output_size = load_model(
        MODEL_PATH, device,
        default_seq_len=DEFAULT_SEQ_LENGTH,
        default_output_size=DEFAULT_OUTPUT_SIZE
    )

    df = pd.read_csv(TEST_CSV)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"CSV 中找不到列: {TARGET_COLUMN}")
    data_series = df[TARGET_COLUMN].values.astype(np.float32)
    print(f"测试数据长度: {len(data_series)}")

    print("生成预测动画中...")
    final_acc = create_animation(
        model, scaler_X, scaler_y, seq_len, output_size,
        data_series, save_path=OUTPUT_GIF, fps=FPS
    )
    print(f"最终整体准确率: {final_acc:.2f}%")

    user_input = input("\n是否生成详细预测报告文本文件？(y/n): ").strip().lower()
    if user_input == 'y':
        script_path = os.path.join(os.path.dirname(__file__), 'test_txt.py')
        cmd = [
            sys.executable, script_path,
            '--model_path', MODEL_PATH,
            '--test_csv', TEST_CSV,
            '--target_col', TARGET_COLUMN
        ]
        subprocess.run(cmd)
    else:
        print("跳过生成文本报告。")


if __name__ == "__main__":
    main()