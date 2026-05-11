import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog

from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM
# 导入报告生成模块
from test_txt import generate_report

def get_file_via_gui(title, file_types):
    """通用的 GUI 文件选择器"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title, filetypes=file_types)
    root.destroy()
    return file_path

def load_model(checkpoint_path, device='cpu', default_seq_len=None, default_output_size=None):
    """
    加载训练好的模型、归一化器及相关参数。
    自动检测模型类型与层数，兼容各种旧版checkpoint。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = checkpoint.get('params', {})
    state_dict = checkpoint['model_state_dict']

    if 'encoder.weight_ih_l0' in state_dict or 'decoder.weight_ih_l0' in state_dict:
        model_type = 'Seq2SeqLSTM'
    else:
        model_type = 'LSTM'
    print(f"检测到模型类型: {model_type}")

    hidden_size = params.get('hidden_size') or params.get('hidden')
    if hidden_size is None:
        if 'lstm.weight_ih_l0' in state_dict:
            hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        elif 'encoder.weight_ih_l0' in state_dict:
            hidden_size = state_dict['encoder.weight_ih_l0'].shape[0] // 4
        else:
            hidden_size = 64

    num_layers = params.get('num_layers') or params.get('layers')
    if num_layers is None:
        max_idx = -1
        for key in state_dict.keys():
            if '_l' in key:
                parts = key.split('_l')
                if len(parts) > 1:
                    num_str = ''.join([ch for ch in parts[1] if ch.isdigit()])
                    if num_str:
                        idx = int(num_str)
                        if idx > max_idx:
                            max_idx = idx
        num_layers = max_idx + 1 if max_idx >= 0 else 1

    dropout = params.get('dropout') or params.get('drop') or 0.2

    seq_len = params.get('seq_len') or params.get('seq_length') or default_seq_len
    if seq_len is None:
        raise ValueError("模型文件中缺少 seq_len，请在 test.py 中设置 DEFAULT_SEQ_LENGTH。")

    output_size = params.get('output_size') or params.get('pred_len') or default_output_size
    if output_size is None:
        if model_type == 'LSTM':
            if 'fc.weight' in state_dict:
                output_size = state_dict['fc.weight'].shape[0]
            elif 'linear.weight' in state_dict:
                output_size = state_dict['linear.weight'].shape[0]
        if output_size is None:
            raise ValueError("模型文件中缺少 output_size，请在 test.py 中设置 DEFAULT_OUTPUT_SIZE。")

    if model_type == 'Seq2SeqLSTM':
        model = Seq2SeqLSTM(
            input_size=1, hidden_size=hidden_size,
            output_size=output_size, num_layers=num_layers, dropout=dropout
        )
    else:
        use_layer_norm = params.get('use_layer_norm')
        if use_layer_norm is None:
            use_layer_norm = any(k.startswith('layer_norm') for k in state_dict.keys())
        model = LSTMMultiStep(
            input_size=1, hidden_size=hidden_size,
            output_size=output_size, num_layers=num_layers,
            dropout=dropout, use_layer_norm=use_layer_norm
        )

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
        pred_norm = model(input_tensor)
    pred_np = pred_norm.cpu().numpy()
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

    all_true_accum, all_pred_accum = [], []
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

        x_future = np.arange(start_idx + seq_len, start_idx + seq_len + output_size)
        x_all = np.arange(start_idx, start_idx + seq_len + output_size)
        y_all_true = np.concatenate([hist_true, future_true])

        ax.plot(x_all, y_all_true, 'b-', label='Actual', linewidth=2)
        ax.plot(x_future, pred, 'r--', label='Predicted', linewidth=2)
        ax.axvline(x=start_idx + seq_len - 1, color='gray', linestyle=':', alpha=0.7)

        # 取消注释或根据实际情况调整Y轴界限，保证动画流畅度
        # ax.set_ylim(0.00, 0.02) 
        
        ax.legend(loc='upper left')
        ax.set_xlabel('Time Step (index)')
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

    return overall_acc if all_true_accum else 0.0

def main():
    print("=" * 60)
    print("时序预测模型推理系统")
    print("=" * 60)

    # 1. GUI 选取模型文件
    model_path = get_file_via_gui(
        title="1. 请选择训练好的模型文件 (.pth)",
        file_types=[("PyTorch 模型", "*.pth"), ("所有文件", "*.*")]
    )
    if not model_path:
        print("已取消模型选择，程序退出。")
        return

    # 2. GUI 选取测试数据文件
    test_csv = get_file_via_gui(
        title="2. 请选择待测试的 CSV 数据文件",
        file_types=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
    )
    if not test_csv:
        print("已取消数据选择，程序退出。")
        return

    TARGET_COLUMN = 'RMS_Value'
    FPS = 2
    DEFAULT_SEQ_LENGTH = 15
    DEFAULT_OUTPUT_SIZE = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[配置项] 设备: {device}")

    # ==================== 动态生成输出目录 ====================
    # 提取模型名称（去掉扩展名，例如 Seq2SeqLSTM_h64...r2_0.8510）
    model_filename = os.path.basename(model_path)
    model_name_no_ext = os.path.splitext(model_filename)[0]
    
    # 构建目标文件夹: 位于 models/LSTM/show/<模型名称>
    # 假设当前环境在项目根目录运行
    show_dir = os.path.abspath(os.path.join("show", model_name_no_ext))
    os.makedirs(show_dir, exist_ok=True)
    print(f"[配置项] 结果将保存至: {show_dir}")
    # =========================================================

    print("\n正在加载模型...")
    model, scaler_X, scaler_y, seq_len, output_size = load_model(
        model_path, device,
        default_seq_len=DEFAULT_SEQ_LENGTH,
        default_output_size=DEFAULT_OUTPUT_SIZE
    )

    df = pd.read_csv(test_csv)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"CSV 中找不到目标列: {TARGET_COLUMN}")
    data_series = df[TARGET_COLUMN].values.astype(np.float32)
    print(f"已加载数据，包含 {len(data_series)} 个时间步。")

    # 执行预测与生成动画
    print("\n正在生成预测动画...")
    gif_path = os.path.join(show_dir, f"{model_name_no_ext}_animation.gif")
    final_acc = create_animation(
        model, scaler_X, scaler_y, seq_len, output_size,
        data_series, save_path=gif_path, fps=FPS
    )
    print(f"最终整体准确率: {final_acc:.2f}%")

    # 执行生成报告
    user_input = input("\n是否生成详细预测报告文本文件？(y/n): ").strip().lower()
    if user_input == 'y':
        # 调用模块内部函数，此时无需再次加载数据和模型，直接利用内存中的数据生成txt
        generate_report(
            model=model, scaler_X=scaler_X, scaler_y=scaler_y, 
            seq_len=seq_len, output_size=output_size, 
            data_series=data_series, 
            model_name=model_filename, 
            test_csv_name=os.path.basename(test_csv), 
            target_col=TARGET_COLUMN, 
            output_dir=show_dir, 
            device=device
        )
    else:
        print("已跳过文本报告生成。")

    print("\n✅ 所有推理任务执行完毕！")

if __name__ == "__main__":
    main()