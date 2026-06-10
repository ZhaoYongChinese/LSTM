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

def load_model(checkpoint_path, device='cpu'):
    """
    加载训练好的模型、归一化器及相关参数。
    自动检测模型类型与层数, 兼容各种旧版checkpoint。
    🆕 v2: 支持 input_size / use_residual / use_time_features 等新参数。
    返回: model, scaler_X, scaler_y, seq_len, output_size, use_time_features
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

    seq_len = params.get('seq_len') or params.get('seq_length')
    if seq_len is None:
        raise ValueError("模型文件中缺少 seq_len，无法继续。")

    output_size = params.get('output_size') or params.get('pred_len')
    if output_size is None:
        if model_type == 'LSTM':
            if 'fc.weight' in state_dict:
                output_size = state_dict['fc.weight'].shape[0]
        if output_size is None:
            raise ValueError("模型文件中缺少 output_size，无法继续。")

    # 🆕 读取新参数（兼容旧 checkpoint，默认值回退）
    ckpt_input_size = params.get('input_size', 1)
    use_residual = params.get('use_residual', False)
    use_time_features = params.get('use_time_features', False)

    if model_type == 'Seq2SeqLSTM':
        model = Seq2SeqLSTM(
            input_size=ckpt_input_size,       # 🆕
            hidden_size=hidden_size,
            output_size=output_size,
            output_feature_size=params.get('output_feature_size', 1),
            num_layers=num_layers,
            dropout=dropout,
            use_residual=use_residual          # 🆕
        )
    else:
        use_layer_norm = params.get('use_layer_norm')
        if use_layer_norm is None:
            use_layer_norm = any(k.startswith('layer_norm') for k in state_dict.keys())
        model = LSTMMultiStep(
            input_size=ckpt_input_size,        # 🆕
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual           # 🆕
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    print(f"[OK] 模型加载成功: {model_type} | hidden={hidden_size}, layers={num_layers}")
    print(f"   seq_len={seq_len}, out_len={output_size}, input_size={ckpt_input_size}")
    print(f"   residual={use_residual}, time_feat={use_time_features}")
    return model, scaler_X, scaler_y, seq_len, output_size, use_time_features  # 🆕 多返回一个值

def compute_mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def predict_sequence(model, scaler_X, scaler_y, input_seq, device='cpu',
                     start_idx=0, use_time_features=False, period=47):
    """
    🆕 v2: 支持时间特征编码 + 统一scaler (1特征) / 旧版scaler (47特征)
    参数:
        input_seq: 原始RMS值序列 [seq_len]
        start_idx: 该序列在原数据中的起始索引 (用于计算日内位置)
        use_time_features: 是否拼接 sin/cos 时间通道
        period: 日内周期步数
    """
    n = len(input_seq)
    rms_norm = scaler_X.transform(input_seq.reshape(-1, 1)).reshape(1, n, 1)

    if use_time_features:
        positions = np.arange(start_idx, start_idx + n) % period
        sin_feat = np.sin(2 * np.pi * positions / period).astype(np.float32).reshape(1, n, 1)
        cos_feat = np.cos(2 * np.pi * positions / period).astype(np.float32).reshape(1, n, 1)
        input_norm = np.concatenate([rms_norm, sin_feat, cos_feat], axis=2)  # [1, n, 3]
    else:
        input_norm = rms_norm  # [1, n, 1]

    input_tensor = torch.FloatTensor(input_norm).to(device)
    with torch.no_grad():
        pred_norm = model(input_tensor)
    pred_np = pred_norm.cpu().numpy()
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(1, -1)

    # 兼容统一scaler (1特征) 和旧版scaler (47特征)
    if hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ == 1:
        pred = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape).flatten()
    else:
        pred = scaler_y.inverse_transform(pred_np).flatten()

    # Log 逆变换: 如果训练时做了 log，这里 exp 还原到原始量纲
    if getattr(scaler_y, 'use_log_transform', False):
        pred = np.exp(pred)

    return pred

def create_animation(model, scaler_X, scaler_y, seq_len, output_size,
                     data_series, save_path='prediction.gif', fps=2,
                     use_time_features=False, data_start_offset=0):
    """
    🆕 v2: 支持时间特征参数传递
    参数:
        data_start_offset: 测试数据在原数据中的起始索引 (用于日内位置计算)
    """
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
        hist_true = data_series[i : i + seq_len]
        future_true = data_series[i + seq_len : i + seq_len + output_size]

        # 🆕 传递 start_idx 和时间特征参数
        global_start = data_start_offset + i
        pred = predict_sequence(
            model, scaler_X, scaler_y, hist_true, device,
            start_idx=global_start,           # 🆕 全局位置索引
            use_time_features=use_time_features # 🆕
        )

        all_true_accum.append(future_true)
        all_pred_accum.append(pred)

        window_mape = compute_mape(future_true, pred)
        window_acc = 100 - window_mape

        overall_true = np.concatenate(all_true_accum)
        overall_pred = np.concatenate(all_pred_accum)
        overall_mape = compute_mape(overall_true, overall_pred)
        overall_acc = 100 - overall_mape

        # 使用局部索引绘制 (相对于测试数据)
        x_future = np.arange(i + seq_len, i + seq_len + output_size)
        x_all = np.arange(i, i + seq_len + output_size)
        y_all_true = np.concatenate([hist_true, future_true])

        ax.plot(x_all, y_all_true, 'b-', label='Actual', linewidth=2)
        ax.plot(x_future, pred, 'r--', label='Predicted', linewidth=2)
        ax.axvline(x=i + seq_len - 1, color='gray', linestyle=':', alpha=0.7)

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

def get_folder_via_gui(title):
    """GUI 文件夹选择器"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def get_test_percent_via_gui(default=15):
    """🆕 GUI 弹窗: 输入测试数据占比 (%)"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    from tkinter import simpledialog
    percent = simpledialog.askfloat(
        "测试数据比例设置",
        "请输入测试数据占全部数据的百分比 (%)\n(取末尾 X% 的数据作为测试集):",
        initialvalue=default,
        minvalue=1,
        maxvalue=99
    )
    root.destroy()
    if percent is None:
        print(f"未输入比例，使用默认值 {default}%")
        return default
    print(f"测试数据比例: {percent}%")
    return percent

def main():
    print("=" * 60)
    print("时序预测模型推理系统 v2")
    print("=" * 60)

    # 1. GUI 选取模型文件
    model_path = get_file_via_gui(
        title="1. 请选择训练好的模型文件 (.pth)",
        file_types=[("PyTorch 模型", "*.pth"), ("所有文件", "*.*")]
    )
    if not model_path:
        print("已取消模型选择，程序退出。")
        return

    # 🆕 2. GUI 选取数据文件夹（替代原来的单个文件）
    data_dir = get_folder_via_gui("2. 请选择测试数据所在的文件夹")
    if not data_dir:
        print("已取消数据选择，程序退出。")
        return

    # 🆕 3. 弹窗选择测试集占比
    test_percent = get_test_percent_via_gui(default=15)

    TARGET_COLUMN = 'RMS_Value'
    FPS = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[配置项] 设备: {device} | 目标列: {TARGET_COLUMN}")

    # 动态生成输出目录
    model_filename = os.path.basename(model_path)
    model_name_no_ext = os.path.splitext(model_filename)[0]
    show_dir = os.path.abspath(os.path.join("show", model_name_no_ext))
    os.makedirs(show_dir, exist_ok=True)
    print(f"[配置项] 结果将保存至: {show_dir}")

    # ---------- 加载模型 ----------
    print("\n正在加载模型...")
    (model, scaler_X, scaler_y,
     seq_len, output_size, use_time_features) = load_model(model_path, device)  # 🆕 解包新返回值

    # ---------- 🆕 从文件夹加载并拼接所有CSV，取末尾 X% ----------
    print(f"\n正在从文件夹加载CSV数据: {data_dir}")
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何CSV文件")
    print(f"找到 {len(csv_files)} 个CSV文件")

    all_series_parts = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        if TARGET_COLUMN not in df.columns:
            print(f"警告: {file} 缺少列 '{TARGET_COLUMN}'，跳过")
            continue
        series = df[TARGET_COLUMN].values.astype(np.float32)
        all_series_parts.append(series)
        print(f"  {file}: {len(series)} 个时间步")

    full_series = np.concatenate(all_series_parts)
    total_len = len(full_series)
    print(f"全部数据总长度: {total_len} 个时间步")

    # 取末尾 test_percent% 作为测试数据
    test_len = max(int(total_len * test_percent / 100), seq_len + output_size)
    data_series = full_series[-test_len:]
    data_start_offset = total_len - test_len  # 🆕 测试数据在全局中的起始索引
    print(f"测试数据: 末尾 {test_len} 个点 ({test_percent}%), 全局起始索引={data_start_offset}")

    # ---------- 执行预测与生成动画 ----------
    print("\n正在生成预测动画...")
    gif_path = os.path.join(show_dir, f"{model_name_no_ext}_animation.gif")
    final_acc = create_animation(
        model, scaler_X, scaler_y, seq_len, output_size,
        data_series, save_path=gif_path, fps=FPS,
        use_time_features=use_time_features,       # 🆕
        data_start_offset=data_start_offset         # 🆕
    )
    print(f"最终整体准确率: {final_acc:.2f}%")

    # ---------- 生成报告 ----------
    user_input = input("\n是否生成详细预测报告文本文件？(y/n): ").strip().lower()
    if user_input == 'y':
        generate_report(
            model=model, scaler_X=scaler_X, scaler_y=scaler_y,
            seq_len=seq_len, output_size=output_size,
            data_series=data_series,
            model_name=model_filename,
            test_csv_name=f"{os.path.basename(data_dir)} (末尾{test_percent}%)",
            target_col=TARGET_COLUMN,
            output_dir=show_dir,
            device=device,
            # 🆕 传递时间特征参数
            use_time_features=use_time_features,
            data_start_offset=data_start_offset
        )
    else:
        print("已跳过文本报告生成。")

    print("\n[OK] 所有推理任务执行完毕!")

if __name__ == "__main__":
    main()