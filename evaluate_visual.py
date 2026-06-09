import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM  # 🆕 支持两种模型
from utils.data_loader import load_multiple_csv
from utils.trainer import evaluate_model

def get_data_dir_via_gui(configured_path):
    """
    和 run.py 保持一致的智能弹窗逻辑
    """
    if configured_path and os.path.exists(configured_path):
        return configured_path
        
    print("\n[系统提示] 数据目录未配置或路径无效，请选择包含CSV数据的测试文件夹...")
    root = tk.Tk()
    root.withdraw()           
    root.attributes('-topmost', True) 
    
    selected_dir = filedialog.askdirectory(title="请选择测试数据文件夹")
    root.destroy()
    
    if not selected_dir:
        raise ValueError("您取消了文件夹选择，程序终止。")
    print(f"[系统提示] 成功选择数据路径: {selected_dir}")
    return selected_dir

def main():
    # 1. 加载配置
    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print("启动模型可视化评估模块...")

    # 🆕 读取新配置
    use_time_features = cfg.get('use_time_features', False)
    input_size = 1 + (2 if use_time_features else 0)

    # 2. 找到最好的模型 (优先从 first 文件夹找)
    best_dir = os.path.join(cfg['result_root'], "first")
    if not os.path.exists(best_dir) or len(os.listdir(best_dir)) == 0:
        print("未在 first 目录找到最佳模型，请确认是否已成功训练。")
        return

    pth_file = [f for f in os.listdir(best_dir) if f.endswith('.pth')][0]
    model_path = os.path.join(best_dir, pth_file)
    print(f"加载最佳模型权重: {pth_file}")

    # 3. 加载 Checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = checkpoint['params']
    scaler_y = checkpoint['scaler_y']

    # 🆕 从 checkpoint 读取新参数（兼容旧模型，默认值关）
    ckpt_input_size = params.get('input_size', input_size)
    ckpt_use_residual = params.get('use_residual', False)
    ckpt_use_time_features = params.get('use_time_features', False)
    ckpt_model_type = params.get('model_type', 'Seq2SeqLSTM')
    print(f"🆕 模型参数: input_size={ckpt_input_size}, residual={ckpt_use_residual}, time_feat={ckpt_use_time_features}, type={ckpt_model_type}")

    # 4. 加载数据 (只需测试集)
    # 🎯 核心修复：引入智能弹窗，再也不会因为空路径崩溃了
    DATA_DIR = get_data_dir_via_gui(cfg.get("data_dir", ""))

    _, _, _, _, X_test, y_test, _, _ = load_multiple_csv(
        data_dir=DATA_DIR, target_col=cfg['target_column'], seq_len=cfg['seq_length'],
        pred_len=cfg['output_size'], test_size=cfg['test_size'], val_size=cfg['val_size'],
        random_seed=cfg['random_seed'],
        use_time_features=ckpt_use_time_features,  # 🆕 与训练时一致
        period=cfg['output_size']
    )
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 5. 初始化模型架构并注入权重
    if ckpt_model_type == 'LSTM' or ckpt_model_type == 'LSTMMultiStep':
        model = LSTMMultiStep(
            input_size=ckpt_input_size,
            hidden_size=params['hidden_size'],
            output_size=params['output_size'],
            num_layers=params['num_layers'],
            dropout=0,
            use_layer_norm=cfg.get('use_layer_norm', False),
            use_residual=ckpt_use_residual           # 🆕
        ).to(device)
    else:
        model = Seq2SeqLSTM(
            input_size=ckpt_input_size,
            hidden_size=params['hidden_size'],
            output_size=params['output_size'],
            output_feature_size=cfg['output_feature_size'],
            num_layers=params['num_layers'],
            dropout=0,
            use_residual=ckpt_use_residual            # 🆕
        ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 6. 执行预测
    _, _, _, overall_r2, pred, true = evaluate_model(model, X_test, y_test, scaler_y)
    print(f"可视化测试集 R²: {overall_r2:.4f}")

# ---------- 7. 绘图优化版 ----------
    
    # 【图 1：打破视觉错觉的单样本对比】
    num_samples_to_plot = min(4, len(pred))
    indices = np.random.choice(len(pred), num_samples_to_plot, replace=False)
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Sample Forecast (Y-axis Fixed to 0)\nOverall R² = {overall_r2:.4f}", fontsize=14)

    # 找到这几个样本中的全局最大值，用于统一Y轴上限
    local_max = max(np.max(true[indices]), np.max(pred[indices]))

    for i, idx in enumerate(indices):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(1, cfg['output_size'] + 1), true[idx], marker='o', color='blue', label='True Signal')
        plt.plot(range(1, cfg['output_size'] + 1), pred[idx], marker='x', color='red', linestyle='--', label='Predicted')
        
        plt.title(f"Test Sample #{idx} (Abs Error: {np.mean(np.abs(true[idx] - pred[idx])):.4f})")
        plt.xlabel("Future Time Steps")
        plt.ylabel(cfg['target_column'])
        plt.xticks(range(1, cfg['output_size'] + 1))
        # 🎯 核心修复：强制 Y 轴从 0 开始，打破显微镜效应
        plt.ylim(0, local_max * 1.2) 
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    sample_fig_path = os.path.join(best_dir, "forecast_fixed_scale.png")
    plt.savefig(sample_fig_path, dpi=300)

    # 【图 2：全局宏观趋势对比 (揭秘 R2 为何这么高)】
    plt.figure(figsize=(15, 5))
    plt.title(f"Global Test Set Prediction (Step 1 of each window)\nOverall R² = {overall_r2:.4f}")
    
    # 为了防止折线太乱，我们只抽取每个预测窗口的第 1 个预测步连成线
    global_true = true[:, 0]
    global_pred = pred[:, 0]
    
    plt.plot(global_true, label='True Signal', color='blue', alpha=0.7)
    plt.plot(global_pred, label='Predicted Trend', color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel("Continuous Test Samples")
    plt.ylabel(f"{cfg['target_column']} (Step 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    global_fig_path = os.path.join(best_dir, "global_trend.png")
    plt.savefig(global_fig_path, dpi=300)
    
    print(f"✅ 新的可视化结果已保存:\n1. {sample_fig_path}\n2. {global_fig_path}")
    plt.show()
if __name__ == "__main__":
    main()