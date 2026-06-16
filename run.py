import os
import sys
import yaml
import torch
import shutil
import itertools
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

from utils.data_loader import load_multiple_csv
from utils.trainer import train_model, evaluate_model, save_model
from utils.plotting import plot_loss_curves
from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM


class Tee:
    """同时输出到控制台和日志文件"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def get_data_dir_via_gui(configured_path):
    if configured_path and os.path.exists(configured_path):
        return configured_path
        
    print("\n[系统提示] 数据目录未配置或路径不存在，请在弹出的窗口中选择包含CSV数据的文件夹...")
    root = tk.Tk()
    root.withdraw()           
    root.attributes('-topmost', True) 
    
    selected_dir = filedialog.askdirectory(title="请选择数据文件夹(DATA_DIR)")
    root.destroy()
    
    if not selected_dir:
        raise ValueError("您取消了文件夹选择，程序终止。")
    print(f"[系统提示] 成功选择数据路径: {selected_dir}")
    return selected_dir

def main():
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    # ---- 日志: 同步输出到控制台和文件 ----
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(log_dir, log_filename)
    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print("时序预测模型训练 - 请选择模型类型")
    print("1. Vanilla LSTM (直接多步输出)")
    print("2. Seq2Seq LSTM (Encoder-Decoder + Attention)")
    print("=" * 60)
    print(f"[日志] 训练记录将保存至: {log_path}")
    choice = input("请输入模型编号 (1/2): ").strip()

    if choice not in ['1', '2']:
        print("输入无效的模型编号，程序已终止。")
        sys.stdout = tee.terminal
        tee.close()
        return

    # ---------- 0. 加载 YAML 配置文件 ----------
    config_path = "config.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}。请确保文件存在！")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 读取配置项
    use_residual = cfg.get('use_residual', False)
    value_threshold = cfg.get('value_threshold', None)
    input_size = 1  # 单变量 RMS，无时间特征

    # 处理 seq_length / output_size: 统一为列表，外层循环遍历
    seq_lengths = cfg['seq_length']
    if not isinstance(seq_lengths, list):
        seq_lengths = [seq_lengths]
    output_sizes = cfg['output_size']
    if not isinstance(output_sizes, list):
        output_sizes = [output_sizes]

    stride_cfg = cfg.get('stride', None)

    DATA_DIR = get_data_dir_via_gui(cfg.get("data_dir", ""))

    torch.manual_seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"残差: {'ON' if use_residual else 'OFF'} | 阈值: {value_threshold}")
    print(f"seq_len 待测: {seq_lengths} | output_size 待测: {output_sizes}")

    batch_size = cfg.get('batch_size', 64)

    # ---------- 2. 参数网格 ----------
    weight_decays = cfg.get('weight_decay', [0])
    if not isinstance(weight_decays, list):
        weight_decays = [weight_decays]

    param_combinations = list(itertools.product(
        cfg['hidden_size'], cfg['num_layers'], cfg['dropout'],
        cfg['learning_rate'], cfg['patience'], cfg['loss_type'], weight_decays
    ))
    per_data = len(param_combinations)
    total = per_data * len(seq_lengths) * len(output_sizes)
    print(f"\n每组(seq_len,out_size)下 {per_data} 组参数 × {len(seq_lengths)}×{len(output_sizes)} = 共 {total} 组")

    best_overall_mape = float('inf')
    best_model_info = None
    global_counter = 0

    # ---------- 3. 外层: seq_len × output_size ----------
    for seq_len in seq_lengths:
        for out_size in output_sizes:
            print("\n" + "=" * 60)
            print(f"加载数据 seq_len={seq_len}, output_size={out_size}")
            print("=" * 60)

            (X_train, y_train, X_val, y_val, X_test, y_test,
             scaler_X, scaler_y) = load_multiple_csv(
                data_dir=DATA_DIR,
                target_col=cfg['target_column'],
                seq_len=seq_len,
                pred_len=out_size,
                test_size=cfg['test_size'],
                val_size=cfg['val_size'],
                random_seed=cfg['random_seed'],
                value_threshold=value_threshold,
                stride=stride_cfg
            )

            # ---------- 内层: 其他参数网格 ----------
            for hidden, layers, drop, lr, patience, loss_type, wd in param_combinations:
                global_counter += 1
                print("\n" + "-" * 50)
                print(f"进度: {global_counter}/{total} | seq_len={seq_len}, out={out_size}")
                print(f"参数: hidden={hidden}, layers={layers}, dropout={drop}, lr={lr}, patience={patience}, loss={loss_type}, wd={wd}")

                if choice == '1':
                    model = LSTMMultiStep(
                        input_size=input_size,
                        hidden_size=hidden,
                        output_size=out_size,
                        num_layers=layers,
                        dropout=drop,
                        use_layer_norm=cfg['use_layer_norm'],
                        use_residual=use_residual
                    )
                elif choice == '2':
                    model = Seq2SeqLSTM(
                        input_size=input_size,
                        hidden_size=hidden,
                        output_size=out_size,
                        num_layers=layers,
                        dropout=drop,
                        use_residual=use_residual
                    )

                model = model.to(device)

                model, best_val_loss, train_losses, val_losses = train_model(
                    model=model,
                    train_data=(X_train, y_train),
                    val_data=(X_val, y_val),
                    epochs=cfg['epochs'],
                    lr=lr,
                    patience=patience,
                    device=device,
                    batch_size=batch_size,
                    loss_type=loss_type,
                    grad_clip=cfg['grad_clip'],
                    weight_decay=wd
                )

                overall_mape, overall_mse, overall_mae, overall_r2, pred, true = evaluate_model(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    scaler_y=scaler_y,
                    device=device
                )

                print(f"R2={overall_r2:.4f} | MAPE={overall_mape:.2f}% | MSE={overall_mse:.6f} | MAE={overall_mae:.6f}")

                model_name = "LSTM" if choice == '1' else "Seq2SeqLSTM"
                base_filename = f"{model_name}_h{hidden}_l{layers}_drop{drop}_lr{lr}_{loss_type}_mape_{overall_mape:.2f}_in{seq_len}_out{out_size}"

                if overall_mape < 5.0:
                    current_save_dir = os.path.join(cfg['result_root'], "accuracy_high")
                    print(f"  MAPE {overall_mape:.2f}% < 5% -> accuracy_high/")
                else:
                    current_save_dir = os.path.join(cfg['result_root'], "other")
                    print(f"  MAPE {overall_mape:.2f}% >= 5% -> other/")

                save_path = save_model(
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    params={
                        'model_type': model_name,
                        'hidden_size': hidden,
                        'num_layers': layers,
                        'dropout': drop,
                        'learning_rate': lr,
                        'patience': patience,
                        'seq_len': seq_len,
                        'output_size': out_size,
                        'loss_type': loss_type,
                        'weight_decay': wd,
                        'mape': overall_mape,
                        'r2_score': overall_r2,
                        'input_size': input_size,
                        'use_residual': use_residual,
                        'value_threshold': value_threshold,
                    },
                    overall_r2=overall_r2,
                    save_dir=current_save_dir,
                    filename=base_filename + '.pth'
                )
                plot_loss_curves(train_losses, val_losses, current_save_dir, base_filename)

                if overall_mape < best_overall_mape:
                    best_overall_mape = overall_mape
                    best_model_info = {
                        'path': save_path,
                        'dir': current_save_dir,
                        'base_filename': base_filename,
                        'mape': overall_mape
                    }

    # ---------- 4. 最佳模型 → first ----------
    print("\n" + "=" * 60)
    if best_model_info:
        print(f"所有训练完成! 最佳 MAPE = {best_model_info['mape']:.2f}%")
        first_dir = os.path.join(cfg['result_root'], "first")
        if os.path.exists(first_dir):
            shutil.rmtree(first_dir)
        os.makedirs(first_dir, exist_ok=True)

        best_dir = best_model_info['dir']
        base_name = best_model_info['base_filename']

        copied_files = []
        for file_name in os.listdir(best_dir):
            if file_name.startswith(base_name):
                src_path = os.path.join(best_dir, file_name)
                dst_path = os.path.join(first_dir, file_name)
                shutil.copy2(src_path, dst_path)
                copied_files.append(file_name)

        print(f"最佳模型 (MAPE={best_model_info['mape']:.2f}%) 已复制到: {first_dir}")
        for cf in copied_files:
            print(f"  - {cf}")
        print("=" * 60)

    sys.stdout = tee.terminal
    tee.close()
    print(f"[日志] 训练记录已保存至: {log_path}")

if __name__ == "__main__":
    main()