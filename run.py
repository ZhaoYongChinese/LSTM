import os
import yaml
import torch
import shutil
import itertools
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from utils.data_loader import load_multiple_csv
from utils.trainer import train_model, evaluate_model, save_model
from utils.plotting import plot_loss_curves
from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM

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
    print("=" * 60)
    print("时序预测模型训练 - 请选择模型类型")
    print("1. Vanilla LSTM (直接多步输出)")
    print("2. Seq2Seq LSTM (Encoder-Decoder + Attention)")
    print("=" * 60)
    choice = input("请输入模型编号 (1/2): ").strip()
    
    # 🎯 优化：增加早期拦截机制，防止误输入继续向后运行报错
    if choice not in ['1', '2']:
        print("输入无效的模型编号，程序已终止。")
        return

    # ---------- 0. 加载 YAML 配置文件 ----------
    config_path = "config.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}。请确保文件存在！")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    DATA_DIR = get_data_dir_via_gui(cfg.get("data_dir", ""))

    torch.manual_seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------- 1. 加载数据 ----------
    print("\n正在从文件夹加载多个CSV文件...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_X, scaler_y) = load_multiple_csv(
        data_dir=DATA_DIR,
        target_col=cfg['target_column'],
        seq_len=cfg['seq_length'],
        pred_len=cfg['output_size'],
        test_size=cfg['test_size'],
        val_size=cfg['val_size'],
        random_seed=cfg['random_seed']
    )

    # 🎯 核心修复：移除以下这三行，避免全量数据塞入 GPU 导致 OOM（显存爆炸）
    # 此时 X_train 等 Tensor 全都在 CPU 内存中安全待命
    # X_train, y_train = X_train.to(device), y_train.to(device)
    # X_val, y_val = X_val.to(device), y_val.to(device)
    # X_test, y_test = X_test.to(device), y_test.to(device)

    # ---------- 2. 参数网格搜索 ----------
    best_overall_r2 = -float('inf') # R2 越大越好，初始化为负无穷
    best_model_info = None

    param_combinations = list(itertools.product(
        cfg['hidden_size'], cfg['num_layers'], cfg['dropout'], 
        cfg['learning_rate'], cfg['patience'], cfg['loss_type']
    ))
    total = len(param_combinations)
    print(f"\n共有 {total} 组参数组合待训练")
    
    batch_size = cfg.get('batch_size', 64) # 获取批大小配置

    for idx, (hidden, layers, drop, lr, patience, loss_type) in enumerate(param_combinations):
        print("\n" + "=" * 50)
        print(f"进度: {idx+1}/{total}")
        print(f"参数: hidden={hidden}, layers={layers}, dropout={drop}, lr={lr}, patience={patience}, loss_type={loss_type}")

        if choice == '1':
            model = LSTMMultiStep(
                input_size=cfg['input_size'],
                hidden_size=hidden,
                output_size=cfg['output_size'],
                num_layers=layers,
                dropout=drop,
                use_layer_norm=cfg['use_layer_norm']
            )
        elif choice == '2':
            model = Seq2SeqLSTM(
                input_size=cfg['input_size'],
                hidden_size=hidden,
                output_size=cfg['output_size'],
                output_feature_size=cfg['output_feature_size'], 
                num_layers=layers,
                dropout=drop
            )

        model = model.to(device)

        # 🎯 修改：传入 device 参数，由 trainer 在内部按批次调度显存
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
            grad_clip=cfg['grad_clip']
        )

        # 🎯 修改：传入 device 参数，让 evaluate_model 安全推理
        overall_mape, overall_mse, overall_mae, overall_r2, pred, true = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y,
            device=device
        )
        
        print(f"测试集整体 R² (决定系数): {overall_r2:.4f} , MSE: {overall_mse:.6f} , MAE: {overall_mae:.6f} (辅助MAPE: {overall_mape:.2f}%)")

        model_name = "LSTM" if choice == '1' else "Seq2SeqLSTM"
        base_filename = f"{model_name}_h{hidden}_l{layers}_drop{drop}_lr{lr}_{loss_type}_r2_{overall_r2:.4f}"

        # 根据 R2 分数进行筛选分类
        if overall_r2 >= 0.80 and overall_mape <= 10:  # 设定双重筛选条件，确保模型不仅 R2 高，还要 MAPE 低
            current_save_dir = os.path.join(cfg['result_root'], "accuracy_high")
            print(f"✅ R² 分数 {overall_r2:.4f} >= 0.80 且 MAPE {overall_mape:.2f}% <= 10%，存入 accuracy_high/ 文件夹")
        else:
            current_save_dir = os.path.join(cfg['result_root'], "other")
            print(f"⚠️ R² 分数 {overall_r2:.4f} < 0.80 或 MAPE {overall_mape:.2f}% > 10%，存入 other/ 文件夹")
    
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
                'seq_len': cfg['seq_length'],
                'output_size': cfg['output_size'],
                'loss_type': loss_type,
                'r2_score': overall_r2 # 记录新指标
            },
            overall_r2=overall_r2,
            save_dir=current_save_dir,
            filename=base_filename + '.pth'
        )
        plot_loss_curves(train_losses, val_losses, current_save_dir, base_filename)

        # 记录并更新全局 R2 最高的模型
        if overall_r2 > best_overall_r2:
            best_overall_r2 = overall_r2
            best_model_info = {
                'path': save_path,
                'dir': current_save_dir,
                'base_filename': base_filename,
                'r2': overall_r2
            }

    # ---------- 3. 筛选结束，将本次运行的最高分模型复制到 first ----------
    print("\n" + "=" * 60)
    if best_model_info:
        print(f"所有训练完成！本次网格搜索最佳模型 R² = {best_overall_r2:.4f}")
        first_dir = os.path.join(cfg['result_root'], "first")
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
                
        print(f"🏆 最佳模型(R²={best_overall_r2:.4f})相关文件已成功提拔并复制到: {first_dir}")
        for cf in copied_files:
            print(f"  - 复制成功: {cf}")
    print("=" * 60)

if __name__ == "__main__":
    main()