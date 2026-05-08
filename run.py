import os
import yaml
import torch
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
    """
    检查配置文件中的路径是否有效。若无效或为空，则弹出系统原生文件夹选择框。
    """
    if configured_path and os.path.exists(configured_path):
        return configured_path
        
    print("\n[系统提示] 数据目录未配置或路径不存在，请在弹出的窗口中选择包含CSV数据的文件夹...")
    root = tk.Tk()
    root.withdraw()           # 隐藏主窗口
    root.attributes('-topmost', True) # 将弹窗置顶，防止被终端窗口遮挡
    
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
    print("2. Seq2Seq LSTM (Encoder-Decoder)")
    print("=" * 60)
    choice = input("请输入模型编号 (1/2): ").strip()

    # ---------- 0. 加载 YAML 配置文件 ----------
    config_path = "config.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}。请确保文件存在！")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 动态获取并确认数据路径
    DATA_DIR = get_data_dir_via_gui(cfg.get("data_dir", ""))

    # 设置随机种子和设备
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

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # ---------- 2. 参数网格搜索 ----------
    best_overall_mape = float('inf')
    best_model_info = None

    # 从配置文件读取网格搜索参数
    param_combinations = list(itertools.product(
        cfg['hidden_size'], cfg['num_layers'], cfg['dropout'], 
        cfg['learning_rate'], cfg['patience'], cfg['loss_type']
    ))
    total = len(param_combinations)
    print(f"\n共有 {total} 组参数组合待训练")

    for idx, (hidden, layers, drop, lr, patience, loss_type) in enumerate(param_combinations):
        print("\n" + "=" * 50)
        print(f"进度: {idx+1}/{total}")
        print(f"参数: hidden={hidden}, layers={layers}, dropout={drop}, lr={lr}, patience={patience}, loss_type={loss_type}")

        # 构建模型
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
                output_feature_size=cfg['output_feature_size'], # 注入新增的维度参数
                num_layers=layers,
                dropout=drop
            )
        else:
            raise ValueError("无效的模型选择")

        model = model.to(device)

        # 训练
        model, best_val_loss, train_losses, val_losses = train_model(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=cfg['epochs'],
            lr=lr,
            patience=patience,
            loss_type=loss_type,
            grad_clip=cfg['grad_clip']
        )

        # 评估（返回 MAPE, MSE, MAE）
        overall_mape, overall_mse, overall_mae, pred, true = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y
        )
        acc = 100 - overall_mape
        print(f"测试集整体 MAPE: {overall_mape:.2f}% , 准确率: {acc:.2f}% , "
              f"MSE: {overall_mse:.6f} , MAE: {overall_mae:.6f}")

        # 生成文件名基础部分
        model_name = "LSTM" if choice == '1' else "Seq2SeqLSTM"
        base_filename = f"{model_name}_h{hidden}_l{layers}_drop{drop}_lr{lr}_{loss_type}_mape{overall_mape:.2f}"

        # 保存到 first 目录（若当前最佳则覆盖更新）
        first_dir = os.path.join(cfg['result_root'], "first")
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
                'loss_type': loss_type
            },
            overall_mape=overall_mape,
            save_dir=first_dir,
            filename=base_filename + '.pth'
        )
        # 绘制 Loss 曲线
        plot_loss_curves(train_losses, val_losses, first_dir, base_filename)

        # 如果准确率 >= 90%，额外存入 accuracy_high 目录
        if acc >= 90.0:
            high_dir = os.path.join(cfg['result_root'], "accuracy_high")
            save_model(
                model=model,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                params={'accuracy': acc},
                overall_mape=overall_mape,
                save_dir=high_dir,
                filename=base_filename + '.pth'
            )
            plot_loss_curves(train_losses, val_losses, high_dir, base_filename)
            print(f"✅ 准确率 {acc:.2f}% >= 90%，已存入 accuracy_high/")

        # 更新全局最佳
        if overall_mape < best_overall_mape:
            best_overall_mape = overall_mape
            best_model_info = (save_path, overall_mape)

    print("\n" + "=" * 60)
    print(f"所有训练完成！最佳模型 MAPE = {best_overall_mape:.2f}%")
    if best_model_info:
        print(f"最佳模型保存路径: {best_model_info[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()