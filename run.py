import os
import torch
import itertools
import numpy as np
import pandas as pd

from utils.data_loader import load_multiple_csv
from utils.trainer import train_model, evaluate_model, save_model
from utils.plotting import plot_loss_curves  # 新增导入
from models.LSTM.model import LSTMMultiStep, Seq2SeqLSTM


def main():
    print("=" * 60)
    print("时序预测模型训练 - 请选择模型类型")
    print("1. Vanilla LSTM (直接多步输出)")
    print("2. Seq2Seq LSTM (Encoder-Decoder)")
    print("=" * 60)
    choice = input("请输入模型编号 (1/2): ").strip()

    # ==================== 超参数配置（可按需修改） ====================
    TARGET_COLUMN = 'RMS_Value'         # 预测目标列名，需与 CSV 文件中的列名一致
    INPUT_SIZE = 1                  # 输入特征维度，单变量时为1
    HIDDEN_SIZE = [16, 32]         # 隐藏层大小，建议使用较大值以提升模型表达能力，尤其是 Seq2Seq 模型可能受益更多
    NUM_LAYERS = [2, 1]          # 建议使用多层以配合 dropout
    DROPOUT = [0.5]                    # 建议使用较大 dropout 以防过拟合，尤其是 Seq2Seq 模型
    EPOCHS = 300                            # 最大训练轮数，建议 Seq2Seq 使用较小值以防过拟合
    LEARNING_RATE = [0.0005]         # 可根据模型复杂度调整
    PATIENCE = [15]           # 早停耐心值，建议 Seq2Seq 使用较小值以防过拟合
    SEQ_LENGTH = 144                       # 输入序列长度（历史时间步数）
    OUTPUT_SIZE = 72            # 输出序列长度（预测未来时间步数）
    TEST_SIZE = 0.15                        # 测试集占比
    VAL_SIZE = 0.15               # 验证集占比
    RANDOM_SEED = 42                        # 随机种子
    DATA_DIR = r"data/show"            # 数据文件夹路径
    RESULT_ROOT = "result"              # 结果保存根目录
    LOSS_TYPE = ['huber', 'mse']        # 损失函数类型列表（支持网格搜索）
    USE_LAYER_NORM = True               # 仅对 Vanilla LSTM 有效
    GRAD_CLIP = 1.0                     # 梯度裁剪阈值
    # ==============================================================

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------- 1. 加载数据 ----------
    print("\n正在从文件夹加载多个CSV文件...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_X, scaler_y) = load_multiple_csv(
        data_dir=DATA_DIR,
        target_col=TARGET_COLUMN,
        seq_len=SEQ_LENGTH,
        pred_len=OUTPUT_SIZE,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED
    )

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # ---------- 2. 参数网格搜索 ----------
    best_overall_mape = float('inf')
    best_model_info = None

    # 参数组合（注意顺序与解包一致）
    param_combinations = list(itertools.product(
        HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE, PATIENCE, LOSS_TYPE
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
                input_size=INPUT_SIZE,
                hidden_size=hidden,
                output_size=OUTPUT_SIZE,
                num_layers=layers,
                dropout=drop,
                use_layer_norm=USE_LAYER_NORM
            )
        elif choice == '2':
            model = Seq2SeqLSTM(
                input_size=INPUT_SIZE,
                hidden_size=hidden,
                output_size=OUTPUT_SIZE,
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
            epochs=EPOCHS,
            lr=lr,
            patience=patience,
            loss_type=loss_type,
            grad_clip=GRAD_CLIP
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
        first_dir = os.path.join(RESULT_ROOT, "first")
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
                'seq_len': SEQ_LENGTH,
                'output_size': OUTPUT_SIZE,
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
            high_dir = os.path.join(RESULT_ROOT, "accuracy_high")
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