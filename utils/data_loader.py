import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_multiple_csv(data_dir, target_col, seq_len, pred_len,
                      test_size=0.15, val_size=0.15, random_seed=42):
    """
    从文件夹读取多个CSV文件，构造训练/验证/测试集。
    支持动态滑窗步长，绝不跨文件采样。

    参数:
        data_dir: 存放CSV文件的文件夹路径
        target_col: 目标列名
        seq_len: 输入序列长度（回溯步数）
        pred_len: 预测步长
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练+验证部分）
        random_seed: 随机种子（保留，但划分时不打乱顺序）

    返回:
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor, scaler_X, scaler_y
    """
    min_req_len = seq_len + pred_len
    all_X, all_y = [], []

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何CSV文件")
    print(f"找到 {len(csv_files)} 个CSV文件")

    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"警告：读取文件 {file} 失败，跳过。错误: {e}")
            continue

        if target_col not in df.columns:
            print(f"警告：文件 {file} 缺少列 '{target_col}'，跳过")
            continue

        series = df[target_col].values.astype(np.float32)
        n = len(series)

        if n < min_req_len:
            print(f"文件 {file} 长度 {n} < {min_req_len}，丢弃")
            continue

        max_samples = n - min_req_len + 1
        if max_samples < 100:
            stride = 1
        elif max_samples < 500:
            stride = 2
        else:
            stride = 10
        print(f"文件 {file} 长度 {n}, 最大样本数 {max_samples}, 采用 stride={stride}")

        X_file, y_file = [], []
        for i in range(0, max_samples, stride):
            x = series[i : i + seq_len]
            y = series[i + seq_len : i + min_req_len]
            X_file.append(x)
            y_file.append(y)

        if X_file:
            all_X.append(np.array(X_file))
            all_y.append(np.array(y_file))

    if not all_X:
        raise ValueError("没有有效数据，请检查文件长度或目标列名")

    X_all = np.concatenate(all_X, axis=0).reshape(-1, seq_len, 1)
    y_all = np.concatenate(all_y, axis=0)  # shape: (total_samples, pred_len)

    print(f"总样本数: {X_all.shape[0]}, 输入形状: {X_all.shape}, 输出形状: {y_all.shape}")

    # 按时间顺序划分（数据已按文件顺序拼接，不随机打乱）
    n_total = len(X_all)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)
    n_train = n_total - n_test - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"样本数不足，无法划分。总样本: {n_total}")

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test, y_test = X_all[-n_test:], y_all[-n_test:]

    print(f"划分后 - 训练: {n_train}, 验证: {n_val}, 测试: {n_test}")

    # 归一化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_2d = X_train.reshape(-1, 1)
    X_val_2d = X_val.reshape(-1, 1)
    X_test_2d = X_test.reshape(-1, 1)

    X_train_norm = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_norm = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    X_test_norm = scaler_X.transform(X_test_2d).reshape(X_test.shape)

    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)

    # 转为Tensor
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.FloatTensor(y_train_norm)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    y_val_tensor = torch.FloatTensor(y_val_norm)
    X_test_tensor = torch.FloatTensor(X_test_norm)
    y_test_tensor = torch.FloatTensor(y_test_norm)

    return (X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor,
            scaler_X, scaler_y)