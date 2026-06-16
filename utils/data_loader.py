import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_multiple_csv(data_dir, target_col, seq_len, pred_len,
                      test_size=0.15, val_size=0.15, random_seed=42,
                      value_threshold=None, stride=None):
    """
    从文件夹读取多个CSV文件，过滤低于阈值的停机数据，
    将所有合格点拼接成连续序列后滑窗构造样本。

    参数:
        data_dir:         存放CSV文件的文件夹路径
        target_col:       目标列名
        seq_len:          输入序列长度（运行点的个数）
        pred_len:         预测步长（运行点的个数）
        test_size:        测试集占比
        val_size:         验证集占比
        random_seed:      随机种子
        value_threshold:  只保留 >= 此值的点 (None=不过滤)
        stride:           滑动步长 (None=自动, 或手动指定整数)

    返回:
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor, scaler_X, scaler_y
    """
    min_req_len = seq_len + pred_len

    # ---------- 1. 读取所有CSV，过滤后拼接 ----------
    all_series = []
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
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

        if value_threshold is not None:
            series = series[series >= value_threshold]
            if len(series) == 0:
                print(f"警告：文件 {file} 过滤后为空，跳过")
                continue

        all_series.append(series)

    if not all_series:
        raise ValueError("没有有效数据，请检查文件或阈值设置")

    full_series = np.concatenate(all_series)
    n_total_points = len(full_series)
    print(f"过滤后总点数: {n_total_points} (阈值>={value_threshold})")
    print(f"值范围: [{full_series.min():.6f}, {full_series.max():.6f}]")

    # ---------- 2. 滑窗构造 X, y ----------
    max_samples = n_total_points - min_req_len + 1
    if max_samples <= 0:
        raise ValueError(f"过滤后点数 {n_total_points} < {min_req_len}，无法构造样本")

    if stride is not None:
        _stride = stride
    elif max_samples < 1000:
        _stride = 1
    elif max_samples < 5000:
        _stride = 2
    else:
        _stride = 10
    print(f"最大样本数: {max_samples}, stride={_stride}")

    X_list, y_list = [], []
    for i in range(0, max_samples, _stride):
        x = full_series[i : i + seq_len]                          # [seq_len]
        y = full_series[i + seq_len : i + seq_len + pred_len]     # [pred_len]
        X_list.append(x)
        y_list.append(y)

    X_all = np.array(X_list).reshape(-1, seq_len, 1)   # [n_samples, seq_len, 1]
    y_all = np.array(y_list)                            # [n_samples, pred_len]
    print(f"总样本数: {X_all.shape[0]}, X shape: {X_all.shape}, y shape: {y_all.shape}")

    # ---------- 3. 时序切分 ----------
    n_total = len(X_all)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)
    n_train = n_total - n_test - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"样本数不足，无法划分。总样本: {n_total}")

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[-n_test:], y_all[-n_test:]

    print(f"划分后 - 训练: {n_train}, 验证: {n_val}, 测试: {n_test}")

    # ---------- 4. 统一归一化 ----------
    scaler = StandardScaler()
    all_train = np.concatenate([X_train.flatten(), y_train.flatten()]).reshape(-1, 1)
    scaler.fit(all_train)

    X_train_norm = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_norm   = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_norm  = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train_norm = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_norm   = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_norm  = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # ---------- 5. 转 Tensor ----------
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.FloatTensor(y_train_norm)
    X_val_tensor   = torch.FloatTensor(X_val_norm)
    y_val_tensor   = torch.FloatTensor(y_val_norm)
    X_test_tensor  = torch.FloatTensor(X_test_norm)
    y_test_tensor  = torch.FloatTensor(y_test_norm)

    return (X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor,
            scaler, scaler)
