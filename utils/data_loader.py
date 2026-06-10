import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_multiple_csv(data_dir, target_col, seq_len, pred_len,
                      test_size=0.15, val_size=0.15, random_seed=42,
                      use_time_features=True, period=47,
                      use_log_transform=False, stride=None):
    """
    从文件夹读取多个CSV文件，构造训练/验证/测试集。
    支持动态滑窗步长，绝不跨文件采样。

    v2 改动:
      - use_time_features: 在输入中加入 sin/cos 日内时间编码
      - 统一 scaler: X 和 y 使用同一个 StandardScaler
      - use_log_transform: 对 RMS 做 log 变换后再归一化
      - stride: 可手动指定滑动步长

    参数:
        data_dir: 存放CSV文件的文件夹路径
        target_col: 目标列名
        seq_len: 输入序列长度（回溯步数）
        pred_len: 预测步长
        test_size: 测试集比例
        val_size: 验证集比例
        random_seed: 随机种子
        use_time_features: 是否在输入中加入 sin/cos 编码
        period: 每日周期步数（默认 47）
        use_log_transform: 是否对 RMS 做 log 变换
        stride: 滑动步长 (None=自动, 或手动指定整数)

    返回:
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor, scaler_X, scaler_y
    """
    min_req_len = seq_len + pred_len
    all_X, all_y = [], []
    num_channels = 1 + (2 if use_time_features else 0)  # RMS + (sin, cos)

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

        # Log 变换: 压缩数量级差距，让低值和高值区域的误差在 loss 中被同等对待
        if use_log_transform:
            series = np.log(series)

        n = len(series)

        if n < min_req_len:
            print(f"文件 {file} 长度 {n} < {min_req_len}，丢弃")
            continue

        max_samples = n - min_req_len + 1
        if stride is not None:
            _stride = stride
        elif max_samples < 1000:
            _stride = 1
        elif max_samples < 5000:
            _stride = 2
        else:
            _stride = 10
        print(f"文件 {file} 长度 {n}, 最大样本数 {max_samples}, 采用 stride={_stride}")

        X_file, y_file = [], []
        for i in range(0, max_samples, _stride):
            x_rms = series[i : i + seq_len]           # [seq_len]
            y = series[i + seq_len : i + min_req_len]  # [pred_len]

            if use_time_features:
                # 🆕 日内时间编码: 每个时间步是当天的第几个半小时 (周期=47)
                positions = np.arange(i, i + seq_len) % period
                sin_feat = np.sin(2 * np.pi * positions / period).astype(np.float32)
                cos_feat = np.cos(2 * np.pi * positions / period).astype(np.float32)
                x = np.stack([x_rms, sin_feat, cos_feat], axis=1)  # [seq_len, 3]
            else:
                x = x_rms.reshape(-1, 1)  # [seq_len, 1]

            X_file.append(x)
            y_file.append(y)

        if X_file:
            all_X.append(np.array(X_file))  # [n_samples, seq_len, channels]
            all_y.append(np.array(y_file))  # [n_samples, pred_len]

    if not all_X:
        raise ValueError("没有有效数据，请检查文件长度或目标列名")

    X_all = np.concatenate(all_X, axis=0)  # [total, seq_len, channels]
    y_all = np.concatenate(all_y, axis=0)  # [total, pred_len]

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

    # ─── 🆕 统一归一化: X 和 y 共用同一个 Scaler ───
    # 拆分 RMS 通道 (channel 0) 和时间特征 (channel 1:, 如 sin/cos)
    if use_time_features:
        X_train_rms = X_train[:, :, 0:1]   # [n, seq_len, 1]
        X_train_time = X_train[:, :, 1:]    # [n, seq_len, 2], 时间特征在 [-1,1] 无需归一化
        X_val_rms   = X_val[:, :, 0:1]
        X_val_time  = X_val[:, :, 1:]
        X_test_rms  = X_test[:, :, 0:1]
        X_test_time = X_test[:, :, 1:]
    else:
        X_train_rms = X_train
        X_val_rms   = X_val
        X_test_rms  = X_test

    # 在训练集上拟合统一 scaler（同时覆盖 X 和 y 的 RMS 值，保证分布一致）
    scaler = StandardScaler()
    all_train_rms = np.concatenate([
        X_train_rms.flatten(),
        y_train.flatten()
    ]).reshape(-1, 1)
    scaler.fit(all_train_rms)
    # 标记是否使用 log 变换，下游 evaluate / predict 据此决定是否 exp 还原
    scaler.use_log_transform = use_log_transform

    # 分别对 X 的 RMS 通道和 y 做变换
    X_train_rms_norm = scaler.transform(X_train_rms.reshape(-1, 1)).reshape(X_train_rms.shape)
    X_val_rms_norm   = scaler.transform(X_val_rms.reshape(-1, 1)).reshape(X_val_rms.shape)
    X_test_rms_norm  = scaler.transform(X_test_rms.reshape(-1, 1)).reshape(X_test_rms.shape)

    y_train_norm = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_norm   = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_norm  = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # 拼回时间特征（时间特征始终在 [-1,1]，无需归一化）
    if use_time_features:
        X_train_norm = np.concatenate([X_train_rms_norm, X_train_time], axis=2)
        X_val_norm   = np.concatenate([X_val_rms_norm,   X_val_time],   axis=2)
        X_test_norm  = np.concatenate([X_test_rms_norm,  X_test_time],  axis=2)
    else:
        X_train_norm = X_train_rms_norm
        X_val_norm   = X_val_rms_norm
        X_test_norm  = X_test_rms_norm

    # 转为Tensor
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.FloatTensor(y_train_norm)
    X_val_tensor   = torch.FloatTensor(X_val_norm)
    y_val_tensor   = torch.FloatTensor(y_val_norm)
    X_test_tensor  = torch.FloatTensor(X_test_norm)
    y_test_tensor  = torch.FloatTensor(y_test_norm)

    # 🆕 返回同一个 scaler（作为 scaler_X 和 scaler_y），保持外部接口兼容
    return (X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor,
            scaler, scaler)