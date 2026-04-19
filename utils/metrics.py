import numpy as np


def compute_mape(y_true, y_pred, epsilon=1e-8):
    """
    计算平均绝对百分比误差 (MAPE)，支持多维输出（多步预测）。
    返回整体平均 MAPE（百分比形式）。
    """
    mape_per_step = []
    for i in range(y_true.shape[1]):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        mape = np.mean(np.abs((true_i - pred_i) / (true_i + epsilon))) * 100
        mape_per_step.append(mape)
    return np.mean(mape_per_step)


def compute_mse(y_true, y_pred):
    """计算均方误差 (MSE)"""
    return np.mean((y_true - y_pred) ** 2)


def compute_mae(y_true, y_pred):
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(y_true - y_pred))