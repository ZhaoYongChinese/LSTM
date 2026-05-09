import numpy as np
from sklearn.metrics import r2_score # 🎯 在这里统一引入 sklearn 的 R2

def compute_mape(y_true, y_pred, epsilon=1e-8):
    """
    计算平均绝对百分比误差 (MAPE)，支持多维输出（多步预测）。
    返回整体平均 MAPE（百分比形式）。
    注意：在微小数值场景下容易失真，仅作为参考辅助指标。
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

def compute_r2(y_true, y_pred):
    """
    🎯 新增：计算决定系数 (R-squared)
    物理意义：衡量模型预测拟合了多少数据方差。
    范围通常在 [-∞, 1] 之间。越接近 1 说明模型预测越完美。
    """
    return r2_score(y_true, y_pred)