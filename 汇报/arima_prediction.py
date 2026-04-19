"""
ARIMA 时序预测 - 滑动窗口多步预测版（和LSTM逻辑完全对齐）
用法：python arima_prediction.py
"""
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 忽略收敛警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== 配置参数（和LSTM完全对齐） ==========
DATA_PATH = "data/elevator/9天rms值(1).csv"
TARGET_COL = "RMS_Value"

# 和LSTM完全一致的序列长度
SEQ_LENGTH = 144    # 历史输入长度
OUTPUT_SIZE = 72    # 预测步长
TRAIN_RATIO = 0.7   # 训练集比例（用于预训练最优阶数，不参与滑动预测）

# 阶数搜索范围
P_VALUES = range(0, 4)
D_VALUES = range(0, 3)
Q_VALUES = range(0, 4)
# ==================================================

def load_data(path, target_col):
    """加载并返回一维时间序列"""
    df = pd.read_csv(path)
    series = df[target_col].values.astype(float)
    return series

def smape(y_true, y_pred):
    """对称平均绝对百分比误差（范围 0~200%）"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.mean(numerator / denominator) * 100

def evaluate(y_true, y_pred):
    """全指标评估"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    s_mape = smape(y_true, y_pred)
    return mae, rmse, s_mape

def determine_best_order(train_series, p_range, d_range, q_range):
    """网格搜索全局最优ARIMA阶数（仅在训练集执行一次，避免每个窗口重复搜索）"""
    best_aic = np.inf
    best_order = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = SARIMAX(
                        train_series, 
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted = model.fit(disp=False, maxiter=500)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    print(f"✅ 全局最优 ARIMA 阶数: {best_order}, 最小AIC: {best_aic:.2f}")
    return best_order

def sliding_window_predict(series, seq_len, pred_len, best_order):
    """
    滑动窗口多步预测（和LSTM完全一致的逻辑）
    输入：完整序列
    输出：所有窗口的真实值、预测值、整体指标
    """
    total_len = len(series)
    max_start_idx = total_len - seq_len - pred_len
    
    if max_start_idx < 0:
        raise ValueError(f"数据长度 {total_len} 不足，至少需要 {seq_len + pred_len} 个点")
    
    all_true = []
    all_pred = []
    window_results = []
    
    print(f"开始滑动窗口预测，总窗口数: {max_start_idx+1}")
    for start_idx in range(max_start_idx + 1):
        # 1. 截取当前窗口的历史数据和未来真实值
        hist_end = start_idx + seq_len
        hist_series = series[start_idx:hist_end]
        future_true = series[hist_end:hist_end + pred_len]
        
        # 2. 用当前窗口的历史数据拟合ARIMA模型，预测未来pred_len个点
        try:
            model = SARIMAX(
                hist_series, 
                order=best_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=200)
            future_pred = fitted_model.forecast(steps=pred_len)
        except Exception as e:
            # 拟合失败时，用历史均值填充
            future_pred = np.full(pred_len, np.mean(hist_series))
            print(f"窗口 {start_idx} 拟合失败，用均值填充")
        
        # 3. 保存结果
        all_true.append(future_true)
        all_pred.append(future_pred)
        
        # 计算当前窗口的指标
        window_mae, window_rmse, window_smape = evaluate(future_true, future_pred)
        window_results.append({
            "start_idx": start_idx,
            "true": future_true,
            "pred": future_pred,
            "mae": window_mae,
            "smape": window_smape
        })
        
        # 打印进度
        if (start_idx + 1) % 20 == 0:
            print(f"已完成 {start_idx+1}/{max_start_idx+1} 个窗口")
    
    # 计算整体指标
    overall_true = np.concatenate(all_true)
    overall_pred = np.concatenate(all_pred)
    overall_mae, overall_rmse, overall_smape = evaluate(overall_true, overall_pred)
    
    print("\n==================== 整体预测结果 ====================")
    print(f"总预测点数: {len(overall_true)}")
    print(f"整体 MAE: {overall_mae:.8f}")
    print(f"整体 RMSE: {overall_rmse:.8f}")
    print(f"整体 sMAPE: {overall_smape:.4f}%")
    print(f"整体准确率: {100 - overall_smape:.4f}%")
    print("=======================================================")
    
    return window_results, overall_true, overall_pred, (overall_mae, overall_rmse, overall_smape)

def create_prediction_plot(series, seq_len, pred_len, window_results, save_path="result/arima_sliding_prediction.png"):
    """绘制最后一个窗口的预测效果图"""
    last_window = window_results[-1]
    start_idx = last_window["start_idx"]
    hist_end = start_idx + seq_len
    
    # 历史+未来真实值
    full_true = series[start_idx:hist_end + pred_len]
    x_full = np.arange(start_idx, hist_end + pred_len)
    x_pred = np.arange(hist_end, hist_end + pred_len)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_full, full_true, label='Actual', linewidth=2, color='#1f77b4')
    plt.plot(x_pred, last_window["pred"], label='Predicted', linestyle='--', linewidth=2, color='#ff7f0e')
    plt.axvline(x=hist_end - 1, color='gray', linestyle=':', alpha=0.7, label='History/Future Boundary')
    plt.legend(loc='upper left')
    plt.title(f'ARIMA{best_order} Sliding Window Prediction | Window Start = {start_idx} | sMAPE: {last_window["smape"]:.2f}%')
    plt.xlabel('Time Step (index)')
    plt.ylabel('RMS Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"预测效果图已保存至: {save_path}")

def create_animation_gif(series, seq_len, pred_len, window_results, save_path="result/arima_prediction_animation.gif", fps=1):
    """生成和LSTM完全一致的滑动预测动图"""
    total_len = len(series)
    max_start_idx = len(window_results) - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(top=0.85)

    def animate(i):
        ax.clear()
        window = window_results[i]
        start_idx = window["start_idx"]
        hist_end = start_idx + seq_len
        
        # 历史+未来真实值
        hist_true = series[start_idx:hist_end]
        future_true = series[hist_end:hist_end + pred_len]
        full_true = np.concatenate([hist_true, future_true])
        pred = window["pred"]
        
        # 坐标
        x_hist = np.arange(start_idx, hist_end)
        x_future = np.arange(hist_end, hist_end + pred_len)
        x_all = np.arange(start_idx, hist_end + pred_len)
        
        # 绘图
        ax.plot(x_all, full_true, 'b-', label='Actual', linewidth=2)
        ax.plot(x_future, pred, 'r--', label='Predicted', linewidth=2)
        ax.axvline(x=hist_end - 1, color='gray', linestyle=':', alpha=0.7)
        
        # 美化
        ax.legend(loc='upper left')
        ax.set_xlabel('Time Step (index)')
        ax.set_ylabel('RMS Value')
        # ax.set_title(f'ARIMA{best_order} Sliding Window | Window {i+1}/{max_start_idx+1} (Start = {start_idx})')
        ax.grid(True, alpha=0.3)
        
        # 指标显示
        ax.text(0.98, 0.95, f'Window sMAPE: {window["smape"]:.2f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 1.02, f'Overall Accuracy: {100 - overall_smape:.2f}%',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center',
                color='green')
        
        return ax,

    print("正在生成预测动图...")
    ani = animation.FuncAnimation(fig, animate, frames=len(window_results),
                                  interval=1000//fps, repeat=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=80)
    print(f"动图已保存至: {save_path}")

def main():
    # 1. 加载数据
    series = load_data(DATA_PATH, TARGET_COL)
    print(f"数据总长度: {len(series)}")
    
    # 2. 划分训练集，预搜索全局最优阶数（仅执行一次，加速滑动预测）
    split_idx = int(len(series) * TRAIN_RATIO)
    train_series = series[:split_idx]
    print(f"训练集长度: {len(train_series)}，用于搜索最优ARIMA阶数")
    
    global best_order
    best_order = determine_best_order(train_series, P_VALUES, D_VALUES, Q_VALUES)
    
    # 3. 滑动窗口多步预测（和LSTM完全一致）
    global overall_smape
    window_results, overall_true, overall_pred, metrics = sliding_window_predict(
        series, SEQ_LENGTH, OUTPUT_SIZE, best_order
    )
    overall_mae, overall_rmse, overall_smape = metrics
    
    # 4. 保存模型和结果
    os.makedirs("result", exist_ok=True)
    # 保存最优阶数和结果
    result_dict = {
        "best_order": best_order,
        "window_results": window_results,
        "overall_metrics": {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "smape": overall_smape,
            "accuracy": 100 - overall_smape
        }
    }
    joblib.dump(result_dict, "result/arima_sliding_results.pkl")
    print(f"预测结果已保存至: result/arima_sliding_results.pkl")
    
    # 5. 绘制效果图
    create_prediction_plot(series, SEQ_LENGTH, OUTPUT_SIZE, window_results)
    
    # 6. 生成动图（可选，耗时较长，可注释）
    create_animation_gif(series, SEQ_LENGTH, OUTPUT_SIZE, window_results)

if __name__ == "__main__":
    main()