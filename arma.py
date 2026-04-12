import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def read_sensor_data(csv_file):
    """
    读取 CSV 文件，解析元数据和传感器数据。
    文件格式：
        第1行：采样频率，如 "100 Hz"
        第2行：总采样点数，整数
        第3行：采样时间，浮点数（单位秒）
        后续行：每个采样点的数值
    返回:
        sampling_freq (float): 采样频率 (Hz)
        total_points (int): 总采样点数
        sampling_time (float): 采样时间 (s)
        data (np.ndarray): 传感器数据数组
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError("文件行数不足，至少需要4行（3行元数据 + 至少1个数据点）")

    # 解析元数据
    sampling_freq_str = lines[0].strip()
    try:
        sampling_freq = float(sampling_freq_str.split()[0])  # 提取数值部分
    except:
        raise ValueError("采样频率格式错误，应为如 '100 Hz'")

    total_points = int(lines[1].strip())
    sampling_time = float(lines[2].strip())

    # 读取数据
    data = []
    for line in lines[3:]:
        line = line.strip()
        if line:
            try:
                data.append(float(line))
            except ValueError:
                print(f"警告：无法解析行 '{line}'，跳过")
    data = np.array(data)

    # 验证数据长度是否与 total_points 一致
    if len(data) != total_points:
        print(f"警告：实际数据点数 {len(data)} 与声明的总采样点数 {total_points} 不一致")

    return sampling_freq, total_points, sampling_time, data

def arma_forecast(data, steps, p, q, n_points=None):
    """
    使用 ARMA(p, q) 模型进行预测。
    参数:
        data (np.ndarray): 完整数据序列
        steps (int): 预测步长
        p (int): AR 阶数
        q (int): MA 阶数
        n_points (int): 用于建模的历史点数。若为 None，则使用全部数据
    返回:
        forecast_values (np.ndarray): 预测值数组，长度为 steps
        model (ARIMA): 拟合好的模型（可用于诊断）
    """
    if n_points is not None and n_points < len(data):
        # 使用最近 n_points 个点
        train_data = data[-n_points:]
    else:
        train_data = data

    # 使用 ARIMA(d=0) 实现 ARMA(p,q)
    model = ARIMA(train_data, order=(p, 0, q))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return np.array(forecast), fitted_model

def main():
    print("=== ARMA 时间序列预测工具 ===")

    # 1. 获取输入文件
    input_file = input("请输入传感器数据文件路径（CSV格式）: ").strip()
    try:
        sampling_freq, total_points, sampling_time, data = read_sensor_data(input_file)
        print(f"读取完成：采样频率={sampling_freq} Hz，总点数={total_points}，采样时间={sampling_time} s")
    except Exception as e:
        print(f"读取文件失败：{e}")
        return

    # 2. 获取预测参数
    steps = int(input("请输入预测步长（未来多少个点）: ").strip())
    n_points_str = input("请输入用于建模的历史点数（直接回车则使用全部数据）: ").strip()
    n_points = int(n_points_str) if n_points_str else None

    p = int(input("请输入 AR 阶数 p (默认 1): ").strip() or "1")
    q = int(input("请输入 MA 阶数 q (默认 1): ").strip() or "1")

    # 3. 执行预测
    print(f"\n使用 ARMA({p},{q}) 模型，基于最近 {n_points if n_points else '全部'} 个点，预测未来 {steps} 步...")
    try:
        forecast, model = arma_forecast(data, steps, p, q, n_points)
        print("预测结果：")
        for i, val in enumerate(forecast, 1):
            print(f"  第 {i} 步: {val:.6f}")

        # 可选：保存预测结果到文件
        save = input("\n是否将预测结果保存到文件？(y/n): ").strip().lower()
        if save == 'y':
            out_file = input("请输入输出文件名（默认 forecast.csv）: ").strip() or "forecast.csv"
            # 保存为单列 CSV，不含元数据
            np.savetxt(out_file, forecast, fmt='%.6f', header="Forecast", comments='')
            print(f"预测结果已保存到 {out_file}")

    except Exception as e:
        print(f"预测失败：{e}")
        print("提示：如果数据量太少，请减少 p/q 或增加历史点数。")

if __name__ == "__main__":
    main()