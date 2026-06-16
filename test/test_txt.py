import os
import numpy as np
from datetime import datetime

def generate_report(model, scaler_X, scaler_y, seq_len, output_size,
                    data_series, model_name, test_csv_name, target_col, output_dir, device):
    """
    生成详细的预测报告文本文件，并保存到指定目录。
    作为模块被 test.py 调用，直接复用内存中的模型，无需重新加载。
    """
    from test import compute_mape, predict_sequence

    total_len = len(data_series)
    max_windows = total_len - seq_len - output_size

    if max_windows < 0:
        print(f"数据长度不足以进行至少一次完整的滑动窗口预测，跳过文本报告生成。")
        return

    all_true, all_pred, windows_info = [], [], []

    print(f"正在生成详细文本报告，共计 {max_windows + 1} 个窗口...")
    for i in range(max_windows + 1):
        hist = data_series[i : i + seq_len]
        future = data_series[i + seq_len : i + seq_len + output_size]
        pred = predict_sequence(model, scaler_X, scaler_y, hist, device)

        # 计算 MAPE 误差
        window_mape = compute_mape(future, pred)
        point_errors = [compute_mape(np.array([future[j]]), np.array([pred[j]])) for j in range(output_size)]

        windows_info.append({
            'original_idx': i,  # 记录这原本是第几次预测
            'start_idx': i,
            'window_mape': window_mape,
            'future_true': future,
            'pred': pred,
            'point_errors': point_errors
        })

        all_true.append(future)
        all_pred.append(pred)

    overall_true = np.concatenate(all_true)
    overall_pred = np.concatenate(all_pred)
    overall_mape = compute_mape(overall_true, overall_pred)

    # 组装文本行
    report_lines = [
        "=" * 80,
        "LSTM 时序预测详细报告",
        "=" * 80,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "【模型与数据信息】",
        f"  模型名称: {model_name}",
        f"  数据来源: {test_csv_name}",
        f"  目标字段: {target_col}",
        f"  滑动窗口总数: {max_windows + 1}",
        f"  整体 MAPE: {overall_mape:.4f}%",
        f"  整体准确率: {100 - overall_mape:.2f}%\n"
    ]

    def format_window(idx, info):
        lines = [
            f"第 {idx + 1} 次预测 (起始索引 {info['start_idx']}) | 窗口 MAPE: {info['window_mape']:.4f}%",
            "-" * 60,
            f"{'点位索引':<8} {'真实值':<15} {'预测值':<15} {'单点 MAPE':<12}",
            "-" * 60
        ]
        for j in range(output_size):
            pt_idx = info['start_idx'] + seq_len + j
            lines.append(f"{pt_idx:<8} {info['future_true'][j]:<15.6f} {info['pred'][j]:<15.6f} {info['point_errors'][j]:<12.4f}%")
        lines.append("")
        return lines

    # ---------- 🆕 先记录 MAPE > 5% 的异常窗口 ----------
    high_error_windows = [info for info in windows_info if info['window_mape'] > 5.0]
    
    if high_error_windows:
        report_lines.append(f"【异常预测记录 (窗口 MAPE > 5%, 共计 {len(high_error_windows)} 次)】")
        for info in high_error_windows:
            report_lines.extend(format_window(info['original_idx'], info))
    else:
        report_lines.append("【异常预测记录 (窗口 MAPE > 5%)】\n  所有预测窗口的 MAPE 均未超过 5%。\n")

    # ---------- 记录详细预测数据 (前5次与最后5次) ----------
    report_lines.append("【详细预测数据 (截取前5次与最后5次)】")

    # 前5次
    for i in range(min(5, len(windows_info))):
        report_lines.extend(format_window(i, windows_info[i]))

    # 最后5次
    if len(windows_info) > 5:
        report_lines.append("...\n【最后5次预测详情】")
        # 防止首尾重叠，确保取到的不包含前5次已取过的部分
        start_idx_for_last_5 = max(5, len(windows_info) - 5)
        for i in range(start_idx_for_last_5, len(windows_info)):
            report_lines.extend(format_window(i, windows_info[i]))

    # 保存文件到动态生成的模型文件夹中
    output_txt_path = os.path.join(output_dir, f"{model_name}_report.txt")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[OK] 详细报告已保存至: {output_txt_path}")