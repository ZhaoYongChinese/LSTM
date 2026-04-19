import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 从 test.py 导入核心函数
from test import load_model, compute_mape, predict_sequence


def generate_report(model_path, test_csv, target_col, default_seq_len=144, default_output_size=72, output_txt=None):
    """
    生成详细的预测报告文本文件
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型（传入默认序列长度和输出长度，兼容旧模型）
    print("正在加载模型...")
    model, scaler_X, scaler_y, seq_len, output_size = load_model(
        model_path, device,
        default_seq_len=default_seq_len,
        default_output_size=default_output_size
    )
    print(f"模型加载成功：输入长度 {seq_len}，输出长度 {output_size}")

    # 读取测试数据
    df = pd.read_csv(test_csv)
    if target_col not in df.columns:
        raise ValueError(f"CSV 中找不到列: {target_col}")
    data_series = df[target_col].values.astype(np.float32)
    total_len = len(data_series)

    # 计算最大滑动窗口数
    max_windows = total_len - seq_len - output_size
    if max_windows < 0:
        raise ValueError(f"数据长度不足，至少需要 {seq_len + output_size} 个点")

    # 收集所有预测结果
    all_true = []
    all_pred = []
    windows_info = []

    print(f"正在对 {max_windows + 1} 个窗口进行预测...")
    for i in range(max_windows + 1):
        hist = data_series[i : i + seq_len]
        future = data_series[i + seq_len : i + seq_len + output_size]
        pred = predict_sequence(model, scaler_X, scaler_y, hist, device)

        # 计算该窗口整体 MAPE
        window_mape = compute_mape(future, pred)

        # 记录每个点的误差
        point_errors = []
        for j in range(output_size):
            err = compute_mape(np.array([future[j]]), np.array([pred[j]]))
            point_errors.append(err)

        windows_info.append({
            'start_idx': i,
            'window_mape': window_mape,
            'future_true': future,
            'pred': pred,
            'point_errors': point_errors
        })

        all_true.append(future)
        all_pred.append(pred)

    # 计算整体 MAPE
    overall_true = np.concatenate(all_true)
    overall_pred = np.concatenate(all_pred)
    overall_mape = compute_mape(overall_true, overall_pred)

    # 生成报告内容
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LSTM 时序预测详细报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 模型信息
    report_lines.append("【模型信息】")
    report_lines.append(f"  模型文件: {os.path.basename(model_path)}")
    report_lines.append(f"  输入序列长度 (SEQ_LENGTH): {seq_len}")
    report_lines.append(f"  输出预测长度 (OUTPUT_SIZE): {output_size}")
    report_lines.append("")

    # 数据信息
    report_lines.append("【测试数据信息】")
    report_lines.append(f"  数据文件: {os.path.basename(test_csv)}")
    report_lines.append(f"  目标列: {target_col}")
    report_lines.append(f"  总数据点数: {total_len}")
    report_lines.append(f"  有效滑动窗口数: {max_windows + 1}")
    report_lines.append(f"  整体 MAPE: {overall_mape:.4f}%")
    report_lines.append(f"  整体准确率: {100 - overall_mape:.2f}%")
    report_lines.append("")

    # 定义辅助函数：格式化输出窗口详情
    def format_window_detail(win_idx, info):
        lines = []
        lines.append(f"第 {win_idx + 1} 次预测 (起始索引 {info['start_idx']})")
        lines.append(f"  本次预测整体 MAPE: {info['window_mape']:.4f}%")
        lines.append("-" * 60)
        lines.append(f"{'数据点':<8} {'真实值':<15} {'预测值':<15} {'该点 MAPE':<12}")
        lines.append("-" * 60)
        for j in range(output_size):
            point_idx = info['start_idx'] + seq_len + j
            true_val = info['future_true'][j]
            pred_val = info['pred'][j]
            err = info['point_errors'][j]
            lines.append(f"{point_idx:<8} {true_val:<15.6f} {pred_val:<15.6f} {err:<12.4f}%")
        lines.append("")
        return lines

    # 前5次预测
    report_lines.append("【前5次预测详情】")
    for i in range(min(5, len(windows_info))):
        report_lines.extend(format_window_detail(i, windows_info[i]))

    # 如果窗口数大于5，输出最后5次预测
    if len(windows_info) > 5:
        report_lines.append("【最后5次预测详情】")
        for i in range(len(windows_info) - 5, len(windows_info)):
            report_lines.extend(format_window_detail(i, windows_info[i]))

    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)

    # 写入文件
    if output_txt is None:
        output_txt = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n详细报告已保存至: {output_txt}")
    return output_txt


def main():
    parser = argparse.ArgumentParser(description='生成LSTM预测详细文本报告')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型文件路径')
    parser.add_argument('--test_csv', type=str, required=True, help='测试数据CSV文件路径')
    parser.add_argument('--target_col', type=str, default='RMS_Value', help='目标列名')
    parser.add_argument('--output', type=str, default=None, help='输出报告文件名（可选）')
    parser.add_argument('--seq_len', type=int, default=144, help='默认输入序列长度（用于旧模型）')
    parser.add_argument('--output_size', type=int, default=72, help='默认输出长度（用于旧模型）')

    args = parser.parse_args()

    generate_report(
        model_path=args.model_path,
        test_csv=args.test_csv,
        target_col=args.target_col,
        default_seq_len=args.seq_len,
        default_output_size=args.output_size,
        output_txt=args.output
    )


if __name__ == "__main__":
    main()