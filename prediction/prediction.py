"""
离线批量预测模块 - 调用训练好的.pth模型进行滑动窗口预测
"""

# ==================== 导入必要的库 ====================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
import os
import time
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 简易解决方案：动态添加所有需要的路径到sys.path
import os
import sys

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录（LSTM目录）
lstm_dir = os.path.dirname(current_dir)

# 将必要的路径添加到sys.path
paths_to_add = [lstm_dir, current_dir]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 现在可以直接导入所有模块
try:
    from setting import *
    print("✅ 成功导入setting模块")
except ImportError as e:
    print(f"❌ 导入setting模块失败: {e}")
    sys.exit(1)

try:
    from filepath import *
    print("✅ 成功导入filepath模块")
except ImportError as e:
    print(f"❌ 导入filepath模块失败: {e}")
    sys.exit(1)

try:
    from path import *  # 同目录的path.py
    print("✅ 成功导入path模块")
except ImportError as e:
    print(f"❌ 导入path模块失败: {e}")
    sys.exit(1)

# ==================== 辅助函数 ====================
def create_directories():
    """创建必要的目录结构"""
    if AUTO_CREATE_DIRS:
        for path in [BATCH_PREDICTION_OUTPUT_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"✅ 创建目录: {path}")

def check_paths():
    """检查所有必要路径是否存在"""
    issues = []
    
    # 检查模型文件
    if not os.path.exists(PREDICTION_MODEL_PATH):
        issues.append(f"❌ 模型文件不存在: {PREDICTION_MODEL_PATH}")
    else:
        print(f"✅ 模型文件: {PREDICTION_MODEL_PATH}")
    
    # 检查输入文件
    if not os.path.exists(BATCH_PREDICTION_INPUT_PATH):
        issues.append(f"❌ 输入文件不存在: {BATCH_PREDICTION_INPUT_PATH}")
    else:
        print(f"✅ 输入文件: {BATCH_PREDICTION_INPUT_PATH}")
    
    # 检查输出目录（自动创建）
    create_directories()
    
    # 检查输出文件是否已存在
    if os.path.exists(BATCH_PREDICTION_STATS_FILE):
        if OVERWRITE_EXISTING:
            print(f"⚠️  将覆盖已存在的统计文件: {BATCH_PREDICTION_STATS_FILE}")
        else:
            response = input(f"文件 {BATCH_PREDICTION_STATS_FILE} 已存在，是否覆盖？(y/n): ")
            if response.lower() != 'y':
                print("❌ 用户选择不覆盖，程序退出")
                sys.exit(0)
    
    if issues:
        print("\n⚠️  路径检查发现问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n请修改 path.py 中的路径配置")
        sys.exit(1)
    
    return True

# ==================== 模型加载函数 ====================
def load_trained_model(model_path):
    """
    加载训练好的.pth模型
    
    参数:
        model_path: 模型文件路径
    
    返回:
        model: 加载的模型
        model_params: 模型参数
        scaler_X: 输入归一化器
        scaler_y: 输出归一化器
    """
    print(f"\n正在加载模型: {model_path}")
    
    try:
        # 加载模型数据
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
        
        # 提取模型参数
        model_params = checkpoint.get('params', {})
        seq_length = checkpoint.get('seq_length', model_params.get('seq_length', 12))
        output_size = checkpoint.get('output_size', model_params.get('output_size', 5))
        hidden_size = checkpoint.get('hidden_size', model_params.get('hidden_size', 512))
        num_layers = checkpoint.get('num_layers', model_params.get('num_layers', 2))
        
        # 加载归一化器
        scaler_X = checkpoint.get('scaler_X')
        scaler_y = checkpoint.get('scaler_y')
        
        print(f"✅ 模型参数:")
        print(f"  - seq_length: {seq_length}")
        print(f"  - output_size: {output_size}")
        print(f"  - hidden_size: {hidden_size}")
        print(f"  - num_layers: {num_layers}")
        
        if scaler_X and scaler_y:
            print(f"✅ 归一化器已加载")
        
        # 创建模型实例
        class LSTMModelMultiStep(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, output_size=3, num_layers=3, dropout=0.2):
                super(LSTMModelMultiStep, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.output_size = output_size
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                lstm_out, (h_n, c_n) = self.lstm(x)
                last_time_step = lstm_out[:, -1, :]
                last_time_step = self.dropout(last_time_step)
                output = self.linear(last_time_step)
                return output
        
        # 创建模型实例
        dropout = model_params.get('dropout', 0.2)
        model = LSTMModelMultiStep(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 设置为评估模式
        
        print(f"✅ 模型加载成功")
        
        return model, model_params, seq_length, output_size, scaler_X, scaler_y
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==================== 数据加载函数 ====================
def load_prediction_data(file_path, target_column):
    """
    加载预测数据
    
    参数:
        file_path: 数据文件路径
        target_column: 目标列名
    
    返回:
        data_array: 目标列数据数组
        data_df: 原始DataFrame（用于后续处理）
    """
    print(f"\n正在加载预测数据: {file_path}")
    
    try:
        # 读取数据
        data_df = pd.read_csv(file_path)
        
        # 检查目标列是否存在
        if target_column not in data_df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        
        # 提取目标列
        data_array = data_df[target_column].values.astype(float)
        
        print(f"✅ 数据加载成功")
        print(f"  数据形状: {data_array.shape}")
        print(f"  数据范围: [{data_array.min():.4f}, {data_array.max():.4f}]")
        print(f"  数据前5个值: {data_array[:5]}")
        
        return data_array, data_df
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        sys.exit(1)

# ==================== 批量预测函数 ====================
def batch_predict_sliding_window(model, data_array, seq_length, output_size, scaler_X, scaler_y):
    """
    滑动窗口批量预测
    
    参数:
        model: 训练好的模型
        data_array: 输入数据数组
        seq_length: 序列长度
        output_size: 预测步长
        scaler_X: 输入归一化器
        scaler_y: 输出归一化器
    
    返回:
        predictions: 预测值数组（与输入数据同长度，未预测位置为NaN）
        prediction_indices: 预测索引列表（记录哪些位置有预测值）
    """
    print(f"\n开始滑动窗口批量预测...")
    print(f"  序列长度: {seq_length}")
    print(f"  预测步长: {output_size}")
    print(f"  总数据量: {len(data_array)}")
    
    # 初始化预测结果数组（与输入数据同长度）
    predictions = np.full(len(data_array), np.nan)
    
    # 记录预测时间
    total_time = 0
    total_predictions = 0
    
    # 滑动窗口预测
    for start_idx in range(len(data_array) - seq_length - output_size + 1):
        # 获取当前窗口数据
        window_data = data_array[start_idx:start_idx + seq_length]
        
        # 归一化窗口数据
        window_norm = scaler_X.transform(window_data.reshape(-1, 1)).reshape(1, seq_length, 1)
        window_tensor = torch.FloatTensor(window_norm)
        
        # 开始计时
        start_time = time.perf_counter()
        
        # 预测
        with torch.no_grad():
            pred_norm = model(window_tensor)
            pred = scaler_y.inverse_transform(pred_norm.numpy())
        
        # 结束计时
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        total_predictions += 1
        
        # 将预测结果填入对应位置
        pred_values = pred[0]
        for i in range(output_size):
            target_idx = start_idx + seq_length + i
            if target_idx < len(data_array):
                # 如果该位置已有预测值，取平均（或者保留第一个预测值）
                if np.isnan(predictions[target_idx]):
                    predictions[target_idx] = pred_values[i]
                else:
                    # 取两个预测值的平均
                    predictions[target_idx] = (predictions[target_idx] + pred_values[i]) / 2
        
        # 每100次预测打印一次进度
        if (start_idx + 1) % 100 == 0 or start_idx == 0:
            progress = (start_idx + 1) / (len(data_array) - seq_length - output_size + 1) * 100
            print(f"  进度: {start_idx + 1}/{len(data_array) - seq_length - output_size + 1} ({progress:.1f}%)")
    
    # 计算平均预测时间
    avg_time_ms = (total_time / total_predictions) * 1000 if total_predictions > 0 else 0
    
    print(f"✅ 批量预测完成")
    print(f"  总预测次数: {total_predictions}")
    print(f"  总预测耗时: {total_time:.2f}秒")
    print(f"  平均单次预测耗时: {avg_time_ms:.2f}毫秒")
    
    # 获取有预测值的索引
    prediction_indices = np.where(~np.isnan(predictions))[0]
    print(f"  有效预测点数: {len(prediction_indices)}")
    
    return predictions, prediction_indices

# ==================== 误差计算函数 ====================
def calculate_prediction_errors(actual_values, predicted_values, prediction_indices):
    """
    计算预测误差
    
    参数:
        actual_values: 真实值数组
        predicted_values: 预测值数组
        prediction_indices: 预测索引列表
    
    返回:
        errors: 误差字典，包含各种误差指标
        error_marks: 误差标记数组（异常/正常）
    """
    print(f"\n计算预测误差...")
    
    # 初始化误差标记数组
    error_marks = np.full(len(actual_values), "-")
    
    # 计算有效预测点的误差
    valid_indices = []
    absolute_errors = []
    relative_errors = []
    
    for idx in prediction_indices:
        if idx < len(actual_values):
            actual = actual_values[idx]
            predicted = predicted_values[idx]
            
            # 检查是否为有效数值
            if not np.isnan(actual) and not np.isnan(predicted):
                # 计算绝对误差
                abs_error = abs(actual - predicted)
                
                # 计算相对误差（MAPE）
                if abs(actual) > 1e-8:  # 避免除以0
                    rel_error = abs_error / abs(actual) * 100
                else:
                    rel_error = np.nan
                
                # 标记误差状态
                if abs_error > ABS_ERROR_THRESHOLD or (not np.isnan(rel_error) and rel_error > MAPE_THRESHOLD):
                    error_marks[idx] = "异常"
                else:
                    error_marks[idx] = "正常"
                
                # 记录有效误差
                if not np.isnan(rel_error):
                    valid_indices.append(idx)
                    absolute_errors.append(abs_error)
                    relative_errors.append(rel_error)
    
    # 计算整体误差指标
    if valid_indices:
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean(np.square(absolute_errors)))
        mape = np.mean(relative_errors)
        
        # 统计异常点
        abnormal_count = np.sum(error_marks == "异常")
        valid_count = len(valid_indices)
        abnormal_ratio = abnormal_count / valid_count * 100 if valid_count > 0 else 0
        
        errors = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'valid_count': valid_count,
            'abnormal_count': abnormal_count,
            'abnormal_ratio': abnormal_ratio,
            'absolute_errors': absolute_errors,
            'relative_errors': relative_errors,
            'valid_indices': valid_indices
        }
        
        print(f"✅ 误差计算完成")
        print(f"  有效预测点: {valid_count}")
        print(f"  整体MAE: {mae:.4f}")
        print(f"  整体RMSE: {rmse:.4f}")
        print(f"  整体MAPE: {mape:.2f}%")
        print(f"  异常点数量: {abnormal_count} ({abnormal_ratio:.1f}%)")
        
        return errors, error_marks
    else:
        print("⚠️  没有有效的预测点进行误差计算")
        return None, error_marks

# ==================== 生成统计报告函数 ====================
def generate_statistics_report(model_params, errors, data_array, predictions, error_marks, seq_length, output_size, model_path):
    """
    生成详细的误差统计报告
    
    参数:
        model_params: 模型参数
        errors: 误差指标字典
        data_array: 真实值数组
        predictions: 预测值数组
        error_marks: 误差标记数组
        seq_length: 序列长度
        output_size: 预测步长
        model_path: 模型文件路径
    
    返回:
        report_content: 报告内容字符串
    """
    print(f"\n生成误差统计报告...")
    
    # 获取模型文件名
    model_filename = os.path.basename(model_path)
    
    # 构建报告头部
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("                    批量预测误差统计报告")
    report_lines.append("=" * 70)
    report_lines.append(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"模型文件: {model_filename}")
    report_lines.append(f"输入文件: {os.path.basename(BATCH_PREDICTION_INPUT_PATH)}")
    report_lines.append("")
    
    # 模型参数信息
    report_lines.append("模型参数配置:")
    report_lines.append("-" * 40)
    for key, value in model_params.items():
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    if errors:
        # 整体预测精度汇总
        report_lines.append("整体预测精度汇总:")
        report_lines.append("-" * 40)
        report_lines.append(f"总数据量: {len(data_array)} 个时间点")
        report_lines.append(f"有效预测点: {errors['valid_count']} 个")
        report_lines.append(f"整体MAE: {errors['mae']:.4f}")
        report_lines.append(f"整体RMSE: {errors['rmse']:.4f}")
        report_lines.append(f"整体MAPE: {errors['mape']:.2f}%")
        report_lines.append(f"异常预测点占比: {errors['abnormal_ratio']:.1f}% ({errors['abnormal_count']}/{errors['valid_count']})")
        report_lines.append(f"误差阈值配置: ABS_ERROR_THRESHOLD={ABS_ERROR_THRESHOLD}, MAPE_THRESHOLD={MAPE_THRESHOLD}%")
        report_lines.append("")
    else:
        report_lines.append("整体预测精度汇总:")
        report_lines.append("-" * 40)
        report_lines.append("⚠️  没有有效的误差计算结果")
        report_lines.append("")
    
    # 详细预测结果表头
    report_lines.append("详细预测结果:")
    report_lines.append("-" * 70)
    report_lines.append("时间索引   真实值        预测值        误差标记")
    report_lines.append("-" * 70)
    
    # 生成详细数据行
    for i in range(len(data_array)):
        actual = data_array[i]
        predicted = predictions[i]
        mark = error_marks[i]
        
        # 格式化显示
        actual_str = f"{actual:.4f}" if not np.isnan(actual) else "-"
        
        if not np.isnan(predicted):
            predicted_str = f"{predicted:.4f}"
        else:
            predicted_str = "-"
        
        # 对齐格式
        idx_str = f"{i:4d}"
        actual_str = actual_str.rjust(10)
        predicted_str = predicted_str.rjust(10)
        mark_str = mark.rjust(6)
        
        report_lines.append(f"{idx_str}    {actual_str}    {predicted_str}    {mark_str}")
    
    report_lines.append("=" * 70)
    
    # 将列表转换为字符串
    report_content = "\n".join(report_lines)
    
    print(f"✅ 统计报告生成完成")
    
    return report_content

# ==================== 保存结果函数 ====================
def save_results_to_files(report_content, data_df, predictions, error_marks, errors):
    """
    将结果保存到文件
    
    参数:
        report_content: 统计报告内容
        data_df: 原始数据DataFrame
        predictions: 预测值数组
        error_marks: 误差标记数组
        errors: 误差指标字典
    """
    print(f"\n保存结果到文件...")
    
    try:
        # 1. 保存统计报告到.txt文件
        with open(BATCH_PREDICTION_STATS_FILE, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ 统计报告已保存: {BATCH_PREDICTION_STATS_FILE}")
        
        # 2. 保存完整数据到.csv文件
        result_df = data_df.copy()
        result_df['预测值'] = predictions
        result_df['误差标记'] = error_marks
        
        # 添加误差列（如果有有效预测）
        if errors and len(errors['valid_indices']) > 0:
            # 创建绝对误差和相对误差列
            abs_errors = np.full(len(predictions), np.nan)
            rel_errors = np.full(len(predictions), np.nan)
            
            for i, idx in enumerate(errors['valid_indices']):
                abs_errors[idx] = errors['absolute_errors'][i]
                rel_errors[idx] = errors['relative_errors'][i]
            
            result_df['绝对误差'] = abs_errors
            result_df['相对误差(%)'] = rel_errors
        
        result_df.to_csv(BATCH_PREDICTION_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 完整数据已保存: {BATCH_PREDICTION_DATA_FILE}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        return False

# ==================== 生成可视化图表函数 ====================
def generate_visualization_plot(data_array, predictions, error_marks, errors, seq_length, output_size):
    """
    生成预测误差可视化图表
    
    参数:
        data_array: 真实值数组
        predictions: 预测值数组
        error_marks: 误差标记数组
        errors: 误差指标字典
        seq_length: 序列长度
        output_size: 预测步长
    """
    print(f"\n生成可视化图表...")
    
    try:
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 获取有效预测点索引
        valid_indices = np.where(~np.isnan(predictions))[0]
        valid_predictions = predictions[valid_indices]
        valid_actuals = data_array[valid_indices]
        
        # 子图1: 真实值与预测值对比
        ax1 = axes[0, 0]
        ax1.plot(data_array, 'b-', label='真实值', linewidth=1.5, alpha=0.7)
        ax1.plot(valid_indices, valid_predictions, 'r--', label='预测值', linewidth=1.5, alpha=0.7, marker='o', markersize=3)
        
        # 标记异常点
        abnormal_indices = np.where(error_marks == "异常")[0]
        if len(abnormal_indices) > 0:
            ax1.scatter(abnormal_indices, data_array[abnormal_indices], 
                       c='red', s=50, marker='*', label='异常点', zorder=5)
        
        ax1.set_xlabel('时间索引')
        ax1.set_ylabel('数值')
        ax1.set_title('真实值与预测值对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 误差曲线
        ax2 = axes[0, 1]
        if errors and len(errors['valid_indices']) > 0:
            error_indices = errors['valid_indices']
            absolute_errors = errors['absolute_errors']
            
            ax2.bar(error_indices, absolute_errors, color='orange', alpha=0.6, label='绝对误差')
            ax2.axhline(y=ABS_ERROR_THRESHOLD, color='r', linestyle='--', label=f'阈值({ABS_ERROR_THRESHOLD})')
            ax2.set_xlabel('时间索引')
            ax2.set_ylabel('绝对误差')
            ax2.set_title('预测绝对误差分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 子图3: 相对误差（MAPE）分布
        ax3 = axes[1, 0]
        if errors and len(errors['valid_indices']) > 0:
            error_indices = errors['valid_indices']
            relative_errors = errors['relative_errors']
            
            ax3.bar(error_indices, relative_errors, color='green', alpha=0.6, label='相对误差(%)')
            ax3.axhline(y=MAPE_THRESHOLD, color='r', linestyle='--', label=f'阈值({MAPE_THRESHOLD}%)')
            ax3.set_xlabel('时间索引')
            ax3.set_ylabel('相对误差(%)')
            ax3.set_title('预测相对误差(MAPE)分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 子图4: 误差统计直方图
        ax4 = axes[1, 1]
        if errors and len(errors['absolute_errors']) > 0:
            # 绝对误差直方图
            ax4.hist(errors['absolute_errors'], bins=30, color='blue', alpha=0.6, label='绝对误差分布')
            ax4.axvline(x=ABS_ERROR_THRESHOLD, color='r', linestyle='--', label=f'阈值({ABS_ERROR_THRESHOLD})')
            ax4.set_xlabel('绝对误差')
            ax4.set_ylabel('频次')
            ax4.set_title('绝对误差分布直方图')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加整体标题
        plt.suptitle(f'LSTM多步预测误差分析 (seq_length={seq_length}, output_size={output_size})', 
                    fontsize=14, y=1.02)
        
        # 保存图表
        plt.savefig(BATCH_PREDICTION_VISUALIZATION_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存: {BATCH_PREDICTION_VISUALIZATION_PATH}")
        
        # 显示图表（可选）
        # plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ 生成可视化图表失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 主函数 ====================
def main():
    """
    离线批量预测主函数
    """
    print("=" * 70)
    print("LSTM多步预测 - 离线批量预测模块")
    print("=" * 70)
    
    # 1. 检查路径配置
    print("\n步骤1: 检查路径配置")
    print("-" * 40)
    check_paths()
    
    # 2. 加载训练好的模型
    print("\n步骤2: 加载训练模型")
    print("-" * 40)
    model, model_params, seq_length, output_size, scaler_X, scaler_y = load_trained_model(PREDICTION_MODEL_PATH)
    
    # 3. 加载预测数据
    print("\n步骤3: 加载预测数据")
    print("-" * 40)
    data_array, data_df = load_prediction_data(BATCH_PREDICTION_INPUT_PATH, TARGET_COLUMN)
    
    # 检查数据长度是否足够
    if len(data_array) < seq_length + output_size:
        print(f"❌ 数据长度不足！需要至少 {seq_length + output_size} 个数据点，当前只有 {len(data_array)} 个")
        sys.exit(1)
    
    # 4. 执行滑动窗口批量预测
    print("\n步骤4: 执行滑动窗口批量预测")
    print("-" * 40)
    predictions, prediction_indices = batch_predict_sliding_window(
        model, data_array, seq_length, output_size, scaler_X, scaler_y
    )
    
    # 5. 计算预测误差
    print("\n步骤5: 计算预测误差")
    print("-" * 40)
    errors, error_marks = calculate_prediction_errors(data_array, predictions, prediction_indices)
    
    # 6. 生成统计报告
    print("\n步骤6: 生成统计报告")
    print("-" * 40)
    report_content = generate_statistics_report(
        model_params, errors, data_array, predictions, error_marks, 
        seq_length, output_size, PREDICTION_MODEL_PATH
    )
    
    # 7. 保存结果到文件
    print("\n步骤7: 保存结果到文件")
    print("-" * 40)
    save_success = save_results_to_files(report_content, data_df, predictions, error_marks, errors)
    
    # 8. 生成可视化图表
    print("\n步骤8: 生成可视化图表")
    print("-" * 40)
    if errors:
        viz_success = generate_visualization_plot(
            data_array, predictions, error_marks, errors, seq_length, output_size
        )
    else:
        print("⚠️  没有有效的误差数据，跳过图表生成")
        viz_success = False
    
    # 9. 打印总结
    print("\n" + "=" * 70)
    print("预测完成总结")
    print("=" * 70)
    
    print(f"📊 输入数据: {len(data_array)} 个数据点")
    print(f"📊 有效预测: {len(prediction_indices)} 个预测值")
    
    if errors:
        print(f"📊 整体精度: MAPE = {errors['mape']:.2f}%")
        print(f"📊 异常比例: {errors['abnormal_ratio']:.1f}%")
    
    print(f"\n📁 生成文件:")
    print(f"  ✅ 统计报告: {BATCH_PREDICTION_STATS_FILE}")
    print(f"  ✅ 完整数据: {BATCH_PREDICTION_DATA_FILE}")
    if viz_success:
        print(f"  ✅ 可视化图表: {BATCH_PREDICTION_VISUALIZATION_PATH}")
    
    print("\n🎯 预测完成！")
    print("=" * 70)

# ==================== 执行主函数 ====================
if __name__ == "__main__":
    # 设置误差阈值（如果setting.py中没有，则使用默认值）
    try:
        from setting import ABS_ERROR_THRESHOLD, MAPE_THRESHOLD
    except ImportError:
        # 默认阈值
        ABS_ERROR_THRESHOLD = 0.5
        MAPE_THRESHOLD = 10.0
        print(f"⚠️  使用默认误差阈值: ABS_ERROR_THRESHOLD={ABS_ERROR_THRESHOLD}, MAPE_THRESHOLD={MAPE_THRESHOLD}%")
    
    main()