# ==================== 导入必要的库 ====================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
import time
import itertools
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入配置
from setting import *
from filepath import *

# 设置随机种子，保证结果可复现
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 准备数据（多步预测版本） ====================
def load_and_prepare_data_multi_step(file_path, target_column='RATE', seq_length=6, output_size=3):
    """
    加载数据并准备为多步预测LSTM所需的格式
    
    参数:
        file_path: 数据文件路径
        target_column: 目标列名
        seq_length: 序列长度（回溯步数）
        output_size: 预测步长（预测未来几个时间点）
    
    返回:
        X: 输入序列 [样本数, seq_length, 特征数]
        y: 目标值 [样本数, output_size]
    """
    # 1. 加载数据
    print(f"正在加载数据: {file_path}")
    data = pd.read_csv(file_path)
    
    # 2. 提取目标列
    if target_column not in data.columns:
        raise ValueError(f"数据中不存在列: {target_column}")
    
    target_series = data[target_column].values.astype(float)
    print(f"数据形状: {target_series.shape}, 数据范围: [{target_series.min():.2f}, {target_series.max():.2f}]")
    
    # 3. 构造多步预测序列数据
    X, y = [], []
    # 注意：需要确保有足够的后续数据点
    for i in range(len(target_series) - seq_length - output_size + 1):
        X.append(target_series[i:i+seq_length])
        # 取接下来output_size个点作为目标
        y.append(target_series[i+seq_length:i+seq_length+output_size])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, output_size)  # 形状变为[样本数, output_size]
    
    print(f"构造后的数据形状 - X: {X.shape}, y: {y.shape}")
    print(f"预测未来 {output_size} 个时间点")
    return X, y

# ==================== 数据归一化与划分（多步预测版本） ====================
def prepare_datasets_multi_step(X, y, test_size=0.15, val_size=0.15):
    """
    准备多步预测的训练集、验证集和测试集，并进行归一化
    
    参数:
        X: 输入特征 [样本数, seq_length, 1]
        y: 目标值 [样本数, output_size]
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
    
    返回:
        归一化后的各个数据集和归一化器
    """
    # 1. 划分训练+验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # 时间序列数据不随机打乱
    )
    
    # 2. 从训练+验证集中再划分出验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, shuffle=False
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  目标值维度: {y_train.shape[1]} (预测步长)")
    
    # 3. 归一化（只使用训练集计算参数）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑X以便归一化
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[2])
    
    # 拟合并转换
    X_train_norm = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_norm = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    X_test_norm = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    
    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)
    
    # 转换为PyTorch张量
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

# ==================== 搭建多步预测模型 ====================
class LSTMModelMultiStep(nn.Module):
    """
    多步预测LSTM模型定义
    输出维度为output_size，预测未来多个时间点
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=3, num_layers=3, dropout=0.2):
        super(LSTMModelMultiStep, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层 - 输出维度为output_size
        self.linear = nn.Linear(hidden_size, output_size)
        
        print(f"初始化多步LSTM模型:")
        print(f"  输入特征: {input_size}")
        print(f"  隐藏层大小: {hidden_size}")
        print(f"  LSTM层数: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  输出大小: {output_size} (预测未来{output_size}个时间点)")
    
    def forward(self, x):
        # x形状: [batch_size, seq_length, input_size]
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out形状: [batch_size, seq_length, hidden_size]
        
        # 只取最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]
        
        # Dropout
        last_time_step = self.dropout(last_time_step)
        
        # 全连接层 - 输出多个时间点的预测
        output = self.linear(last_time_step)  # 形状: [batch_size, output_size]
        
        return output

# ==================== 训练函数（适用于多步预测） ====================
def train_model_multi_step(model, train_data, val_data, epochs=500, lr=0.001, patience=30):
    """
    训练多步预测模型
    
    参数:
        model: LSTM模型
        train_data: 训练数据 (X_train, y_train)
        val_data: 验证数据 (X_val, y_val)
        epochs: 最大训练轮数
        lr: 学习率
        patience: 早停耐心值
    
    返回:
        model: 训练好的模型
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        best_epoch: 最佳模型对应的轮数
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失，适用于多输出
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器（每20轮衰减到原来的0.9）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    print(f"\n开始训练多步预测模型...")
    print(f"  最大轮数: {epochs}")
    print(f"  初始学习率: {lr}")
    print(f"  早停耐心值: {patience}")
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        y_pred_train = model(X_train)
        train_loss = criterion(y_pred_train, y_train)
        
        # 反向传播
        optimizer.zero_grad()
        train_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 验证模式
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
        
        # 记录损失
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 每50轮打印一次训练信息
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'轮次 [{epoch+1:4d}/{epochs}] | '
                  f'训练损失: {train_loss.item():.6f} | '
                  f'验证损失: {val_loss.item():.6f} | '
                  f'学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 早停判断
        if patience_counter >= patience:
            print(f'\n早停触发！在轮次 {epoch+1} 停止训练。')
            print(f'最佳模型在轮次 {best_epoch+1}，验证损失: {best_val_loss:.6f}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'训练完成！最佳验证损失: {best_val_loss:.6f}')
    
    return model, train_losses, val_losses, best_epoch

# ==================== 评估函数（多步预测版本） ====================
def evaluate_model_multi_step(model, X_test, y_test, scaler, seq_length):
    """
    评估多步预测模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试集输入
        y_test: 测试集真实值（归一化后的）
        scaler: 归一化器
        seq_length: 序列长度
    
    返回:
        评估指标和预测结果
    """
    model.eval()
    with torch.no_grad():
        # 预测
        y_pred_norm = model(X_test)
        
        # 反归一化
        y_pred = scaler.inverse_transform(y_pred_norm.numpy())
        y_true = scaler.inverse_transform(y_test.numpy())
    
    # 获取输出维度
    output_size = y_true.shape[1]
    
    print(f"\n模型评估结果 (预测未来{output_size}步):")
    
    # 为每个预测步长单独计算指标
    step_results = {}
    for step in range(output_size):
        mae = np.mean(np.abs(y_true[:, step] - y_pred[:, step]))
        rmse = np.sqrt(np.mean((y_true[:, step] - y_pred[:, step]) ** 2))
        mape = np.mean(np.abs((y_true[:, step] - y_pred[:, step]) / (y_true[:, step] + 1e-8))) * 100
        
        step_results[f'step_{step+1}'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # 整体指标
    overall_mape = np.mean([step_results[f'step_{i+1}']['MAPE'] for i in range(output_size)])
    
    return y_true, y_pred, step_results, overall_mape

# ==================== 批量预测耗时统计函数 ====================
def batch_prediction_time_stats(model, X_test, y_test, scaler_y):
    """
    统计批量预测的耗时
    
    参数:
        model: 训练好的模型
        X_test: 测试集输入
        y_test: 测试集真实值（归一化后的）
        scaler_y: 输出归一化器
        
    返回:
        dict: 包含批量预测耗时统计信息的字典
    """
    print(f"\n开始批量预测耗时统计...")
    print(f"测试集样本数: {len(X_test)}")
    
    model.eval()
    
    # 开始计时
    batch_start_time = time.perf_counter()
    
    with torch.no_grad():
        # 批量预测
        y_pred_norm = model(X_test)
        # 反归一化（包含在计时内，因为这也是预测过程的一部分）
        y_pred = scaler_y.inverse_transform(y_pred_norm.numpy())
    
    # 结束计时
    batch_end_time = time.perf_counter()
    
    # 计算总耗时（秒）
    total_time_seconds = batch_end_time - batch_start_time
    
    # 计算平均单样本耗时（毫秒）
    avg_time_per_sample_ms = (total_time_seconds / len(X_test)) * 1000
    
    print(f"批量预测完成")
    print(f"  总耗时: {total_time_seconds:.4f} 秒")
    print(f"  平均单样本耗时: {avg_time_per_sample_ms:.4f} 毫秒")
    
    return {
        'total_time_seconds': total_time_seconds,
        'num_samples': len(X_test),
        'avg_time_per_sample_ms': avg_time_per_sample_ms,
        'timestamp': datetime.now()
    }

# ==================== 保存耗时统计结果 ====================
def save_time_stats_to_file(params_dict, step_results, overall_mape, 
                           single_pred_time_ms, batch_stats, model_filename):
    """
    将耗时统计结果保存到txt文件
    
    参数:
        params_dict: 参数字典
        step_results: 步长评估结果
        overall_mape: 整体MAPE
        single_pred_time_ms: 单次预测耗时（毫秒）
        batch_stats: 批量预测统计信息字典
        model_filename: 模型文件名
    """
    # 根据模型文件名生成统计文件名
    stats_filename = model_filename.replace('.pth', '.txt')
    
    try:
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("预测耗时统计报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"统计时间: {batch_stats['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("模型参数:\n")
            f.write("-" * 40 + "\n")
            for key, value in params_dict.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"  整体MAPE: {overall_mape:.2f}%\n")
            f.write(f"  整体精度: {100 - overall_mape:.2f}%\n\n")
            
            f.write("分步评估结果:\n")
            f.write("-" * 40 + "\n")
            for step in step_results.keys():
                results = step_results[step]
                step_num = int(step.split('_')[1])
                f.write(f"  第{step_num}步预测 (t+{step_num}):\n")
                f.write(f"    MAE: {results['MAE']:.4f}\n")
                f.write(f"    RMSE: {results['RMSE']:.4f}\n")
                f.write(f"    MAPE: {results['MAPE']:.2f}%\n")
                f.write(f"    精度: {100 - results['MAPE']:.2f}%\n")
            f.write("\n")
            
            f.write("单次预测耗时统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  单次示例预测耗时: {single_pred_time_ms:.4f} 毫秒\n\n")
            
            f.write("批量预测耗时统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  测试集样本总数: {batch_stats['num_samples']}\n")
            f.write(f"  批量预测总耗时: {batch_stats['total_time_seconds']:.4f} 秒\n")
            f.write(f"  平均单样本预测耗时: {batch_stats['avg_time_per_sample_ms']:.4f} 毫秒\n\n")
            
            f.write("性能评估:\n")
            f.write("-" * 40 + "\n")
            
            # 根据耗时评估性能
            if batch_stats['avg_time_per_sample_ms'] < 1.0:
                f.write("  ✅ 预测性能优秀：平均单样本预测耗时 < 1毫秒\n")
            elif batch_stats['avg_time_per_sample_ms'] < 10.0:
                f.write("  ✅ 预测性能良好：平均单样本预测耗时 < 10毫秒\n")
            elif batch_stats['avg_time_per_sample_ms'] < 50.0:
                f.write("  ⚠️  预测性能一般：平均单样本预测耗时 < 50毫秒\n")
            else:
                f.write("  ⚠️  预测性能较慢：平均单样本预测耗时 ≥ 50毫秒\n")
            
            # 根据MAPE评估模型性能
            if overall_mape < 5.0:
                f.write("  ✅ 模型性能优秀，可用于实际故障预测！\n")
            elif overall_mape < 10.0:
                f.write("  ✅ 模型性能良好，可用于实际故障预测！\n")
            elif overall_mape < 15.0:
                f.write("  ⚠️  模型性能一般，建议进一步优化参数。\n")
            else:
                f.write("  ❌ 模型性能较差，需要重新设计或更多数据。\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"耗时统计结果已保存到: {stats_filename}")
        return stats_filename
        
    except Exception as e:
        print(f"保存耗时统计文件时发生错误: {e}")
        return None

# ==================== 创建目录结构 ====================
def create_directory_structure():
    """创建输出目录结构"""
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        print(f"创建输出根目录: {OUTPUT_ROOT}")
    
    # 创建子文件夹
    for folder in SUB_FOLDERS:
        folder_path = os.path.join(OUTPUT_ROOT, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建子文件夹: {folder_path}")

# ==================== 生成参数组合 ====================
def generate_param_combinations():
    """生成所有参数组合的笛卡尔积"""
    # 获取所有参数列表
    param_grid = {
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'patience': PATIENCE,
        'seq_length': SEQ_LENGTH,
        'output_size': OUTPUT_SIZE
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # 使用itertools.product生成笛卡尔积
    param_combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        param_combinations.append(param_dict)
    
    print(f"生成了 {len(param_combinations)} 组参数组合")
    return param_combinations

# ==================== 生成模型文件名 ====================
def generate_model_filename(params_dict):
    """根据参数生成模型文件名"""
    filename = f"lstm_h{params_dict['hidden_size']}_l{params_dict['num_layers']}"
    filename += f"_e{params_dict['epochs']}_lr{params_dict['learning_rate']}"
    filename += f"_p{params_dict['patience']}_s{params_dict['seq_length']}"
    filename += f"_o{params_dict['output_size']}_d{params_dict['dropout']}.pth"
    return filename

# ==================== 单组参数训练函数 ====================
def train_single_model(params_dict, model_results):
    """训练单组参数模型"""
    
    print("\n" + "=" * 60)
    print(f"当前训练参数组合:")
    print("-" * 40)
    for key, value in params_dict.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    try:
        # 1. 加载和准备多步预测数据
        X, y = load_and_prepare_data_multi_step(
            DATA_PATH, TARGET_COLUMN, 
            params_dict['seq_length'], params_dict['output_size']
        )
        
        # 2. 数据归一化和划分
        (X_train, y_train, 
         X_val, y_val, 
         X_test, y_test, 
         scaler_X, scaler_y) = prepare_datasets_multi_step(X, y, TEST_SIZE, VAL_SIZE)
        
        # 3. 创建多步预测模型
        model = LSTMModelMultiStep(
            input_size=INPUT_SIZE,
            hidden_size=params_dict['hidden_size'],
            output_size=params_dict['output_size'],
            num_layers=params_dict['num_layers'],
            dropout=params_dict['dropout']
        )
        
        # 4. 训练多步预测模型
        model, train_losses, val_losses, best_epoch = train_model_multi_step(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=params_dict['epochs'],
            lr=params_dict['learning_rate'],
            patience=params_dict['patience']
        )
        
        # 5. 评估多步预测模型
        y_true, y_pred, step_results, overall_mape = evaluate_model_multi_step(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler_y,
            seq_length=params_dict['seq_length']
        )
        
        # 6. 示例预测（带耗时统计）
        sample_idx = 0
        sample_input_norm = X_test[sample_idx].numpy().flatten()
        sample_input_original = scaler_X.inverse_transform(
            sample_input_norm.reshape(-1, 1)
        ).flatten()
        
        # 进行单次预测并统计耗时
        model.eval()
        start_time = time.perf_counter()
        input_seq = sample_input_original[-params_dict['seq_length']:].reshape(1, params_dict['seq_length'], 1)
        input_seq_norm = scaler_X.transform(input_seq.reshape(-1, 1)).reshape(1, params_dict['seq_length'], 1)
        input_tensor = torch.FloatTensor(input_seq_norm)
        with torch.no_grad():
            pred_norm = model(input_tensor)
            pred = scaler_y.inverse_transform(pred_norm.numpy())
        end_time = time.perf_counter()
        single_pred_time_ms = (end_time - start_time) * 1000
        
        # 7. 批量预测耗时统计
        batch_stats = batch_prediction_time_stats(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y
        )
        
        # 8. 保存模型文件
        model_filename = generate_model_filename(params_dict)
        temp_model_path = os.path.join(OUTPUT_ROOT, f"temp_{model_filename}")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': params_dict,
            'seq_length': params_dict['seq_length'],
            'output_size': params_dict['output_size'],
            'hidden_size': params_dict['hidden_size'],
            'num_layers': params_dict['num_layers'],
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'step_results': step_results,
            'overall_mape': overall_mape
        }, temp_model_path)
        
        # 9. 保存耗时统计文件
        stats_filename = save_time_stats_to_file(
            params_dict, step_results, overall_mape,
            single_pred_time_ms, batch_stats, temp_model_path
        )
        
        # 10. 记录结果
        model_results.append({
            'params': params_dict,
            'overall_mape': overall_mape,
            'avg_time_per_sample_ms': batch_stats['avg_time_per_sample_ms'],
            'model_path': temp_model_path,
            'stats_path': stats_filename if stats_filename else None
        })
        
        print(f"✅ 参数组合训练完成，整体MAPE: {overall_mape:.2f}%")
        
    except Exception as e:
        print(f"❌ 参数组合训练失败: {e}")
        import traceback
        traceback.print_exc()

# ==================== 分级存储结果 ====================
def store_results_by_rank(model_results):
    """根据MAPE排序并分级存储结果"""
    
    if not model_results:
        print("没有训练完成的模型")
        return
    
    # 按MAPE排序（越小越好）
    model_results.sort(key=lambda x: x['overall_mape'])
    
    print("\n" + "=" * 60)
    print("模型性能排序结果（按MAPE升序）:")
    print("=" * 60)
    for i, result in enumerate(model_results):
        print(f"{i+1}. MAPE: {result['overall_mape']:.2f}% | "
              f"耗时: {result['avg_time_per_sample_ms']:.2f}ms | "
              f"参数: h{result['params']['hidden_size']}_l{result['params']['num_layers']}")
    
    # 复制文件到对应目录
    for i, result in enumerate(model_results):
        if i == 0:  # 第一名
            target_folder = 'first'
        elif i == 1:  # 第二名
            target_folder = 'second'
        else:  # 其他
            target_folder = 'other'
        
        target_dir = os.path.join(OUTPUT_ROOT, target_folder)
        
        # 复制模型文件
        model_filename = os.path.basename(result['model_path'])
        target_model_path = os.path.join(target_dir, model_filename)
        try:
            shutil.copy2(result['model_path'], target_model_path)
            print(f"✅ 复制模型文件到 {target_folder}: {model_filename}")
        except Exception as e:
            print(f"❌ 复制模型文件失败: {e}")
        
        # 复制统计文件
        if result['stats_path']:
            stats_filename = os.path.basename(result['stats_path'])
            target_stats_path = os.path.join(target_dir, stats_filename)
            try:
                shutil.copy2(result['stats_path'], target_stats_path)
                print(f"✅ 复制统计文件到 {target_folder}: {stats_filename}")
            except Exception as e:
                print(f"❌ 复制统计文件失败: {e}")
        
        # 删除临时文件
        try:
            os.remove(result['model_path'])
            if result['stats_path']:
                os.remove(result['stats_path'])
        except Exception as e:
            print(f"⚠️  删除临时文件失败: {e}")
    
    print("\n" + "=" * 60)
    print("分级存储完成:")
    print(f"  first文件夹: 存储MAPE最低的模型 ({model_results[0]['overall_mape']:.2f}%)")
    if len(model_results) > 1:
        print(f"  second文件夹: 存储MAPE第二低的模型 ({model_results[1]['overall_mape']:.2f}%)")
    print(f"  other文件夹: 存储其余 {max(0, len(model_results)-2)} 个模型")
    print("=" * 60)

# ==================== 主函数 ====================
def main():
    """主函数"""
    
    print("=" * 60)
    print("多步LSTM故障信号预测 - 参数遍历训练")
    print("=" * 60)
    
    # 1. 创建目录结构
    create_directory_structure()
    
    # 2. 生成参数组合
    param_combinations = generate_param_combinations()
    
    # 3. 存储所有模型结果
    model_results = []
    
    # 4. 遍历所有参数组合进行训练
    total_combinations = len(param_combinations)
    for i, params_dict in enumerate(param_combinations):
        print(f"\n📊 进度: {i+1}/{total_combinations} ({((i+1)/total_combinations*100):.1f}%)")
        train_single_model(params_dict, model_results)
    
    # 5. 分级存储结果
    if model_results:
        store_results_by_rank(model_results)
        
        # 打印最佳模型信息
        best_model = model_results[0]
        print(f"\n🎯 最佳模型参数:")
        for key, value in best_model['params'].items():
            print(f"  {key}: {value}")
        print(f"🎯 最佳模型性能: MAPE = {best_model['overall_mape']:.2f}%")
        print(f"🎯 最佳模型路径: {os.path.join(OUTPUT_ROOT, 'first', os.path.basename(best_model['model_path']))}")
    else:
        print("\n❌ 没有成功训练的模型")
    
    print("\n" + "=" * 60)
    print("所有训练完成！")
    print("=" * 60)

# ==================== 执行主函数 ====================
if __name__ == "__main__":
    main()