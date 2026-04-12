# ==================== 导入必要的库 ====================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

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
            
            # 保存最佳模型到文件
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.item(),
                'val_loss': val_loss.item(),
                'output_size': model.output_size
            }, 'best_lstm_multi_step_model.pth')
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
        
        print(f"\n第{step+1}步预测结果 (t+{step+1}):")
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        print(f"  均方根误差 (RMSE): {rmse:.4f}")
        print(f"  平均绝对百分比误差 (MAPE): {mape:.2f}%")
    
    # 整体指标
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    overall_mape = np.mean([step_results[f'step_{i+1}']['MAPE'] for i in range(output_size)])
    
    print(f"\n{'='*50}")
    print(f"整体预测结果:")
    print(f"  平均绝对误差 (MAE): {overall_mae:.4f}")
    print(f"  均方根误差 (RMSE): {overall_rmse:.4f}")
    print(f"  平均绝对百分比误差 (MAPE): {overall_mape:.2f}%")
    print(f"  平均相对精度: {100 - overall_mape:.2f}%")
    
    return y_true, y_pred, step_results

# ==================== 预测函数（带耗时统计） ====================
def make_multi_step_predictions(model, recent_data, scaler_X, scaler_y, seq_length=6):
    """
    使用训练好的模型进行多步预测，并统计单次预测耗时
    
    参数:
        model: 训练好的模型
        recent_data: 最近的历史数据 [seq_length,]
        scaler_X: 输入归一化器
        scaler_y: 输出归一化器
        seq_length: 序列长度
    
    返回:
        tuple: (预测结果列表, 预测耗时毫秒数)
    """
    model.eval()
    
    # 开始计时（使用高精度计时器）
    start_time = time.perf_counter()
    
    # 准备输入数据
    input_seq = recent_data[-seq_length:].reshape(1, seq_length, 1)
    
    # 归一化
    input_seq_norm = scaler_X.transform(input_seq.reshape(-1, 1)).reshape(1, seq_length, 1)
    input_tensor = torch.FloatTensor(input_seq_norm)
    
    # 进行预测
    with torch.no_grad():
        pred_norm = model(input_tensor)
        pred = scaler_y.inverse_transform(pred_norm.numpy())
    
    # 结束计时
    end_time = time.perf_counter()
    
    # 计算耗时（转换为毫秒）
    elapsed_time_ms = (end_time - start_time) * 1000
    
    # 获取预测结果
    predictions = pred[0].tolist()
    
    return predictions, elapsed_time_ms

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
def save_time_stats_to_file(single_pred_time_ms, batch_stats, filename='prediction_time_stats.txt'):
    """
    将耗时统计结果保存到txt文件
    
    参数:
        single_pred_time_ms: 单次预测耗时（毫秒）
        batch_stats: 批量预测统计信息字典
        filename: 输出文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("预测耗时统计报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"统计时间: {batch_stats['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
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
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"耗时统计结果已保存到: {filename}")
        return True
        
    except Exception as e:
        print(f"保存耗时统计文件时发生错误: {e}")
        return False

# ==================== 主函数 ====================
def main():
    # 1. 参数设置
    data_path = r'C:\Users\asus\Desktop\hehe\直梯\赵勇\代码\LSTM模板\多步预测模板\data\csv_output\正常_sliding_window.csv'
    target_column = 'RMS_Value'
    seq_length = 12  # 序列长度（回溯步数）
    output_size = 5  # 🎯 预测未来任意多个时间点，这里设置为5
    
    # 模型参数
    input_size = 1  # 输入特征维度
    hidden_size = 512  # LSTM隐藏层大小
    num_layers = 2  # LSTM层数
    dropout = 0.2  # Dropout概率
    
    # 训练参数
    epochs = 500  # 最大训练轮数
    learning_rate = 0.005  # 学习率
    patience = 30  # 早停耐心值
    
    print("=" * 60)
    print(f"多步LSTM故障信号预测 (预测步长={output_size})")
    print("=" * 60)
    
    # 2. 加载和准备多步预测数据
    print("\n步骤 1: 加载和准备多步预测数据")
    print("-" * 40)
    
    X, y = load_and_prepare_data_multi_step(
        data_path, target_column, seq_length, output_size
    )
    
    # 3. 数据归一化和划分
    print("\n步骤 2: 数据归一化和划分")
    print("-" * 40)
    
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test, 
     scaler_X, scaler_y) = prepare_datasets_multi_step(X, y, test_size=0.15, val_size=0.15)
    
    # 4. 创建多步预测模型
    print("\n步骤 3: 创建多步LSTM模型")
    print("-" * 40)
    
    model = LSTMModelMultiStep(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # 5. 训练多步预测模型
    print("\n步骤 4: 训练多步预测模型")
    print("-" * 40)
    
    model, train_losses, val_losses, best_epoch = train_model_multi_step(
        model=model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=epochs,
        lr=learning_rate,
        patience=patience
    )
    
    # 6. 评估多步预测模型
    print("\n步骤 5: 评估多步预测模型")
    print("-" * 40)
    
    y_true, y_pred, step_results = evaluate_model_multi_step(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler_y,
        seq_length=seq_length
    )
    
    # 7. 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_length': seq_length,
        'output_size': output_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'evaluation': step_results
    }, f'final_lstm_multi_step_{output_size}.pth')
    
    print(f"\n模型已保存为: final_lstm_multi_step_{output_size}.pth")
    
    # 8. 示例预测（带耗时统计）
    print("\n步骤 7: 示例预测（带耗时统计）")
    print("-" * 40)
    
    # 获取一些测试数据用于演示
    sample_idx = 0
    sample_input_norm = X_test[sample_idx].numpy().flatten()
    
    sample_input_original = scaler_X.inverse_transform(
        sample_input_norm.reshape(-1, 1)
    ).flatten()
    
    print(f"示例输入序列 (最近{seq_length}个时间点):")
    for i in range(seq_length):
        print(f"  t-{seq_length-i}: {sample_input_original[i]:.4f}")
    
    # 进行预测并统计耗时
    print(f"\n开始单次预测...")
    predictions, single_pred_time_ms = make_multi_step_predictions(
        model=model,
        recent_data=sample_input_original,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        seq_length=seq_length
    )
    
    print(f"单次预测耗时: {single_pred_time_ms:.4f} 毫秒")
    
    print(f"\n未来预测结果:")
    for i in range(len(predictions)):
        print(f"  未来第{i+1}个时间点 (t+{i+1}): {predictions[i]:.4f}")
    
    # 获取对应的真实值进行比较
    y_true_original = scaler_y.inverse_transform(y_test[sample_idx].reshape(1, -1))[0]
    print(f"\n实际未来值 (用于比较):")
    for i in range(len(y_true_original)):
        print(f"  实际第{i+1}个时间点 (t+{i+1}): {y_true_original[i]:.4f}")
    
    # 计算预测误差
    errors = np.abs(y_true_original - predictions)
    print(f"\n预测误差:")
    for i in range(len(errors)):
        print(f"  第{i+1}步误差: {errors[i]:.4f}")
    
    # 9. 批量预测耗时统计
    print("\n步骤 8: 批量预测耗时统计")
    print("-" * 40)
    
    batch_stats = batch_prediction_time_stats(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler_y=scaler_y
    )
    
    # 10. 保存耗时统计结果
    print("\n步骤 9: 保存耗时统计结果")
    print("-" * 40)
    
    save_success = save_time_stats_to_file(single_pred_time_ms, batch_stats)
    
    if save_success:
        print("耗时统计结果已成功保存到 prediction_time_stats.txt")
    
    print("\n" + "=" * 60)
    print("所有步骤完成！")
    print("=" * 60)
    
    return model, scaler_X, scaler_y, y_true, y_pred, step_results, single_pred_time_ms, batch_stats

# ==================== 执行主函数 ====================
if __name__ == "__main__":
    # 运行主函数
    model, scaler_X, scaler_y, y_true, y_pred, step_results, single_pred_time_ms, batch_stats = main()
    
    # 性能总结
    print("\n" + "=" * 60)
    print("性能总结")
    print("=" * 60)
    
    for step in step_results.keys():
        results = step_results[step]
        step_num = int(step.split('_')[1])
        print(f"第{step_num}步预测 (t+{step_num}):")
        print(f"  MAE: {results['MAE']:.4f}")
        print(f"  RMSE: {results['RMSE']:.4f}")
        print(f"  MAPE: {results['MAPE']:.2f}%")
        print(f"  精度: {100 - results['MAPE']:.2f}%")
        print()
    
    # 预测精度分析
    output_size = len(step_results)
    mape_values = [step_results[f'step_{i+1}']['MAPE'] for i in range(output_size)]
    
    print(f"预测精度分析:")
    for i in range(output_size):
        print(f"  第{i+1}步精度: {100 - mape_values[i]:.2f}%")
    
    # 计算精度下降趋势
    if output_size > 1:
        accuracy_drops = []
        for i in range(1, output_size):
            drop = mape_values[i] - mape_values[i-1]
            accuracy_drops.append(drop)
        
        print(f"\n精度下降趋势:")
        for i in range(len(accuracy_drops)):
            print(f"  第{i+2}步相对于第{i+1}步精度下降: {accuracy_drops[i]:.2f}个百分点")
        
        avg_drop = np.mean(accuracy_drops)
        print(f"  平均每步精度下降: {avg_drop:.2f}个百分点")
    
    # 耗时性能总结
    print(f"\n预测耗时性能总结:")
    print(f"  单次示例预测耗时: {single_pred_time_ms:.4f} 毫秒")
    print(f"  测试集批量预测总耗时: {batch_stats['total_time_seconds']:.4f} 秒")
    print(f"  测试集样本总数: {batch_stats['num_samples']}")
    print(f"  平均单样本预测耗时: {batch_stats['avg_time_per_sample_ms']:.4f} 毫秒")
    
    # 判断模型是否可用
    avg_mape = np.mean(mape_values)
    if avg_mape < 5.0:
        print(f"\n✅ 模型性能优秀，可用于实际故障预测！")
    elif avg_mape < 10.0:
        print(f"\n✅ 模型性能良好，可用于实际故障预测！")
    elif avg_mape < 15.0:
        print(f"\n⚠️ 模型性能一般，建议进一步优化参数。")
    else:
        print(f"\n❌ 模型性能较差，需要重新设计或更多数据。")
    
    print(f"\n平均MAPE: {avg_mape:.2f}%")
    print(f"整体精度: {100 - avg_mape:.2f}%")