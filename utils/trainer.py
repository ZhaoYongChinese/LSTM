import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 🎯 核心修改：统一从本地的 metrics.py 中导入所有评价指标
from .metrics import compute_mape, compute_mse, compute_mae, compute_r2

def train_model(model, train_data, val_data, epochs, lr, patience,
                batch_size=64,
                loss_type='mse', grad_clip=1.0, step_size=20, gamma=0.9):
    """
    训练模型（已加入 Mini-batch 批处理），支持早停、学习率衰减、梯度裁剪。
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # 构建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 选择损失函数
    if loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    use_teacher_forcing = hasattr(model, 'teacher_forcing_ratio') and model.teacher_forcing_ratio > 0

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        epoch_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            if use_teacher_forcing:
                pred_train = model(batch_X, target=batch_y)
            else:
                pred_train = model(batch_X)

            loss = criterion(pred_train, batch_y)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0) 
            
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_dataset)

        # ---------- 验证阶段 ----------
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                pred_val = model(batch_X_val)
                val_loss = criterion(pred_val, batch_y_val)
                epoch_val_loss += val_loss.item() * batch_X_val.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ---------- 记录与早停逻辑 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"早停触发于 epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, best_val_loss, train_losses, val_losses

def evaluate_model(model, X_test, y_test, scaler_y):
    """
    评估模型，返回整体 R2、MAPE、MSE、MAE、预测值和真实值。
    """
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test)
        pred = scaler_y.inverse_transform(pred_norm.cpu().numpy())
        true = scaler_y.inverse_transform(y_test.cpu().numpy())

    # 🎯 调用本地 metrics.py 的 compute_r2
    overall_r2 = compute_r2(true, pred)
    overall_mape = compute_mape(true, pred)
    overall_mse = compute_mse(true, pred)
    overall_mae = compute_mae(true, pred)
    
    return overall_mape, overall_mse, overall_mae, overall_r2, pred, true

def save_model(model, scaler_X, scaler_y, params, overall_r2, save_dir, filename):
    """
    保存模型、归一化器、参数和评价指标。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'params': params,
        'overall_r2': overall_r2 
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")
    return save_path