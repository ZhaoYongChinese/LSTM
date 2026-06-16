import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 🎯 核心修改：统一从本地的 metrics.py 中导入所有评价指标
from .metrics import compute_mape, compute_mse, compute_mae, compute_r2

def train_model(model, train_data, val_data, epochs, lr, patience, device,
                batch_size=64,
                loss_type='mse', grad_clip=1.0, weight_decay=0):
    """
    训练模型（Mini-batch），支持早停、自适应学习率衰减、梯度裁剪、AdamW正则化。
    v3: AdamW + ReduceLROnPlateau 替代 Adam + StepLR
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 选择损失函数
    if loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss()

    # AdamW: 解耦权重衰减，防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ReduceLROnPlateau: 验证 loss 不降时自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2,
        min_lr=1e-7
    )

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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

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

        avg_train_loss = epoch_train_loss / len(train_dataset)

        # ---------- 验证阶段 ----------
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                pred_val = model(batch_X_val)
                val_loss = criterion(pred_val, batch_y_val)
                epoch_val_loss += val_loss.item() * batch_X_val.size(0)

        avg_val_loss = epoch_val_loss / len(val_dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 自适应学习率: 验证 loss 停滞时自动降 lr
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ---------- 早停逻辑 ----------
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
                  f"LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"早停触发于 epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss, train_losses, val_losses


def evaluate_model(model, X_test, y_test, scaler_y, device):
    """
    评估模型，返回整体 R2、MAPE、MSE、MAE、预测值和真实值。
    🎯 修复 OOM 隐患：加入 device 参数。
    """
    model.eval()
    with torch.no_grad():
        # 🎯 将测试集输入数据移动到 GPU 上进行推理预测
        X_test = X_test.to(device)
        pred_norm = model(X_test)
        
        # 统一转回 cpu numpy 进行反归一化和评价指标计算
        # 兼容统一scaler (1特征) 和旧版scaler (47特征)
        pred_np = pred_norm.cpu().numpy()
        true_np = y_test.cpu().numpy()
        if hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ == 1:
            pred = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
            true = scaler_y.inverse_transform(true_np.reshape(-1, 1)).reshape(true_np.shape)
        else:
            pred = scaler_y.inverse_transform(pred_np)
            true = scaler_y.inverse_transform(true_np)

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