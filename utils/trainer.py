import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .metrics import compute_mape
from .metrics import compute_mape, compute_mse, compute_mae


def train_model(model, train_data, val_data, epochs, lr, patience,
                loss_type='mse', grad_clip=1.0, step_size=20, gamma=0.9):
    """
    训练模型，支持早停、学习率衰减、梯度裁剪。

    参数:
        model: PyTorch模型
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        epochs: 最大训练轮数
        lr: 初始学习率
        patience: 早停耐心值
        loss_type: 'mse' 或 'huber'
        grad_clip: 梯度裁剪阈值
        step_size: 学习率衰减步长
        gamma: 衰减因子

    返回:
        model: 训练好的模型（已加载最佳权重）
        best_val_loss: 最佳验证损失
        train_losses: 每个epoch的训练损失列表
        val_losses: 每个epoch的验证损失列表
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # 选择损失函数
    if loss_type.lower() == 'huber':
        criterion = nn.HuberLoss()
        print("使用损失函数: Huber Loss")
    else:
        criterion = nn.MSELoss()
        print("使用损失函数: MSE Loss")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 检查是否为 Seq2Seq 模型，以便启用 Teacher Forcing
    use_teacher_forcing = hasattr(model, 'teacher_forcing_ratio') and model.teacher_forcing_ratio > 0
    if use_teacher_forcing:
        print(f"检测到 Seq2Seq 模型，Teacher Forcing 比例 = {model.teacher_forcing_ratio}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        optimizer.zero_grad()

        if use_teacher_forcing:
            pred_train = model(X_train, target=y_train)
        else:
            pred_train = model(X_train)

        loss = criterion(pred_train, y_train)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = criterion(pred_val, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # 打印训练信息
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {loss.item():.6f} | "
                  f"Val Loss: {val_loss.item():.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 早停判断
        if patience_counter >= patience:
            print(f"早停触发于 epoch {epoch+1}")
            break

    # 加载最佳模型权重
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已加载最佳模型，验证损失: {best_val_loss:.6f}")

    return model, best_val_loss.item(), train_losses, val_losses


def evaluate_model(model, X_test, y_test, scaler_y):
    """
    评估模型，返回整体 MAPE、MSE、MAE、预测值和真实值（已反归一化）
    """
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test)
        pred = scaler_y.inverse_transform(pred_norm.cpu().numpy())
        true = scaler_y.inverse_transform(y_test.cpu().numpy())

    overall_mape = compute_mape(true, pred)
    overall_mse = compute_mse(true, pred)
    overall_mae = compute_mae(true, pred)
    return overall_mape, overall_mse, overall_mae, pred, true


def save_model(model, scaler_X, scaler_y, params, overall_mape, save_dir, filename):
    """
    保存模型、归一化器、参数和评估指标到指定目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'params': params,
        'overall_mape': overall_mape
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")
    return save_path