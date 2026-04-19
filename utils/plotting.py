import matplotlib.pyplot as plt
import os


def plot_loss_curves(train_losses, val_losses, save_dir, filename_base):
    """
    绘制训练和验证损失曲线并保存为 PNG 文件。

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 保存目录
        filename_base: 文件名基础（不含扩展名）
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename_base}_loss.png")

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线已保存: {save_path}")