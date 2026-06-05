"""
初步观察数据，在时间轴上显示数据
"""
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def select_file():
    """
    通过 GUI 弹窗选择 CSV 文件
    """
    # 初始化 tkinter
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，限制只能选择 csv 文件
    file_path = filedialog.askopenfilename(
        title="选择振动数据 CSV 文件",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    return file_path

def plot_temporal_overview(file_path, points_per_day=47):
    """
    绘制时域概览图
    """
    # 1. 读取数据
    print(f"正在读取文件: {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # 确保使用正确的列（处理可能的列名空格）
        col_name = df.columns[0]
        if 'RMS_Value' in df.columns:
            col_name = 'RMS_Value'
            
        y_data = df[col_name]
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"成功读取数据，共 {len(y_data)} 个点。")

    # 2. 初始化画布
    plt.figure(figsize=(15, 6)) # 宽幅画布以展示长序列
    
    # 3. 绘制主振动曲线
    plt.plot(y_data.index, y_data.values, color='#1f77b4', linewidth=1.2, label='RMS Value')

    # 4. 绘制每天的垂直分割线 (每 47 个点)
    max_index = len(y_data)
    for i in range(0, max_index, points_per_day):
        # 避免在图例中重复生成分割线的标签
        label = 'Day Boundary' if i == 0 else ""
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.6, linewidth=1, label=label)

    # 5. 图表美化与设置
    plt.title('Temporal Raw Overview of Vibration Data', fontsize=14, pad=15)
    plt.xlabel('Data Points Index', fontsize=12)
    plt.ylabel('RMS Value', fontsize=12)
    
    # X轴刻度：确保刻度线跟分割线对齐（可选，如果数据极长可以注释掉这行）
    # plt.xticks(range(0, max_index, points_per_day)) 
    
    plt.grid(axis='y', linestyle=':', alpha=0.7) # 仅显示Y轴横向网格，保持画面整洁
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 6. 显示图像
    print("正在生成图表...")
    plt.show()

if __name__ == "__main__":
    # 步骤 1: 让用户通过 GUI 选择文件
    target_file = select_file()
    
    # 步骤 2: 判断用户是否选择了文件
    if target_file and os.path.exists(target_file):
        # 步骤 3: 绘制图表
        plot_temporal_overview(target_file)
    else:
        print("未选择文件或文件不存在，程序退出。")