import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def parse_txt_to_csv(input_dir):
    """
    解析指定文件夹下的所有 txt 文件，提取数据并转存为 csv
    """
    # 获取选中的文件夹名称 (例如 "wind")
    # normpath 用于去掉路径末尾可能存在的斜杠，确保 basename 提取正确
    folder_name = os.path.basename(os.path.normpath(input_dir))
    
    # 构建输出路径：当前运行目录下的 data/folder_name
    output_base = os.path.join(os.getcwd(), "data")
    output_dir = os.path.join(output_base, folder_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"📁 目标数据源: {input_dir}")
    print(f"📂 CSV输出至: {output_dir}")
    print("=" * 60)

    # 获取所有 .txt 文件
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    
    if not txt_files:
        print(f"[警告] 文件夹中没有找到任何 .txt 文件！")
        return

    success_count = 0
    for filename in txt_files:
        txt_path = os.path.join(input_dir, filename)
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        all_values = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式匹配所有在 [ ] 内部的内容，re.DOTALL 允许跨行匹配
            list_strings = re.findall(r'\[(.*?)\]', content, re.DOTALL)
            
            for s in list_strings:
                # 按照逗号分割，清洗出有效数字
                parts = s.split(',')
                for p in parts:
                    clean_str = p.strip()
                    if clean_str:
                        try:
                            # 尝试转换为浮点数并追加
                            all_values.append(float(clean_str))
                        except ValueError:
                            # 跳过无法转换为浮点数的脏数据
                            pass
                            
            if all_values:
                # 转换为 DataFrame 并保存
                df = pd.DataFrame({'RMS_Value': all_values})
                df.to_csv(csv_path, index=False)
                print(f"✅ 成功转换: {filename:<15} -> {csv_filename:<15} (提取数据: {len(all_values)} 行)")
                success_count += 1
            else:
                print(f"⚠️ 跳过文件: {filename} (未提取到有效数字)")
                
        except Exception as e:
            print(f"❌ 解析 {filename} 时发生错误: {e}")

    print("=" * 60)
    print(f"🎉 批量转换完成！共处理 {success_count} 个文件。")

def main():
    print("[系统提示] 正在启动文件选择器，请在弹出的窗口中选择您的 TXT 文件夹...")
    
    # 初始化 tkinter 隐藏主窗口并置顶弹窗
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # 弹出文件夹选择对话框
    selected_dir = filedialog.askdirectory(title="请选择包含 TXT 文件的文件夹")
    root.destroy()
    
    if not selected_dir:
        print("[系统提示] 您取消了文件夹选择，程序终止。")
        return
        
    parse_txt_to_csv(selected_dir)

if __name__ == "__main__":
    main()