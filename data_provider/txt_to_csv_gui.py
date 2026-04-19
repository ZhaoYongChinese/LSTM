import re
import csv
from pathlib import Path
from tkinter import Tk, filedialog

def select_file_and_convert():
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    file_path = filedialog.askopenfilename(
        title="选择要转换的 .txt 文件",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not file_path:
        print("未选择文件，退出。")
        return

    input_file = Path(file_path)
    output_file = input_file.with_suffix('.csv')

    number_pattern = re.compile(r'^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$')
    values = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if number_pattern.match(line):
                try:
                    values.append(str(float(line)))
                except ValueError:
                    continue

    if not values:
        print("未找到有效数字行！")
        return

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RMS_Value'])
        for v in values:
            writer.writerow([v])

    print(f"转换成功！\n输入: {input_file}\n输出: {output_file}\n数据行数: {len(values)}")

if __name__ == "__main__":
    select_file_and_convert()