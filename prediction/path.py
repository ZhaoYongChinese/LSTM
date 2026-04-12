"""
预测相关路径配置
"""

# ==================== 输入输出路径配置 ====================

# 输入数据路径（离线批量预测的.csv文件）
BATCH_PREDICTION_INPUT_PATH = r'C:\Users\asus\Desktop\hehe\直梯\数据\1\output\rms_values.csv'

# 输出文件夹路径
BATCH_PREDICTION_OUTPUT_DIR = r'C:\Users\asus\Desktop\hehe\直梯\LSTM\prediction\result'

# 输出文件名（自动添加时间戳）
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_PREDICTION_STATS_FILE = f'{BATCH_PREDICTION_OUTPUT_DIR}\\预测误差统计_{timestamp}.txt'
BATCH_PREDICTION_VISUALIZATION_PATH = f'{BATCH_PREDICTION_OUTPUT_DIR}\\预测误差图表_{timestamp}.png'
BATCH_PREDICTION_DATA_FILE = f'{BATCH_PREDICTION_OUTPUT_DIR}\\预测完整数据_{timestamp}.csv'

# 模型文件路径（默认使用first文件夹中最好的模型）
# 可以手动修改为其他模型路径
PREDICTION_MODEL_PATH = r'C:\Users\asus\Desktop\hehe\直梯\LSTM\output\first\temp_lstm_h128_l2_e500_lr0.005_p30_s12_o5_d0.3.pth'


# ==================== 路径配置标志 ====================

# 是否自动创建不存在的文件夹
AUTO_CREATE_DIRS = True

# 如果输出文件已存在是否覆盖（True=覆盖，False=询问）
OVERWRITE_EXISTING = False