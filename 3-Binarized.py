import os
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载功能连接矩阵
data_dir = './abide_data'
correlation_matrices = np.load(os.path.join(data_dir, 'func_conn_matrices.npy'))

# 选择第一个受试者的相关矩阵
correlation_matrix = correlation_matrices[0]

# 定义二值化函数
def binarize_matrix(matrix, threshold, threshold_type='absolute'):
    if threshold_type == 'absolute':
        return np.abs(matrix) > threshold
    elif threshold_type == 'proportional':
        flattened = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
        threshold_value = np.percentile(flattened, 100 - threshold)
        return np.abs(matrix) > threshold_value

# 生成并显示二值化功能连接矩阵图
def plot_binarized_matrix(matrix, threshold, threshold_type):
    binarized_matrix = binarize_matrix(matrix, threshold, threshold_type)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(binarized_matrix, cmap='binary', interpolation='nearest')
    plt.colorbar(label='连接')
    
    if threshold_type == 'absolute':
        plt.title(f"二值化功能连接矩阵 (阈值 = {threshold})")
    else:
        plt.title(f"二值化功能连接矩阵 (最强 {threshold}% 连接)")
    
    plt.tight_layout()
    plt.show()

# 生成并显示二值化功能连接矩阵图
plot_binarized_matrix(correlation_matrix, 0.3, 'absolute')
plot_binarized_matrix(correlation_matrix, 15, 'proportional')
