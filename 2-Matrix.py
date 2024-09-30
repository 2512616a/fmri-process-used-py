import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.getcwd()
func_conn_file = os.path.join(current_dir, 'func_conn_matrices.npy')

if os.path.exists(func_conn_file):
    func_conn_matrices = np.load(func_conn_file)
else:
    print(f"错误：在当前工作目录 '{current_dir}' 中未找到 'func_conn_matrices.npy' 文件。")
    func_conn_matrices = None

# 假设func_conn_matrices是一个3D数组，每个2D矩阵代表一个受试者的功能连接矩阵
# 我们可以选择第一个受试者的矩阵来可视化

# 选择第一个受试者的矩阵
subject_matrix = func_conn_matrices[0]

# 创建热图
plt.figure(figsize=(12, 10))
plt.imshow(subject_matrix, cmap='viridis', aspect='auto')
cbar = plt.colorbar(label='功能连接强度', fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('功能连接强度', fontsize=12)

plt.title('功能连接矩阵', fontsize=16)
plt.xlabel('ROI', fontsize=12)
plt.ylabel('ROI', fontsize=12)

# 添加网格线
plt.grid(which='major', color='w', linestyle='-', linewidth=0.5)

# 调整刻度标签
plt.tick_params(axis='both', which='major', labelsize=10)

# 显示图像
plt.tight_layout()
plt.show()

# 如果你想保存图像，可以使用以下代码
# plt.savefig('functional_connectivity_matrix.png', dpi=300, bbox_inches='tight')

# 关闭图像
plt.close()

print("功能连接矩阵已生成并显示。")


