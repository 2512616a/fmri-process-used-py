import os
from nilearn import datasets
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.input_data import NiftiLabelsMasker
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

# 设置数据目录
data_dir = './abide_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 检查数据是否已存在
if os.path.exists(os.path.join(data_dir, 'ABIDE_pcp')):
    print("ABIDE数据集已存在，直接加载本地数据。")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, n_subjects=5, pipeline='cpac', 
                                     band_pass_filtering=True, global_signal_regression=False,
                                     verbose=0)
else:
    print("正在下载ABIDE数据集...")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, n_subjects=5, pipeline='cpac', 
                                     band_pass_filtering=True, global_signal_regression=False)

# 获取预处理后的fMRI数据文件路径
func_filenames = abide.func_preproc
print("获取预处理后的fMRI数据文件路径完成。")

# 使用Harvard-Oxford图谱
print("正在加载Harvard-Oxford图谱...")
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
print("Harvard-Oxford图谱加载完成。")

# 获取正确的标签
labels = [region for region in atlas.labels if region != '']
print("获取图谱标签完成。")

# 使用Harvard-Oxford图谱提取时间序列
print("正在使用Harvard-Oxford图谱提取时间序列...")
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

time_series = []
for i, func_file in enumerate(func_filenames):
    print(f"正在处理第 {i+1}/{len(func_filenames)} 个功能文件...")
    ts = masker.fit_transform(func_file)
    time_series.append(ts)
print("时间序列提取完成。")

# 直接显示时间序列图
print("正在绘制时间序列图...")
for i, ts in enumerate(time_series):
    plt.figure(figsize=(10, 6))
    for roi in range(ts.shape[1]):
        plt.plot(ts[:, roi], label=labels[roi])
    plt.title(f'受试者 {i+1} 的时间序列')
    plt.xlabel('时间点')
    plt.ylabel('信号强度')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()
print("时间序列图显示完成。")

# 创建Mean Time Series文件夹
mean_ts_dir = os.path.join(data_dir, 'Mean Time Series')
if not os.path.exists(mean_ts_dir):
    os.makedirs(mean_ts_dir)
    print(f"创建Mean Time Series目录: {mean_ts_dir}")

# 生成四位受试者的ROI Mean Time Series
print("正在生成ROI Mean Time Series图...")
for subject_index in range(4):  # 只处理前四位受试者
    print(f"正在处理受试者 {subject_index + 1}...")
    ts = time_series[subject_index]
    
    print(f"正在计算受试者 {subject_index + 1} 的每个ROI的平均时间序列...")
    # 计算每个ROI的平均时间序列
    mean_ts = np.mean(ts, axis=0)
    
    print(f"正在为受试者 {subject_index + 1} 创建图表...")
    # 创建一个美观的图表
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    print(f"正在为受试者 {subject_index + 1} 绘制每个ROI的平均时间序列...")
    # 绘制每个ROI的平均时间序列
    for i, label in enumerate(labels[:10]):  # 只显示前10个ROI以保持图表清晰
        plt.plot(mean_ts[i], label=label.split(' ')[0], linewidth=2)
    
    print(f"正在为受试者 {subject_index + 1} 设置图表标题和标签...")
    plt.title(f"受试者 {subject_index + 1} 的ROI平均时间序列", fontsize=16)
    plt.xlabel("时间点", fontsize=12)
    plt.ylabel("BOLD信号强度", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    print(f"正在保存受试者 {subject_index + 1} 的ROI平均时间序列图...")
    # 保存图表
    plt.savefig(os.path.join(mean_ts_dir, f'subject_{subject_index + 1}_mean_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已为受试者 {subject_index + 1} 生成ROI平均时间序列图。")

print("所有ROI平均时间序列图生成完毕。")

# 计算功能连接矩阵
print("正在计算功能连接矩阵...")
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrices = correlation_measure.fit_transform(time_series)
print("功能连接矩阵计算完成。")

# 保存功能连接矩阵
np.save(os.path.join(data_dir, 'func_conn_matrices.npy'), correlation_matrices)
print("功能连接矩阵已保存为 func_conn_matrices.npy")

print("标签数量:", len(labels))
print("矩阵形状:", correlation_matrices[0].shape)

# 如果需要，调整标签数量
labels = labels[:correlation_matrices[0].shape[0]]
print("标签数量调整完成。")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']        
plt.rcParams['axes.unicode_minus'] = False

# 打印受试者信息
print("正在处理受试者信息...")
phenotypic = abide.phenotypic
""" print("表型数据的类型:", type(phenotypic))
print("表型数据的字段:", phenotypic.dtype.names) """

# 获取受试者ID和诊断组信息
subject_id_field = 'SUB_ID' if 'SUB_ID' in phenotypic.dtype.names else 'subject_id'
dx_group_field = 'DX_GROUP' if 'DX_GROUP' in phenotypic.dtype.names else 'dx_group'

for i in range(len(phenotypic)):
    subject_id = phenotypic[subject_id_field][i]
    dx_group = phenotypic[dx_group_field][i]
    status = "患者" if dx_group == 1 else "健康对照"
    print(f"受试者 {i+1} (ID: {subject_id}): {status}")

# 创建Functional correlation matrix文件夹
functional_corr_dir = os.path.join(data_dir, 'Functional correlation matrix')
if not os.path.exists(functional_corr_dir):
    os.makedirs(functional_corr_dir)
    print(f"创建功能连接矩阵目录: {functional_corr_dir}")

# 为所有受试者生成功能连接矩阵图
print("正在为所有受试者生成功能连接矩阵图...")
for subject_index in range(len(correlation_matrices)):
    print(f"正在处理受试者 {subject_index + 1}...")
    correlation_matrix = correlation_matrices[subject_index]
    
    # 绘制功能连接矩阵图
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='相关系数')
    plt.title(f"受试者 {subject_index + 1} 的功能连接矩阵")
    plt.tight_layout()
    
    # 保存功能连接矩阵图
    plt.savefig(os.path.join(functional_corr_dir, f'subject_{subject_index + 1}_functional_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已为受试者 {subject_index + 1} 生成功能连接矩阵图。")

print("所有受试者的功能连接矩阵图生成完毕。")
# 构建第一个受试者的Functional Graph
print("正在构建第一个受试者的功能图...")
subject_index = 0
correlation_matrix = correlation_matrices[subject_index]

# 创建一个无向图
G = nx.Graph()

# 添加节点
for i, label in enumerate(labels):
    G.add_node(i, label=label)

# 添加边
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        weight = correlation_matrix[i, j]
        if weight > 0:  # 只添加正相关的边
            G.add_edge(i, j, weight=weight)

print("功能图构建完成。")

# 绘制Functional Graph
print("正在绘制功能图...")
plt.figure(figsize=(16, 14))
pos = nx.spring_layout(G)

# 设置边的颜色映射
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
edge_cmap = plt.cm.get_cmap('coolwarm')
edge_colors = [edge_cmap(w) for w in edge_weights]

# 绘制边，使用颜色表示连接强度
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')

# 添加节点标签（简化的ROI名称）
simplified_labels = [label.split(' ')[0] for label in labels]  # 只取第一个单词作为简化标签
nx.draw_networkx_labels(G, pos, {i: label for i, label in enumerate(simplified_labels)}, font_size=8)

plt.title(f"受试者 {subject_index + 1} 的功能连接图")
plt.axis('off')
plt.tight_layout()

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('连接强度')

plt.savefig(os.path.join(data_dir, f'subject_{subject_index + 1}_functional_graph.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"已为受试者 {subject_index + 1} 生成功能连接图。")

# 打印一些图的基本信息
print(f"节点数量: {G.number_of_nodes()}")
print(f"边的数量: {G.number_of_edges()}")
print(f"平均聚类系数: {nx.average_clustering(G)}")
print(f"平均最短路径长度: {nx.average_shortest_path_length(G)}")

# 生成Core Functional Network
print("正在生成核心功能网络...")
plt.figure(figsize=(16, 14))
pos = nx.spring_layout(G)

# 使用和谐的颜色方案
edge_colors = plt.cm.viridis(np.linspace(0, 1, G.number_of_edges()))

# 绘制边，使用和谐的颜色
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1, alpha=0.6)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', alpha=0.8)

# 不添加节点标签和边的权重信息

plt.title(f"受试者 {subject_index + 1} 的核心功能网络")
plt.axis('off')
plt.tight_layout()

plt.savefig(os.path.join(data_dir, f'subject_{subject_index + 1}_core_functional_network.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"已为受试者 {subject_index + 1} 生成核心功能网络图。")

# 创建Binarized Network Graph文件夹
binarized_dir = os.path.join(data_dir, 'Binarized Network Graph')
if not os.path.exists(binarized_dir):
    os.makedirs(binarized_dir)

# 对第一位患者进行二值化处理
subject_index = next(i for i, dx in enumerate(phenotypic[dx_group_field]) if dx == 1)
correlation_matrix = correlation_matrices[subject_index]

# 创建二值化图的函数
def create_binarized_graph(matrix, threshold, threshold_type='absolute'):
    G = nx.Graph()
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if threshold_type == 'absolute':
                if abs(matrix[i, j]) > threshold:
                    G.add_edge(i, j)
            elif threshold_type == 'proportional':
                if abs(matrix[i, j]) > threshold:
                    G.add_edge(i, j)
    return G

# 绘制二值化图的函数
def plot_binarized_graph(G, title):
    plt.figure(figsize=(16, 14))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='lightblue', node_size=300, with_labels=False)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

# 绘制并保存二值化图
thresholds = [0.2, 0.3]
for threshold in thresholds:
    G_bin = create_binarized_graph(correlation_matrix, threshold)
    plot_binarized_graph(G_bin, f"受试者 {subject_index + 1} 的二值化网络图 (阈值 = {threshold})")
    plt.savefig(os.path.join(binarized_dir, f'subject_{subject_index + 1}_binarized_graph_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已为受试者 {subject_index + 1} 生成二值化网络图 (阈值 = {threshold})。")

# 计算比例阈值
def calculate_proportional_threshold(matrix, percentage):
    flattened = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
    return np.percentile(flattened, 100 - percentage)

percentages = [15, 10]
for percentage in percentages:
    threshold = calculate_proportional_threshold(correlation_matrix, percentage)
    G_bin = create_binarized_graph(correlation_matrix, threshold, 'proportional')
    plot_binarized_graph(G_bin, f"受试者 {subject_index + 1} 的二值化网络图 (最强 {percentage}% 连接)")
    plt.savefig(os.path.join(binarized_dir, f'subject_{subject_index + 1}_binarized_graph_top_{percentage}percent.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已为受试者 {subject_index + 1} 生成二值化网络图 (最强 {percentage}% 连接)。")

print("所有二值化网络图已生成完毕。")

# 生成功能连接矩阵的二值化图
def plot_binarized_matrix(matrix, threshold, threshold_type, title):
    plt.figure(figsize=(12, 10))
    if threshold_type == 'absolute':
        binarized_matrix = np.abs(matrix) > threshold
    elif threshold_type == 'proportional':
        threshold_value = calculate_proportional_threshold(matrix, threshold)
        binarized_matrix = np.abs(matrix) > threshold_value
    
    plt.imshow(binarized_matrix, cmap='binary', interpolation='nearest')
    plt.colorbar(label='连接')
    plt.title(title)
    plt.tight_layout()

# 生成并保存二值化功能连接矩阵图
thresholds = [0.2, 0.3]
for threshold in thresholds:
    plot_binarized_matrix(correlation_matrix, threshold, 'absolute', f"受试者 {subject_index + 1} 的二值化功能连接矩阵 (阈值 = {threshold})")
    plt.savefig(os.path.join(binarized_dir, f'subject_{subject_index + 1}_binarized_matrix_threshold_{threshold}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已为受试者 {subject_index + 1} 生成二值化功能连接矩阵图 (阈值 = {threshold})。")

percentages = [15, 10]
for percentage in percentages:
    plot_binarized_matrix(correlation_matrix, percentage, 'proportional', f"受试者 {subject_index + 1} 的二值化功能连接矩阵 (最强 {percentage}% 连接)")
    plt.savefig(os.path.join(binarized_dir, f'subject_{subject_index + 1}_binarized_matrix_top_{percentage}percent.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已为受试者 {subject_index + 1} 生成二值化功能连接矩阵图 (最强 {percentage}% 连接)。")

print("所有二值化功能连接矩阵图已生成完毕。")
