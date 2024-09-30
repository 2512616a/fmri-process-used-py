# 绘制ROI时间序列图
# 加载数据集
# 图谱提取时间序列
# 绘制时间序列图

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
# 输出图谱的所有标签
print("图谱标签:")
for i, label in enumerate(labels):
    print(f"{i+1}. {label}")
print(f"总共有 {len(labels)} 个标签")

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
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False



print("正在绘制时间序列图...")
for i, ts in enumerate(time_series):
    plt.figure(figsize=(20, 10))  # 增加图片宽度
    for roi in range(ts.shape[1]):
        plt.plot(ts[:, roi], label=labels[roi])
    plt.title(f'受试者 {i+1} 的时间序列')
    plt.xlabel('时间点')
    plt.ylabel('信号强度')
    # 设置图例
    plt.legend(
        loc='center right',           # 将图例放置在左侧中央
        bbox_to_anchor=(1, 0.5),     # 将图例锚点设置在图形右侧中央（x=1, y=0.5）
        fontsize=8,                  # 设置图例字体大小为8
        bbox_transform=plt.gcf().transFigure,  # 使用整个图形作为参考进行变换
        fancybox=True,               # 给图例添加圆角边框
        shadow=True,                 # 为图例添加阴影效果
        ncol=1                       # 将图例设置为单列显示
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # 调整右边界，为图例留出更多空间
    plt.savefig(os.path.join(data_dir, f'subject_{i+1}_time_series.png'), dpi=300, bbox_inches='tight')
    plt.show()
print("时间序列图显示完成。")

