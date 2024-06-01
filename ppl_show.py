import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取包含两列数据的 CSV 文件
data = pd.read_csv('xxx_ppl_new.csv')

# 获取第一列数据
data_col1 = data.iloc[:, 0].dropna()

# 获取第二列数据
data_col2 = data.iloc[:, 1].dropna()

# 将数据列转换为数值类型
data_col1 = pd.to_numeric(data_col1, errors='coerce')
data_col2 = pd.to_numeric(data_col2, errors='coerce')

# 删除含有缺失值的行
data_col1 = data_col1.dropna()
data_col2 = data_col2.dropna()
print(data_col1)
print(data_col2)

# 绘制第一列数据的概率分布曲线
plt.figure(figsize=(10, 6))
data_col1_sorted = data_col1.sort_values()
data_col1_sorted.plot(kind='kde', color='lightcoral', label='human', bw_method=0.8)


# 填充第一列数据曲线下方的区域
#plt.fill_between(data_col1_sorted.index, data_col1_sorted, color='blue', alpha=0.8)

# 绘制第二列数据的概率分布曲线
data_col2_sorted = data_col2.sort_values()
data_col2_sorted.plot(kind='kde', color='blue', label='chatgpt', bw_method=0.8)
# 填充第二列数据曲线下方的区域（限定在0到15之间）
#plt.fill_between(data_col2_sorted.index, data_col2_sorted, color='lightblue', alpha=0.2)

# 设置纵轴范围为0到0.4
plt.ylim(0, 1.2)
# 设置横轴范围为0到15
plt.xlim(0, 15)
plt.xlabel('Value of perplexity')
plt.ylabel('Density')
plt.legend()
 
# 指定保存的文件夹路径
save_folder = 'xxx/images_perplexity'

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 保存为 JPG 格式到指定文件夹，设置高分辨率保存
plt.savefig(os.path.join(save_folder, 'arti_new_distribution_ppl.jpg'), format='jpg', dpi=800)

plt.show()
