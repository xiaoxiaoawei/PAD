import pandas as pd
import matplotlib.pyplot as plt
import os
# 读取CSV文件
data = pd.read_csv('xxx.csv')
#data = pd.read_csv('xxx.csv')

# 计算直方图的区间数
num_bins = 180  # 可根据实际数据范围进行调整
# 中文设置200，区间1-5
# 英文设置180，区间1-40
# 绘制频率直方图
plt.figure(figsize=(10, 6))
# sentence
#plt.hist(data['human_Perplexity'], bins=num_bins, alpha=0.5, label='humam', color='lightcoral', density=True, range=(1, 5))
#plt.hist(data['chatGPT_Perplexity'], bins=num_bins, alpha=0.5, label='chatGPT', color='lightblue', density=True, range=(1, 5))
# full
plt.hist(data['human'], bins=num_bins, alpha=0.5, label='humam', color='lightcoral', density=True, range=(1, 40))
plt.hist(data['chatGPT'], bins=num_bins, alpha=0.5, label='chatGPT', color='lightblue', density=True, range=(1, 40))

plt.xlabel('Perplexity')
plt.ylabel('Propotion')
#plt.title('Comparison of Perplexity Distribution On All (English)')
plt.title('Comparison of Perplexity Distribution On Full (English)')
plt.legend()
plt.show()


# 指定保存的文件夹路径
save_folder = 'xxx/new_ppl_images'

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 保存为 JPG 格式到指定文件夹，设置高分辨率保存
#plt.savefig(os.path.join(save_folder, 'ALL_CCC_perplexity_comparison.jpg'), format='jpg', dpi=300)
plt.savefig(os.path.join(save_folder, 'ALL_Full_EEE_perplexity_comparison.jpg'), format='jpg', dpi=300)

plt.show()
print('Image saved successfully')
print('======well done!======')
