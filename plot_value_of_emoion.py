import matplotlib.pyplot as plt
import os

# 数据
labels = ['Neutral', 'Positive', 'Negative',]
human_counts = [0.6113, 0.2097, 0.179]
generated_counts = [0.529, 0.2455, 0.2254]

total_human = sum(human_counts)
total_generated = sum(generated_counts)

# 计算比例
human_percentages = [count / total_human * 100 for count in human_counts]
generated_percentages = [count / total_generated * 100 for count in generated_counts]

# 绘制直方图
barWidth = 0.35
r1 = range(len(labels))
r2 = [x + barWidth for x in r1]

plt.bar(r1, human_percentages, color='lightcoral', width=barWidth, edgecolor='black', label='human', linewidth=0.5)
plt.bar(r2, generated_percentages, color='lightblue', width=barWidth, edgecolor='black', label='chatGPT', linewidth=0.5)

plt.xlabel('Sentiment')
plt.xticks([r + barWidth/2 for r in range(len(labels))], labels)
plt.ylabel('Percentage (%)')
plt.title('Sentiment Distibution Of ALL (English)')
plt.legend()

# 显示每个柱形的百分比
for i in r1:
    plt.text(i, human_percentages[i] + 0.2, f"{human_percentages[i]:.2f}", ha='center', va='bottom')

for i in r1:
    plt.text(i + barWidth, generated_percentages[i] + 0.2, f"{generated_percentages[i]:.2f}", ha='center', va='bottom')

plt.show()

save_folder = '/work/data/hw/project/9.1_linguistic_analysis/images_sentiment'

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 保存为 JPG 格式到指定文件夹，设置高分辨率保存
plt.savefig(os.path.join(save_folder, 'EEE_ALl_sentiment_analysis.jpg'), format='jpg', dpi=500)
