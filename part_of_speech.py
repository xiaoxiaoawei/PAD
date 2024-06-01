import os
import spacy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载Spacy的中文模型
nlp = spacy.load("zh_core_web_sm")

# 从CSV文件中读取数据
df = pd.read_csv('xxx.csv')

# 获取标签列和文本列
labels = ['human', 'chatgpt']  # 标签
texts = df.iloc[:, :]  # 假设有两列文本数据与两类标签对应
#print(texts)
# 初始化词性统计字典
pos_counts = {}
total_words = {}

# 计算每个类别中每个词性出现的概率
for label, text_row in zip(labels, texts.iterrows()):
    text1 = text_row[1][0]  # 获取第一列文本数据
    text2 = text_row[1][1]  # 获取第二列文本数据
    
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    if label not in total_words:
        total_words[label] = 0
    total_words[label] += len(doc1) + len(doc2)
    
    if label not in pos_counts:
        pos_counts[label] = {}
    
    for token in doc1:
        pos = token.pos_
        if pos in pos_counts[label]:
            pos_counts[label][pos] += 1
        else:
            pos_counts[label][pos] = 1
    
    for token in doc2:
        pos = token.pos_
        if pos in pos_counts[label]:
            pos_counts[label][pos] += 1
        else:
            pos_counts[label][pos] = 1

# 计算每个类别中每个词性的平均比例
pos_probs = {label: {pos: count / total_words[label] * 100 for pos, count in pos_counts[label].items()} for label in pos_counts}

# 获取所有出现的词性
all_pos = set([pos for label_pos in pos_counts.values() for pos in label_pos])

# 按照词性概率值由高到低排序
sorted_pos = sorted(all_pos, key=lambda pos: sum(pos_probs[label].get(pos, 0) for label in pos_probs), reverse=True)

# 确定颜色列表
colors = {'human': 'lightcoral', 'chatgpt': 'lightblue'}

# 绘制条形图展示词性平均比例
plt.figure(figsize=(12, 6))

bar_width = 0.35
index = np.arange(len(sorted_pos))

for i, label in enumerate(pos_probs):
    plt.bar(index + i * bar_width, [pos_probs[label].get(pos, 0) for pos in sorted_pos], bar_width, color=colors[label], edgecolor='black', linewidth=0.5, label=label)

plt.xlabel('Part of Speech')
plt.ylabel('Probability (%)')
plt.title('Average Part of Speech Distribution (Mech)')
plt.xticks(index + bar_width / 2, sorted_pos, rotation=45)
plt.legend()

# 显示每个条形图的概率值
for i, pos in enumerate(sorted_pos):
    for j, label in enumerate(pos_probs):
        plt.text(index[i] + j * bar_width, pos_probs[label].get(pos, 0) + 0.5, f"{pos_probs[label].get(pos, 0):.2f}", ha='center', va='bottom', color='black', fontsize=7)

plt.tight_layout()

# 将绘制的图片保存到当前目录
save_folder = 'xxx/image_pos'

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 保存为 JPG 格式到指定文件夹，设置高分辨率保存
plt.savefig(os.path.join(save_folder, 'mech_pos.jpg'), format='jpg', dpi=800)

plt.show()
