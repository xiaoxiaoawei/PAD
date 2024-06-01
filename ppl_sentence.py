import csv
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import numpy as np

# 设置环境变量 CUDA_LAUNCH_BLOCKING
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 指定本地模型路径
model_path = 'xxx/Wenzhong-GPT2-110M'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载本地语言模型
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 分割文本函数
def split_text(text):
    halfway_index = len(text) // 2
    first_half = text[:halfway_index].rsplit('.', 1)[0] + '.'
    second_half = text[halfway_index:].split('.', 1)[-1]
    return first_half, second_half

# 处理单条数据的函数
def process_data(row):
    text = row[0]  # 选择第一列文本

    # 检查文本的 token 数
    token_count = len(tokenizer.encode(text))

    if token_count > 1024:
        perplexity = np.nan
    else:
        if token_count > 2048:
            first_half, second_half = split_text(text)
            first_third, second_third = split_text(second_half)

            input_ids_1 = tokenizer.encode(first_half, return_tensors='pt').to(device)
            input_ids_2 = tokenizer.encode(second_half, return_tensors='pt').to(device)
            input_ids_3 = tokenizer.encode(second_third, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs_1 = model(input_ids_1, labels=input_ids_1)
                loss_1 = outputs_1.loss
                perplexity_1 = 2 ** loss_1.item()

                outputs_2 = model(input_ids_2, labels=input_ids_2)
                loss_2 = outputs_2.loss
                perplexity_2 = 2 ** loss_2.item()

                outputs_3 = model(input_ids_3, labels=input_ids_3)
                loss_3 = outputs_3.loss
                perplexity_3 = 2 ** loss_3.item()

            perplexity = (perplexity_1 + perplexity_2 + perplexity_3) / 3

        elif token_count > 1024:
            first_half, second_half = split_text(text)

            input_ids_1 = tokenizer.encode(first_half, return_tensors='pt').to(device)
            input_ids_2 = tokenizer.encode(second_half, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs_1 = model(input_ids_1, labels=input_ids_1)
                loss_1 = outputs_1.loss
                perplexity_1 = 2 ** loss_1.item()

                outputs_2 = model(input_ids_2, labels=input_ids_2)
                loss_2 = outputs_2.loss
                perplexity_2 = 2 ** loss_2.item()

            perplexity = (perplexity_1 + perplexity_2) / 2

        else:
            input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = 2 ** loss.item()

    writer.writerow(row + [perplexity if not np.isnan(perplexity) else 'NAN'])

# 打开原始 CSV 文件和目标 CSV 文件
with open('xxx.csv', 'r') as input_file, open('xxx_ppl_sentence.csv', 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    header = next(reader)
    writer.writerow(header + ['Perplexity'])

    total_rows = sum(1 for row in reader)
    input_file.seek(0)
    next(reader)

    for row in tqdm(reader, total=total_rows, desc="Processing", bar_format="{l_bar}{bar}"):
        process_data(row)

print("======================已保存成功===================。")
