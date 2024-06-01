import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from tqdm import tqdm
import json
import os

# 准备数据集
class PatentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# 读取 CSV 文件
df = pd.read_csv('xxx/ALL_CCC_patent_data_full.csv')
data = df.to_dict(orient='records')

# 分割训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

# 加载分词器和模型
model_path = "xlmRoberta"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)

# 构建数据集和数据加载器
train_dataset = PatentDataset(train_data, tokenizer, max_length=512)
val_dataset = PatentDataset(val_data, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(10):
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', colour='blue'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# 评估模型
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Validation', colour='blue'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
#auroc = roc_auc_score(true_labels, predicted_probabilities)

print(f"Accuracy on validation set: {accuracy}")
print(f"Recall on validation set: {recall}")
print(f"F1 Score on validation set: {f1}")
#print(f"AUROC on validation set: {auroc}")

# 保存实验结果
results = {
    'Accuracy': accuracy,
    'Recall': recall,
    'F1': f1
    #"AUROC": auroc
}

json_file_path = "xxx/evaluation_results.json"
if not os.path.exists(json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file)
        print(f"New JSON file created: {json_file_path}")
else:
    print("JSON file already exists. Skipping creation.")

print(f"Evaluation results saved to {json_file_path}.")
print("+++++++++Training complete!+++++++++++++")


# 保存模型
model_save_path = "xxx/5_ALL_CCC_full_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
