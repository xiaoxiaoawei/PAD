import pandas as pd
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from scipy.special import softmax
from tqdm import tqdm 
import os
import torch

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

MODEL = "xxx/itter-xlm-roberta-base-sentiment"

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)

# PT
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
model.to(device)
model.eval()

# Read the CSV file
data = pd.read_csv('xxx/E_elec.csv')

# Create an empty list to store emotion scores
emotion_scores = []

# Iterate through each row, process the text, and calculate sentiment scores
for index, row in tqdm(data.iterrows(), total=len(data)):
    human_text = row['human']
    chatGPT_text = row['generate']
    
    # Data preprocessing
    human_text = preprocess(human_text)
    chatGPT_text = preprocess(chatGPT_text)
    
    # Calculate sentiment scores
    scores = []
    texts = [human_text, chatGPT_text]
    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True,max_length=512)
        encoded_input.to(device)
        output = model(**encoded_input)
        logits = output.logits.detach().cpu().numpy()
        scores.append(softmax(logits))
        #print(scores)
    
    # Append sentiment scores to the list
    emotion_scores.append({'human_negative': scores[0][0][0], 'human_neutral': scores[0][0][1],'human_positive': scores[0][0][2],
    'chatGPT_negative': scores[1][0][0],'chatGPT_neutral': scores[1][0][1],'chatGPT_positive': scores[1][0][2]})

    #print(emotion_scores)
    #print(f"Human: {scores[0][0][0]}, ChatGPT: {scores[1][0][0]}")

# Convert the list of sentiment scores to a DataFrame
emotion_scores_df = pd.DataFrame(emotion_scores)

# Save sentiment scores to a CSV file
output_dir = 'xxx/EEE_emotion_score'
emotion_scores_df.to_csv(os.path.join(output_dir, 'EEE_elec_emotion_scores.csv'), index=False)

print("Emotion scores saved to emotion_scores.csv")
print("work done!")
