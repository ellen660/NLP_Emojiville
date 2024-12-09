import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel,BertForMaskedLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("emojify_cleaned_10k_labelled.csv", encoding='utf-8', engine='python')
dataset.head()
dataset.shape

# Extract tokens and emojis
tweets = dataset['Tokens'].apply(eval)  # Convert string representation of list to actual list
emoji_pattern = re.compile(
    "["  
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002500-\U00002BEF"  # Chinese/Japanese/Korean characters
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "]",
    flags=re.UNICODE,
)

# Extract emojis
def extract_emoji(tokens):
    return [token for token in tokens if emoji_pattern.match(token)]

tweets_with_emojis = [(tokens, extract_emoji(tokens)[0]) for tokens in tweets if extract_emoji(tokens)]

from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add emojis to tokenizer vocabulary
emoji_list = set(emoji for _, emoji in tweets_with_emojis)
new_tokens = [emoji for emoji in emoji_list if emoji not in tokenizer.vocab]
tokenizer.add_tokens(new_tokens)

from transformers import BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load pre-trained BERT model and resize embeddings for new tokens
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

# Create a Dataset class
class EmojiDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, _ = self.data[idx]
        text = ' '.join(tokens)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128,
        )
        # Remove the singleton batch dimension
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs

#sample = emoji_dataset[0]
#print({k: v.shape for k, v in sample.items()})

# Create dataset
emoji_dataset = EmojiDataset(tweets_with_emojis, tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./bert-emoji',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    learning_rate=5e-5,
)

#Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emoji_dataset,
)

# Train model
trainer.train()
from torch.utils.data import DataLoader

dataloader = DataLoader(emoji_dataset, batch_size=8)
batch = next(iter(dataloader))
print({k: v.shape for k, v in batch.items()})

import numpy as np
import torch

# Get the device of the model
device = next(model.parameters()).device

emoji_embeddings = {}
unk_id = tokenizer.convert_tokens_to_ids("[UNK]")

for emoji in emoji_list:
    token_id = tokenizer.convert_tokens_to_ids(emoji)
    if token_id == unk_id:
        print(f"Warning: Emoji {emoji} is not recognized in the tokenizer vocabulary.")
        continue  # Skip unknown tokens
    # Move tensor to the device of the model
    token_tensor = torch.tensor([token_id]).to(device)
    # Get the embedding
    embedding = model.get_input_embeddings()(token_tensor).detach().cpu().numpy()  # Move back to CPU for saving
    emoji_embeddings[emoji] = embedding.squeeze()  # Squeeze to remove singleton dimension

# Save emoji embeddings
import pickle
with open("bert_emoji_embeddings.pkl", "wb") as f:
    pickle.dump(emoji_embeddings, f)

print("Emoji embeddings saved successfully!")

import pickle
# test cases
# Load the emoji embeddings
# with open("bert_emoji_embeddings.pkl", "rb") as f:
#     emoji_embeddings = pickle.load(f)

# # Verify the number of embeddings
# print(f"Number of emojis with embeddings: {len(emoji_embeddings)}")
# emoji = "😊"  # Replace with any emoji you want to inspect
# if emoji in emoji_embeddings:
#     print(f"Embedding for {emoji}: {emoji_embeddings[emoji]}")
# else:
#     print(f"Emoji {emoji} not found in the saved embeddings.")
# embedding_dimensions = {emoji: embedding.shape for emoji, embedding in emoji_embeddings.items()}
# print(embedding_dimensions)