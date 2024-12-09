import openai
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
import tiktoken
import ast
import torch.nn.functional as F
import pickle
import random

def emotion_mappings():
    return {'disgust' : 0, 'anger': 1, 'joy' : 2, 'sadness':3, 'confusion': 4, 'surprise':5, 'love':6, 'fear':7, 'calm':6}

def pos_mappings():
    return {'ADP': 0, 'NOUN': 1, 'VERB': 2, 'PROPN': 3, 'NUM': 4, 'PUNCT': 5, 'ADJ': 6, "INTJ": 7, 'PRON': 1}

cache = {
    "ada_002": "Embeddings/10k_ada_embeddings.pkl",
    "gpt2": "Embeddings/gpt2_emoji_embeddings.pkl",
    #"gpt2_sentence": "datasets/gpt2_sentence_embeddings_cache.pkl",
    'elmo': 'Embeddings/emoji_10k_embeddings.pkl',
    "cbow": "Embeddings/cbow_embeddings.pkl",
    "bert": "Embeddings/10k_bert_embeddings2.pkl"
    #'bert': 'datasets/emoji_embeddings (1).pkl'
}

class SentimentDataset(Dataset):
    def __init__(self, csv_file, label, model, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        self.label = label
        self.model = model
        assert self.label in ['sentiment', 'emotion', 'pos'], 'label must be either sentiment, emotion or pos'
        # assert self.model in ['gpt', 'elmo', 'cbow', 'bert'], 'model must be either gpt or elmo'
        self.bad_keys = [16, 219, 303, 310, 405]
        # print(f'processing embeddings for {self.__len__()} samples')
        #load pickle 
        try:
            with open(cache[model], 'rb') as f:
                print(f'loading embeddings from cache')
                self.embeddings_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('No stuff found')
        # breakpoint()
        print('done processing embeddings')
    
    # Function to detect emojis using regex
    def contains_emoji(self, text):
        emoji_pattern = re.compile(
            "["  # Emoji characters
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)
        
        return emoji_pattern.findall(text)
        
    def __len__(self):
        if self.debug == True:
            return 48
        return min(len(self.data), len(self.embeddings_cache))

    def __getitem__(self, idx):

        # breakpoint()

        try:
            emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
        except:
            if self.model == 'cbow' or self.model == 'bert':
                idx = 0
            else:
                idx = idx + 1
            try:
                emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
            except:
                idx = idx + 1
                emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
            # if emoji_embeddings is None:
            #     # print(f'bad key {idx}')
            #     idx = idx + 1
            #     emoji_embeddings = self.embeddings_cache[idx]

        emoji_embeddings = (emoji_embeddings - torch.mean(emoji_embeddings))/torch.std(emoji_embeddings)
    
        if self.label == 'sentiment':
            sentiment = self.data.loc[idx, 'Sentiment_score']
            sentiment_label = torch.tensor(sentiment, dtype=torch.long)
            mapped_label = sentiment_label + 1
            # Create one-hot encoding with 3 classes
            one_hot_label = F.one_hot(mapped_label, num_classes=3)
            #convert to float
            one_hot_label = one_hot_label.type(torch.float32)
            # breakpoint()
        
        elif self.label == 'emotion':
            emotion = self.data.loc[idx, 'Sentiment_emotion']
            #lowercase
            emotion = emotion.lower()
            emotion_label = torch.tensor(emotion_mappings()[emotion], dtype=torch.long)
            one_hot_label = F.one_hot(emotion_label, num_classes=8)
            one_hot_label = one_hot_label.type(torch.float32)
        
        elif self.label == "pos":
            pos = self.data.loc[idx, 'Part_of_speech']
            pos_label = torch.tensor(pos_mappings()[pos], dtype=torch.long)
            one_hot_label = F.one_hot(pos_label, num_classes=8)
            one_hot_label = one_hot_label.type(torch.float32)
            # breakpoint()

        return emoji_embeddings, one_hot_label

class ControlDataset(Dataset):
    def __init__(self, csv_file, label, model, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        self.label = label
        self.model = model
        assert self.label in ['sentiment', 'emotion', 'pos'], 'label must be either sentiment, emotion or pos'
        # assert self.model in ['gpt', 'elmo', 'cbow', 'bert'], 'model must be either gpt or elmo but got {self.model}'
        self.bad_keys = [16, 219, 303, 310, 405]
        # print(f'processing embeddings for {self.__len__()} samples')
        #load pickle 
        with open(cache[model], 'rb') as f:
            print(f'loading embeddings from cache')
            self.embeddings_cache = pickle.load(f)

    def __len__(self):
        if self.debug == True:
            return 48
        return min(len(self.data), len(self.embeddings_cache))
    
    def __getitem__(self, idx):
        # if idx in self.bad_keys:
        #     #choose another index
        #     print(f'bad key {idx}')
        #     idx = idx + 1

        try:
            emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
        except:
            if self.model == 'cbow' or self.model == 'bert':
                idx = 0
            else:
                idx = idx + 1
            try:
                emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
            except:
                idx = idx + 1
                emoji_embeddings = torch.tensor(self.embeddings_cache[idx], dtype=torch.float)
        # emoji_embeddings = self.embeddings_cache[idx]
        # breakpoint()

        # if emoji_embeddings is None:
        #     # print(f'bad key {idx}')
        #     idx = idx + 1
        #     emoji_embeddings = self.embeddings_cache[idx]
        #     if emoji_embeddings is None:
        #         # print(f'bad key {idx}')
        #         idx = idx + 1
        #         emoji_embeddings = self.embeddings_cache[idx]

        emoji_embeddings = (emoji_embeddings - torch.mean(emoji_embeddings))/torch.std(emoji_embeddings)

        if self.label == 'sentiment':
            #return random number between 0 and 2
            sentiment_label = torch.tensor(random.randint(0, 2), dtype=torch.long)
            one_hot_label = F.one_hot(sentiment_label, num_classes=3)
            one_hot_label = one_hot_label.type(torch.float32)
        
        elif self.label == 'emotion':
            #return random number between 0 and 7
            emotion_label = torch.tensor(random.randint(0, 7), dtype=torch.long)
            one_hot_label = F.one_hot(emotion_label, num_classes=8)
            one_hot_label = one_hot_label.type(torch.float32)
            # breakpoint()
        
        elif self.label == "pos":
            #return random number between 0 and 6
            pos_label = torch.tensor(random.randint(0, 7), dtype=torch.long)
            one_hot_label = F.one_hot(pos_label, num_classes=8)
            one_hot_label = one_hot_label.type(torch.float32)
    
        return emoji_embeddings, one_hot_label
    
if __name__ == "__main__":
    # Usage example
    csv_file = 'datasets/emojify_cleaned_10k_labelled.csv'
    batch_size = 48
    label = 'emotion'
    dataset = SentimentDataset(csv_file, label, 'cbow', debug=False)
    # dataset = ControlDataset(csv_file, label, debug=False)
    print(f'length of dataset: {dataset.__len__()}')

    feature, label = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # # Loop through the dataloader
    # labels_count = {0: 0, 1: 0, 2: 0}
    labels_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0} #[10, 20, 3, 10, 0, 10, 5, 20]
    #pos {0: 5, 1: 140, 2: 142, 3: 4, 4: 1, 5: 12, 6: 196, 7: 0} -> [10, 5, 5, 10, 10, 10, 5]
    for i in range(len(dataset)):
        feature, label = dataset.__getitem__(i)
        labels_count[label.argmax().item()] += 1
    breakpoint()
