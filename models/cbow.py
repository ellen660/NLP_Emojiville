import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
import tiktoken
import ast
import torch.nn.functional as F
import pickle

from gensim.models import keyedvectors

# Define a custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, csv_file, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        print(f'processing embeddings for {self.__len__()} samples')

        # change for emoji2vec
        self.embeddings_cache = {idx: self.get_embeddings(idx) for idx in range(self.__len__())}
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

    #  Function to get embeddings from OpenAI
    def get_emoji2vec_embeddings(self, text):
        """Fetch embeddings for a list of texts from Emoji2Vec's output"""
        e2v = keyedvectors.load_word2vec_format('emoji2vec.bin', binary=True)
        if text in e2v: return e2v[text]    # Produces an embedding vector of length 300
        return None
        
    def __len__(self):
        if self.debug == True:
            return 48
        return len(self.data)
    
    def get_embeddings(self, idx):
        """
        cache the emoji embeddings to avoid repeated calls to OpenAI and is faster
        used in the __getitem__ method
        """
        # idx is a row number

        # Get the text and label for the current index
        print("IDX-----------")
        print(self.data.loc[idx])
        #sentence = self.data.loc[idx, 'cleaned_text']
        tokenize = self.data.loc[idx, 'Tokens']
        words_list = ast.literal_eval(tokenize)
        emojis = [token for token in words_list if self.contains_emoji(token)]

        # get the embeddings
        emoji_embeddings = None
        for emoji in emojis:
            embed = self.get_emoji2vec_embeddings(emoji)
            if embed is None: continue
            if emoji_embeddings is None:
                emoji_embeddings = torch.tensor(embed, dtype=torch.float)
            else: 
                emoji_embeddings = emoji_embeddings + torch.tensor(embed, dtype=torch.float)
        return emoji_embeddings

    def __getitem__(self, idx):
        emoji_embeddings = self.embeddings_cache[idx]

        sentiment = self.data.loc[idx, 'Sentiment_score']
        sentiment_label = torch.tensor(sentiment, dtype=torch.long)
        mapped_label = sentiment_label + 1
        # Create one-hot encoding with 3 classes
        one_hot_label = F.one_hot(mapped_label, num_classes=3)
        #convert to float
        one_hot_label = one_hot_label.type(torch.float32)
        # breakpoint()
        
        return emoji_embeddings, one_hot_label
    

if __name__ == "__main__":

    # Usage example
    csv_file = 'emojify_cleaned_500_labelled.csv'
    batch_size = 48
    dataset = SentimentDataset(csv_file, debug=False)

    with open('cbow_embeddings.pkl', 'wb') as f:
        pickle.dump(dataset.embeddings_cache, f)

    feature, label = dataset.__getitem__(0)
    print(f'label: {label}')
    dataset.__getitem__(1)
    dataset.__getitem__(2)
    dataset.__getitem__(3)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # # # Loop through the dataloader
    # for features, labels in dataloader:
    #     print("Features (Emoji embeddings):", features.shape)
    #     print("Labels (Sentiment):", labels)
    #     break