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

#  Function to get embeddings from OpenAI
def get_chatgpt_embeddings(texts):
    """Fetch embeddings for a list of texts from ChatGPT's API."""
    api_key = "put your API key here"
    string = " ".join(texts)
    response = openai.embeddings.create(
            model= "text-embedding-ada-002",
            input=string
        )
    # breakpoint()
    return response.data[0].embedding # Change this

def emotion_mappings():
    return {'disgust' : 0, 'anger': 1, 'joy' : 2, 'sadness':3, 'confusion': 4, 'surprise':5, 'love':6, 'fear':7, 'calm':6}

def pos_mappings():
    return {'ADP': 0, 'NOUN': 1, 'VERB': 2, 'PROPN': 3, 'NUM': 4, 'PUNCT': 5, 'ADJ': 6, "INTJ": 7}

# Define a custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, csv_file, label, model, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        self.label = label
        self.model = model
        assert self.label in ['sentiment', 'emotion', 'pos'], 'label must be either sentiment, emotion or pos'
        assert self.model in ['gpt', 'elmo'], 'model must be either gpt or elmo'
        self.bad_keys = [16, 219, 303, 310, 405]
        print(f'processing embeddings for {self.__len__()} samples')
        #load pickle 
        try:
            with open('datasets/10k_gpt_embeddings.pkl', 'rb') as f:
                print(f'loading embeddings from cache')
                self.embeddings_cache = pickle.load(f)
            with open('datasets/10k_gpt_all_embeddings.pkl', 'rb') as f:
                print(f'loading all embeddings from cache')
                self.all_embeddings = pickle.load(f)
            with open('datasets/10k_elmo_embeddings.pkl', 'rb') as f:
                print(f'loading elmo embeddings from cache')
                self.elmo_embeddings = pickle.load(f)
                # breakpoint()
                # self.elmo_embeddings = {}
                # for dictionary in elmo_embeddings:
                #     emoji_idx = list(dictionary['emoji_embeddings'].keys())[0]
                #     self.elmo_embeddings[dictionary['index']] = dictionary['emoji_embeddings'][emoji_idx]
        except FileNotFoundError:
            self.embeddings_cache = {}
            self.all_embeddings = {}
            for idx in range(self.__len__()):
                emoji_embeddings, all_embeddings = self.get_embeddings(idx)
                self.embeddings_cache[idx] = emoji_embeddings
                self.all_embeddings[idx] = all_embeddings
                # self.embeddings_cache = {idx: self.get_embeddings(idx) for idx in range(self.__len__())} #16 doesn't work for some reason
                
                # save this to a pickle file
                # breakpoint()
            if not self.debug:
                with open('datasets/gpt_embeddings.pkl', 'wb') as f:
                    pickle.dump(self.embeddings_cache, f)
                with open('datasets/gpt_all_embeddings.pkl', 'wb') as f:
                    pickle.dump(self.all_embeddings, f)

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
        return min(len(self.data), 9000)
    
    def get_embeddings(self, idx):
        """
        cache the emoji embeddings to avoid repeated calls to OpenAI and is faster
        used in the __getitem__ method
        """
        # Get the text and label for the current index
        # sentence = self.data.loc[idx, 'cleaned_text']
        tokenize = self.data.loc[idx, 'Tokens']
        words_list = ast.literal_eval(tokenize)
        emojis = [token for token in words_list if self.contains_emoji(token)]
        # breakpoint()
        
        emoji_embeddings = None
        all_embeddings = None
        
        emoji_embeddings = torch.tensor(get_chatgpt_embeddings(emojis), dtype=torch.float)
        all_embeddings = torch.tensor(get_chatgpt_embeddings(words_list), dtype=torch.float)
        return emoji_embeddings, all_embeddings

    def __getitem__(self, idx):
        # if idx in self.bad_keys:
        #     #choose another index
        #     print(f'bad key {idx}')
        #     idx = idx + 1

        emoji_embeddings = self.embeddings_cache[idx]
        all_embeddings = self.all_embeddings[idx]
        elmo_embeddings = self.elmo_embeddings[idx]
        # breakpoint()

        if emoji_embeddings is None or all_embeddings is None or elmo_embeddings is None:
            # print(f'bad key {idx}')
            idx = idx + 1
            emoji_embeddings = self.embeddings_cache[idx]
            all_embeddings = self.all_embeddings[idx]
            elmo_embeddings = self.elmo_embeddings[idx]
            if emoji_embeddings is None or all_embeddings is None or elmo_embeddings is None:
                # print(f'bad key {idx}')
                idx = idx + 1
                emoji_embeddings = self.embeddings_cache[idx]
                all_embeddings = self.all_embeddings[idx]
                elmo_embeddings = self.elmo_embeddings[idx]

        emoji_embeddings = (emoji_embeddings - torch.mean(emoji_embeddings))/torch.std(emoji_embeddings)
        all_embeddings = (all_embeddings - torch.mean(all_embeddings))/torch.std(all_embeddings)
        elmo_embeddings = (elmo_embeddings - torch.mean(elmo_embeddings))/torch.std(elmo_embeddings)

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

        if self.model == 'gpt':
            return emoji_embeddings, one_hot_label
            # return emoji_embeddings, one_hot_label
            # return all_embeddings, one_hot_label
        elif self.model == 'elmo':
            return elmo_embeddings, one_hot_label

class ControlDataset(Dataset):
    def __init__(self, csv_file, label, model, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        self.label = label
        self.model = model
        assert self.label in ['sentiment', 'emotion', 'pos'], 'label must be either sentiment, emotion or pos'
        assert self.model in ['gpt', 'elmo'], 'model must be either gpt or elmo but got {self.model}'
        self.bad_keys = [16, 219, 303, 310, 405]
        print(f'processing embeddings for {self.__len__()} samples')
        #load pickle 
        with open('datasets/10k_gpt_embeddings.pkl', 'rb') as f:
            print(f'loading embeddings from cache')
            self.embeddings_cache = pickle.load(f)
        with open('datasets/10k_gpt_all_embeddings.pkl', 'rb') as f:
            print(f'loading all embeddings from cache')
            self.all_embeddings = pickle.load(f)
        with open('datasets/10k_elmo_embeddings.pkl', 'rb') as f:
            print(f'loading elmo embeddings from cache')
            self.elmo_embeddings = pickle.load(f)
            # self.elmo_embeddings = {}
            # for dictionary in elmo_embeddings:
            #     emoji_idx = list(dictionary['emoji_embeddings'].keys())[0]
            #     self.elmo_embeddings[dictionary['index']] = dictionary['emoji_embeddings'][emoji_idx]

    def __len__(self):
        if self.debug == True:
            return 48
        return min(len(self.data), 9000)
    
    def __getitem__(self, idx):
        # if idx in self.bad_keys:
        #     #choose another index
        #     print(f'bad key {idx}')
        #     idx = idx + 1

        emoji_embeddings = self.embeddings_cache[idx]
        all_embeddings = self.all_embeddings[idx]
        elmo_embeddings = self.elmo_embeddings[idx]
        # breakpoint()

        if emoji_embeddings is None or all_embeddings is None or elmo_embeddings is None:
            # print(f'bad key {idx}')
            idx = idx + 1
            emoji_embeddings = self.embeddings_cache[idx]
            all_embeddings = self.all_embeddings[idx]
            elmo_embeddings = self.elmo_embeddings[idx]
            if emoji_embeddings is None or all_embeddings is None or elmo_embeddings is None:
                # print(f'bad key {idx}')
                idx = idx + 1
                emoji_embeddings = self.embeddings_cache[idx]
                all_embeddings = self.all_embeddings[idx]
                elmo_embeddings = self.elmo_embeddings[idx]

        emoji_embeddings = (emoji_embeddings - torch.mean(emoji_embeddings))/torch.std(emoji_embeddings)
        all_embeddings = (all_embeddings - torch.mean(all_embeddings))/torch.std(all_embeddings)
        elmo_embeddings = (elmo_embeddings - torch.mean(elmo_embeddings))/torch.std(elmo_embeddings)

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
    
        if self.model == 'gpt':
            return emoji_embeddings, one_hot_label
            # return emoji_embeddings, one_hot_label
            # return all_embeddings, one_hot_label
        elif self.model == 'elmo':
            return elmo_embeddings, one_hot_label
    
if __name__ == "__main__":

    # words_list = ['agreed', '😏']
    # print(get_chatgpt_embeddings(words_list))

    # Usage example
    # csv_file = 'cleaned_backhand_index_pointing_right_with_sentiment.csv'
    csv_file = 'datasets/emojify_cleaned_10k_labelled.csv'
    batch_size = 48
    label = 'pos'
    dataset = SentimentDataset(csv_file, label, 'elmo', debug=False)
    # dataset = ControlDataset(csv_file, label, debug=False)
    print(f'length of dataset: {dataset.__len__()}')

    feature, label = dataset.__getitem__(0)
    # dataset.__getitem__(1)
    # dataset.__getitem__(2)
    # dataset.__getitem__(3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # # Loop through the dataloader
    # labels_count = {0: 0, 1: 0, 2: 0}
    labels_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0} #[10, 20, 3, 10, 0, 10, 5, 20]
    #pos {0: 5, 1: 140, 2: 142, 3: 4, 4: 1, 5: 12, 6: 196, 7: 0} -> [10, 5, 5, 10, 10, 10, 5]
    for i in range(len(dataset)):
        feature, label = dataset.__getitem__(i)
        labels_count[label.argmax().item()] += 1
        # if feature is None:
        #     breakpoint()
        # print("Features (Emoji embeddings):", feature.shape)
        # print("Labels:", label)
        # labels_set.add(label)
    breakpoint()
    # for features, labels in dataloader:
        # print("Features (Emoji embeddings):", features.shape)
        # print("Labels:", labels)
    #     for label in labels:
    #         labels_set.add(label)
    # breakpoint()
