from transformers import GPT2Tokenizer, GPT2Model
import torch
import pandas as pd
import ast
import re
from tqdm import tqdm
import numpy as np

csv_file = 'datasets/emojify_cleaned_10k_labelled.csv'
data = pd.read_csv(csv_file)

# Load the pre-trained GPT2 model and tokenizer
model_name = "gpt2"  # Can be replaced with other GPT models like "gpt2-medium" or "gpt-neo"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def find_emoji_indices(tokenized_tweet):
    emoji_indices = [index for index, token in enumerate(tokenized_tweet) if is_emoji(token)]
    return emoji_indices

def is_emoji(token):
    emoji_pattern = re.compile(
        "["  # Emoji ranges
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return bool(emoji_pattern.match(token))

# Input text
# text = "ChatGPT provides contextualized embeddings."
# text1 = ['been', 'an', 'emotional', 'wreck', 'all', 'day', '🙄']
# sentence1 = ' '.join(text1)
# text2 = ["don't", 'worry', 'benny', 'was', 'equally', 'unimpressed', 'with', 'how', 'i', 'handled', 'the', 'news', '🙄']
# sentence2 = ' '.join(text2)

# emoji = ' 🙄'
# emoji_token = tokenizer(emoji, return_tensors="pt", padding=True, truncation=True)

# # Tokenize the input text
# inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
# inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)
# print(inputs1)
# print(inputs2)
# print(emoji_token)

# Pass the input through the model
# with torch.no_grad():  # No need for gradients if just extracting embeddings
#     outputs1 = model(**inputs1)
#     outputs2 = model(**inputs2)
#     emoji_outputs = model(**emoji_token)

# # Extract token-level embeddings
# token_embeddings1 = outputs1.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
# token_embeddings2 = outputs2.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
# emoji_embeddings = emoji_outputs.last_hidden_state

# # Convert token IDs back to text tokens for alignment
# tokens1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0])
# tokens2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0])
# emoji_tokens = tokenizer.convert_ids_to_tokens(emoji_token['input_ids'][0])

# breakpoint()

# Example: Print token embeddings and tokens
# for token, embedding in zip(tokens1, token_embeddings1[0]):
#     print(f"Token: {token}, Embedding shape: {embedding.shape}")

# for tk, emb in zip(tokens2, token_embeddings2[0]):
#     print(f"Token: {tk}, Embedding shape: {emb.shape}")

# for tk, emb in zip(emoji_tokens, emoji_embeddings[0]):
#     print(f"Token: {tk}, Embedding shape: {emb.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def get_emoji_embeddings(idx):
    tokenize = data.loc[idx, 'Tokens']
    words_list = ast.literal_eval(tokenize)
    sentence = " ".join(words_list)  # Reconstruct the sentence
    emoji_indices = find_emoji_indices(words_list)  # Find all emoji indices in the tokenized tweet

    if not emoji_indices:  # Skip if no emojis are found
        return None, None
    
    emoji = " " + words_list[emoji_indices[0]]  # Extract the emoji token
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    emoji_inputs = tokenizer(emoji, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        emoji_outputs = model(**emoji_inputs)
    embeddings = outputs.last_hidden_state
    emoji_embedding = emoji_outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    emoji_tokens = tokenizer.convert_ids_to_tokens(emoji_inputs['input_ids'][0])
    sentence_emoji_tokens = [i for i,token in enumerate(tokens) if token in emoji_tokens]
    sentence_embeddings = embeddings[0, sentence_emoji_tokens, :].sum(dim=0)
    emoji_embeddings = emoji_embedding[0, :, :].sum(dim=0)
    if len(sentence_emoji_tokens) != len(emoji_tokens):
        print(f"Embeddings should have the same dimensions, but got {len(sentence_emoji_tokens)} and {len(emoji_tokens)}")
        return None, None
    # assert len(sentence_emoji_tokens) == len(emoji_tokens), "Embeddings should have the same dimensions"
    cosine = cosine_similarity(sentence_embeddings, emoji_embeddings)
    print(f"Cosine similarity between sentence and emoji embeddings: {cosine}")
    # breakpoint()
    
    return sentence_embeddings, emoji_embeddings

# def get_emoji_embeddings(idx):
#     tokenize = data.loc[idx, 'Tokens']
#     words_list = ast.literal_eval(tokenize)
#     sentence = " ".join(words_list)  # Reconstruct the sentence
#     emoji_indices = find_emoji_indices(words_list)  # Find all emoji indices in the tokenized tweet

#     if not emoji_indices:  # Skip if no emojis are found
#         return None, None

#     emoji = " " + words_list[emoji_indices[0]]  # Extract the emoji token
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
#     emoji_inputs = tokenizer(emoji, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         emoji_outputs = model(**emoji_inputs)
#     embeddings = outputs.last_hidden_state
#     emoji_embedding = emoji_outputs.last_hidden_state
    
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#     emoji_tokens = tokenizer.convert_ids_to_tokens(emoji_inputs['input_ids'][0])

#     sentence_emoji_tokens = [i for i,token in enumerate(tokens) if token in emoji_tokens]

#     # emoji_embedding = hidden_states[0, emoji_indices, :].sum(dim=0)  # Shape: [hidden_dim]

#     sentence_embeddings = embeddings[0, sentence_emoji_tokens, :].sum(dim=0)
#     emoji_embeddings = emoji_embedding[0, :, :].sum(dim=0)
#     assert len(sentence_embeddings) == len(emoji_embeddings), "Embeddings should have the same dimensions"
#     cosine = cosine_similarity(sentence_embeddings, emoji_embeddings)
#     print(f"Cosine similarity between sentence and emoji embeddings: {cosine}")
#     breakpoint()
    
#     return sentence_embeddings, emoji_embeddings

embeddings_cache = {}
emoji_embeddings = {}
for idx in tqdm(range(len(data))):
    sentence, emoji = get_emoji_embeddings(idx)
    embeddings_cache[idx] = sentence
    emoji_embeddings[idx] = emoji

#save to pickle 
import pickle

with open('datasets/gpt2_sentence_embeddings_cache.pkl', 'wb') as f:
    pickle.dump(embeddings_cache, f)
with open('datasets/gpt2_emoji_embeddings.pkl', 'wb') as f:
    pickle.dump(emoji_embeddings, f)