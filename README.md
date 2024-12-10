# NLP_Emojiville
Github repository for NLP emoji probing. This readme details the components of the repository and how to ensure reproducible results. For implementation details and discussion, see paper. 

Embeddings: https://drive.google.com/file/d/1mmSihg4zxrabWEDY0W7Cs-JY1tW7OKpl/view?usp=drive_link 

Raw data: https://drive.google.com/file/d/1P-yfPFpuZhVQIzmgYbOmpDPU8urDf6Jy/view?usp=sharing 
# **Emoji Language Embeddings with Fine-Tuned BERT**
## **Requirements**
Install the required dependencies using `pip install -r requirements.txt`.
Dependencies and versions used:
- Python 3
- Transformers library
- PyTorch
- scikit-learn
- pandas
- emoji
- OpenAI
- ProcessPoolExecutor
- json
- ast
- nltk
- TweetTokenizer
- DetectorFactory
- torch
- re
- tiktoken
- pickle
- random
- collections
- Counter
- locale
- torchmetrics
- tensorflow
- numpy
- tqdm

## **Data**
In our Data folder, we have 3 more folders: Cleaning Data, Datasets, and Labeling Data.
- Cleaning Data: this folder holds emojify.ipynb, which details how to clean our dataset. You don't need to run this file because the data is already clean and located in the Datasets folder.
- Datasets: this holds 5 csv files. The first csv file is 10,000 tweets with emojis, cleaned. The second CSV file contains labeled data for those 10k tweets. This is our main dataset used to extract emoji embeddings. The emojify_cleaned_500.csv along with its labeled csv are small samples we used to test pipeline/workflow purposes. We have also provided a link above to the raw dataset. We also manually labeled 30 tweets, documented in manually_labeled_cleaned_30.csv, and prompted GPT to fill in the rest.
- Labeling Data: this folder holds the code notebook used to prompt GPT for the rest of the labels. 

## **Embeddings**
This folder holds a txt file that links to our embeddings in a 154 MB zip file since the pickle embeddings are too large to upload to GitHub. There are 5 pickle files in there, one for each model: GPT, BERT, ELMO, ADA, and CBOW.  You need to download this zip file and locate it in the same repository as this GitHub repo. 

## **Probe**
This folder contains our linear probe code and our data loader code. The linear probe code is called in main.py, a file discussed later on, and our data loader file creates datasets for sentiment and control.  
## **Models**
The models folder holds 5 Python files/notebooks that detail how our embeddings are obtained. You do not need to run them, but they are there for documentation purposes with code comments. Note that BERT does not have emoji embeddings, so they were added to the vocabulary and trained to learn the emojis. 
## **Results**
This houses our notebook for emoji statistics and a folder for our 5 models' predictions. The emoji statistics code groups emojis, counts occurrences and gets percentages for our emoji predictions across all classes. It also grabs the raw statistics from the cleaned dataset, before any models are applied to it. The metrics.py file generates metrics for our Tensorboard such as accuracy, precision, recall, and F1 score. It is called later in another file. Lastly, in our Model Prediction CSVs folder, we house our 5 model predictions. Each CSV file contains 4 columns: a tweet index number, a sentiment score, an emotion, and a part of speech label. 

## **Setup and Installation**
1. Clone this repository.
2. Pip install all the dependencies listed above.
3. Download ZIP file for embeddings, extract, and make sure it is located also in this repository: https://drive.google.com/file/d/1mmSihg4zxrabWEDY0W7Cs-JY1tW7OKpl/view?usp=drive_link
4. Run python main.py 

