from Probe.dataloader import SentimentDataset, ControlDataset
from Probe.linear_probe import LinearProbe
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Results.metrics import MulticlassMetrics
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import pandas as pd

#TODO add args parse
def emotion_mappings():
    return {'disgust' : 0, 'anger': 1, 'joy' : 2, 'sadness':3, 'confusion': 4, 'surprise':5, 'love':6, 'fear':7, 'calm':6}

def pos_mappings():
    return {'ADP': 0, 'NOUN': 1, 'VERB': 2, 'PROPN': 3, 'NUM': 4, 'PUNCT': 5, 'ADJ': 6, "INTJ": 7, 'PRON': 1}

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device, {device}")
#format time in data, hour, min
cur_time = time.strftime("%Y%m%d-%H%M")
mapping_500 = {'sentiment': [3, torch.tensor([5, 10, 2])],
            'emotion': [8, torch.tensor([10, 20, 3, 10, 0, 10, 5, 20])],
            'pos': [7, torch.tensor([10, 5, 5, 10, 10, 10, 5])]
           }
mapping_10k = {'sentiment': [3, torch.tensor([5, 8, 2])],
            'emotion': [8, torch.tensor([10, 20, 5, 8, 20, 8, 8, 15])],
            'pos': [8, torch.tensor([20, 5, 5, 20, 20, 20, 3, 20])]
           }
mapping_model = {
    'ada_002': 1536,
    'gpt2': 768,
    'gpt2_sentence': 768,
    'elmo': 1024,
    'cbow': 300,
    'bert': 768
}
model_name = 'gpt2'
batch_size = 48
csv_file = 'datasets/emojify_cleaned_10k_labelled.csv'
num_epochs = 75
learning_rate = 1e-5
weight_decay = 0.001

torch.manual_seed(42) #making it random but reproducible

#Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        # if key == 'confusion':
        #     fig = plot_confusion_matrix(metrics[key])
        #     writer.add_figure(f"Confusion Matrix {phase}", fig, epoch_index)
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
            writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            #average 
            for i in range(value.shape[0]):
                writer.add_scalar("%s/%s_%d"%(phase, key, i), value[i], epoch_index)
    writer.flush()

embedding_dim = mapping_model[model_name]
overall = {}

for label in ['sentiment', 'emotion', 'pos']:
    num_classes, weight = mapping_10k[label]

    for i in range(1):

        if i == 0:
            dataset = SentimentDataset(csv_file, label, model_name, debug=False)
            log_dir = f"csv/{model_name}/{label}/{cur_time}"
            weight = weight/weight.sum()
            weight = weight.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            dataset = ControlDataset(csv_file, label, model_name, debug=False)
            log_dir = f"csv/{model_name}/{label}/control_{cur_time}"
            loss_fn = nn.CrossEntropyLoss()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        # breakpoint()

        # Create DataLoaders for each subset
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Evaluation
        metrics = MulticlassMetrics(device=device, num_classes=num_classes)
        writer = SummaryWriter(log_dir = log_dir)

        #script to train the model
        model = LinearProbe(embedding_dim, num_classes)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
            #Validation
            model.eval()
            with torch.no_grad():
                epoch_loss = 0
                for i, (x, y) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch")):
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)

                    # breakpoint()
                    loss = loss_fn(y_pred, y)
                    epoch_loss += loss.item()
                    metrics.fill_metrics(y_pred, y)
                print(f"Epoch {epoch}, validation loss: {epoch_loss}")
                computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
                logger(writer, computed_metrics, 'val', epoch)
                metrics.clear_metrics()
            #Training
            model.train()
            epoch_loss = 0
            for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch")):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                metrics.fill_metrics(y_pred, y)
            print(f"Epoch {epoch}, training loss: {epoch_loss}")
            computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
            logger(writer, computed_metrics, 'train', epoch)
            metrics.clear_metrics()

        # Save the final model at the end of all epochs
        model_save_path = f"model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Final model saved at {model_save_path}")
        writer.close()
        
        model.eval()    
        # for i, (x,y) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch")):
        #     x, y = x.to(device), y.to(device)
        #     y_pred = model(x)
        #     breakpoint()
        if label == 'sentiment':
            col_name = 'Sentiment_score'
            mapping = {0: -1, 1: 0, 2: 1}
        elif label == 'emotion':
            col_name = 'Sentiment_emotion'
            mapping = {0: 'disgust', 1: 'anger', 2: 'joy', 3: 'sadness', 4: 'confusion', 5: 'surprise', 6: 'love', 7: 'fear'}
        elif label == 'pos':
            col_name = 'Part_of_speech'
            mapping = {0: 'ADP', 1: 'NOUN', 2: 'VERB', 3: 'PROPN', 4: 'NUM', 5: 'PUNCT', 6: 'ADJ', 7: 'INTJ'}   

        for idx in range(len(dataset)):
            x, y = dataset.__getitem__(idx)
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            #take softmax
            y_pred = torch.softmax(y_pred, dim=0)
            y_hat = torch.argmax(y_pred)
            if idx in overall:
                overall[idx][col_name] = mapping[y_hat.item()]
            else:
                overall[idx] = {col_name: mapping[y_hat.item()]}
        # breakpoint()
        
#save overall to log_dir
# breakpoint()
import pickle
with open(f"{log_dir}/overall.pkl", "wb") as f:
    pickle.dump(overall, f)

df = pd.DataFrame.from_dict(overall, orient="index")

# Save the DataFrame to a file in the log directory
df.to_csv(f"{log_dir}/overall.csv", index_label="idx")

print(f"DataFrame saved to {log_dir}/overall.csv")
# #load it back in
# with open(f"{log_dir}/overall.pkl", "rb") as f:
#     overall = pickle.load(f)

# breakpoint()

