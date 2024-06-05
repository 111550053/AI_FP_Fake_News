import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from bert import BERT, BERTDataset
from preprocess import preprocessing_function

import numpy as np

import time

warnings.filterwarnings("ignore")


def prepare_data():
    # do not modify
    df_test = pd.read_csv('./data/news_articles.csv', nrows = 2094)
    return df_test


def second_part(model_type, df_test, N):
    #configure
    bert_config = {
        'batch_size': 8,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    # load and dataset and set model
    path = 'project_2.pt'
    model = torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def collate_fn(data):
        sequences, labels = zip(*data)
        sequences, labels = list(sequences), list(labels)
        sequences = model.tokenizer(sequences, padding=True, truncation=True,max_length=512, return_tensors="pt")
        return sequences,torch.tensor(labels)

    test_data = BERTDataset(df_test)
    test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)
   
    test(
        model_type=model_type,
        model=model,
        test_dataloader=test_dataloader,
        optimizer=torch.optim.Adam(model.parameters(), lr=model.config['lr']),
        loss_fn=nn.CrossEntropyLoss().to(model.config['device']),
        config=model.config
    )

def test(model_type, model, test_dataloader, optimizer, loss_fn, config):
    # testing stage
    model.eval()
    pred = []
    labels = []
    for X, y in tqdm(test_dataloader):
        pre = model.forward(X.to(config['device']))
        # print(pre)
        # print(y)
        # time.sleep(2)
        pred.append(pre.argmax(1).cpu())
        labels.append(y)

    precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    print(f"fake amout: {np.sum(pred)}")
    print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")
    # END YOUR CODE
    
if  __name__ == '__main__':
    # get argument
    model_type = 'BERT'
    N = 2 # we only use bi-gram in this assignment, but you can try different N


    # read and prepare data
    df_test = prepare_data()

    label_mapping = {'Real': 0, "Fake": 1}
    df_test['label'] = df_test['label'].map(label_mapping)
    # feel free to add more text preprocessing method
    df_test['text'] = df_test['text'].apply(preprocessing_function)
    
    second_part(model_type, df_test, N)