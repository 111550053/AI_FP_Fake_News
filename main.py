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

warnings.filterwarnings("ignore")


def prepare_data():
    # do not modify
    df_train = pd.read_csv('./data/IMDB_train.csv')
    df_test = pd.read_csv('./data/IMDB_test.csv')
    return df_train, df_test

def get_argument():
    # do not modify
    config = {
        'model_type': "BERT",
        'part': 2,
        'preprocess': 1,
    }
    return config

def first_part(model_type, df_train, df_test, N):
    # load and train model
    if model_type == 'ngram':
        model = Ngram(N)
        model.train(df_train)
    else:
        raise NotImplementedError

    # test performance of model
    perplexity = model.compute_perplexity(df_test)
    print("Perplexity of {}: {}".format(model_type, perplexity))

    return model

def second_part(model_type, df_train, df_test, N):

    # training configure
    bert_config = {
        'batch_size': 8,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }
    

    
    # load and dataset and set model
    model = BERT('distilbert-base-uncased', config = bert_config)

    def collate_fn(data):
        sequences, labels = zip(*data)
        sequences, labels = list(sequences), list(labels)
        sequences = model.tokenizer(sequences, padding=True, truncation=True,max_length=512, return_tensors="pt")
        return sequences,torch.tensor(labels)

    train_data = BERTDataset(df_train)
    test_data = BERTDataset(df_test)
    train_dataloader = DataLoader(train_data, batch_size = bert_config['batch_size'], collate_fn = collate_fn)
    test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)
   
    train(
        model_type=model_type,
        model=model,
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader,
        optimizer=torch.optim.Adam(model.parameters(), lr=model.config['lr']),
        loss_fn=nn.CrossEntropyLoss().to(model.config['device']),
        config=model.config
    )

def train(model_type, model, train_dataloader, test_dataloader, optimizer, loss_fn, config):
    '''
    total_loss: the accumulated loss of training
    labels: the correct labels set for the test set
    pred: the predicted labels set for the test set
    '''
    # TO-DO 2-2: Implement the training function
    # BEGIN YOUR CODE
    # pf = 0
    for epoch in range(config['epochs']): 
        model.train()
        # training stage
        total_loss = 0
        # pr = []
        # la = []
        for X, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            pred = model.forward(X.to(config['device']))
            loss = loss_fn(pred, y.to(config['device']))
            total_loss += loss.item()
            # pr.append(pred.argmax(1).cpu())
            # la.append(y)
            loss.backward()
            optimizer.step()

        # precision, recall, f1, support = precision_recall_fscore_support(la, pr, average='macro', zero_division=1)
        # precision = round(precision, 4)
        # recall = round(recall, 4)
        # f1 = round(f1, 4)
        # avg_loss = round(total_loss/len(train_dataloader), 4)
        # print(f"Epoch: {epoch}, F1 score: {f1}, Precision: {precision}, Recall: {recall}, Loss: {avg_loss}")

        # testing stage
        model.eval()
        pred = []
        labels = []
        with torch.no_grad():
            for X, y in tqdm(test_dataloader):
                pre = model.forward(X.to(config['device']))
                pred.append(pre.argmax(1).cpu())
                labels.append(y)

        precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        avg_loss = round(total_loss/len(train_dataloader), 4)
        print(f"Epoch: {epoch}, F1 score: {f1}, Precision: {precision}, Recall: {recall}, Loss: {avg_loss}")
        # if pf-f1 > 0.01 :
        #     break
        # pf = f1
    
    # END YOUR CODE
    
if  __name__ == '__main__':
    # get argument
    config = get_argument()
    model_type, preprocessed = config['model_type'], config['preprocess']
    N = 2 # we only use bi-gram in this assignment, but you can try different N

    # read and prepare data
    df_train, df_test = prepare_data()
    label_mapping = {'negative': 0, 'positive': 1}
    df_train['sentiment'] = df_train['sentiment'].map(label_mapping)
    df_test['sentiment'] = df_test['sentiment'].map(label_mapping)

    # feel free to add more text preprocessing method
    if preprocessed:
        df_train['review'] = df_train['review'].apply(preprocessing_function)
        df_test['review'] = df_test['review'].apply(preprocessing_function)

    if config['part'] == 1:
        # part 1: implement bi-gram model
        first_part(model_type, df_train, df_test, N)
    elif config['part'] == 2:
        # part 2: implement your nn model from scratch
        # part 3: implement your nn model from other pre-trained model
        second_part(model_type, df_train, df_test, N)