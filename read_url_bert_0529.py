import pandas as pd
import requests
from bs4 import BeautifulSoup

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('FakeNewsNet.csv')
'''
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        
        return article_text
    except requests.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return ""

for index, row in df.iterrows():
    url = row['news_url']
    print(f"Accessing URL: {url}")
    article_text = extract_text_from_url(url)
    
    if article_text:
        print(f"Extracted text from {url}:\n{article_text[:500]}...\n") 
        break
    else:
        print(f"No text extracted from {url}")
        break
'''

#..............................................................................................................................
#..............................................................................................................................

import torch
import torch.nn as nn
import requests
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        
        return article_text
    except requests.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return ""

class BERTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = {}
        for idx, row in df.iterrows():
            self.data[idx] = (row['news_url'], row['real'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url, real = self.data[idx]
        article_text = extract_text_from_url(url)
        return article_text, torch.tensor(real)


class BERT_IMDB(nn.Module):
    '''
    Fine-tuning DistillBert with two MLPs.
    '''

    def __init__(self, pretrained_type):
        super().__init__()

        num_labels = 2
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_type, num_labels=num_labels)
        
        # TO-DO 2-1: Construct a classifier
        # BEGIN YOUR CODE
        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0),
            nn.Linear(512, num_labels),
            # nn.Softmax(),
        ) 

        # END YOUR CODE

    def forward(self, **pretrained_text):
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        logits = self.classifier(pretrained_output)
        
        return logits


class BERT():
    def __init__(self, pretrained_type, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_type)
        self.model = BERT_IMDB(pretrained_type).to(config['device'])
    
    def forward(self, text):    
        outputs = self.model(**text)
        return outputs

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
