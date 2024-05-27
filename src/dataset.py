import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    # Load the dataset
    news_data = pd.read_csv(file_path)

    # Filter necessary columns and drop any rows with NaN values
    news_data = news_data[['Title', 'Category']].dropna().reset_index(drop=True)

    # Encode labels
    labels, uniques = pd.factorize(news_data['Category'])
    news_data['label'] = labels

    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(news_data['Title'], news_data['label'], test_size=0.2, random_state=42)

    # Ensure the texts are in the correct format and reset indices
    train_texts = train_texts.reset_index(drop=True).tolist()
    test_texts = test_texts.reset_index(drop=True).tolist()
    train_labels = train_labels.reset_index(drop=True).tolist()
    test_labels = test_labels.reset_index(drop=True).tolist()

    return train_texts, test_texts, train_labels, test_labels, uniques

def tokenize_data(train_texts, test_texts, max_length=512):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    return train_encodings, test_encodings
