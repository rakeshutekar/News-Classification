import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer

from dataset import NewsDataset, load_data, tokenize_data
from train import train_model


def evaluate_model(trainer, test_dataset, uniques):
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = torch.argmax(predictions, dim=1)

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=uniques)

    print('Accuracy:', accuracy)
    print('Classification Report:', report)

if __name__ == "__main__":
    train_texts, test_texts, train_labels, test_labels, uniques = load_data('../data/2024_4.csv')
    train_encodings, test_encodings = tokenize_data(train_texts, test_texts)

    test_dataset = NewsDataset(test_encodings, test_labels)

    # Assuming train_model function returns the trainer instance
    trainer = train_model('../data/2024_4.csv')

    evaluate_model(trainer, test_dataset, uniques)
