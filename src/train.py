import torch
from transformers import (BertForSequenceClassification, Trainer,
                          TrainingArguments)

from dataset import NewsDataset, load_data, tokenize_data


def train_model(data_path, output_dir='./results'):
    train_texts, test_texts, train_labels, test_labels, uniques = load_data(data_path)
    train_encodings, test_encodings = tokenize_data(train_texts, test_texts)

    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(uniques))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=3,              
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=16,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        evaluation_strategy="steps",     
        eval_steps=100,                  
        save_steps=100,                  
        load_best_model_at_end=True,     
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=test_dataset            
    )

    trainer.train()

if __name__ == "__main__":
    train_model('../data/2024_4.csv')
