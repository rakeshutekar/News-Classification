import torch
from transformers import BertTokenizer

from train import train_model


def classify_headline(headline, trainer, tokenizer, uniques):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model = trainer.model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return uniques[predicted_class]

if __name__ == "__main__":
    train_texts, test_texts, train_labels, test_labels, uniques = load_data('../data/2024_4.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Assuming train_model function returns the trainer instance
    trainer = train_model('../data/2024_4.csv')

    headline = "New breakthrough in AI technology"
    prediction = classify_headline(headline, trainer, tokenizer, uniques)
    print(f"The headline is classified as: {prediction}")
