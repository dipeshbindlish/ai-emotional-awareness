from transformers import AutoModelForSequenceClassification

def get_model():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
