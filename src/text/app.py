import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.getenv("TEXT_MODEL_PATH")
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return label_names[pred]


iface = gr.Interface(
    fn=predict, inputs="text", outputs="text", title="Emotion Classifier"
)
iface.launch(
    # share=True # Share publicly via Gradio's servers
)