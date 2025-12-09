import gradio as gr
from transformers import pipeline

clf = pipeline("text-classification", model="bert-base-uncased")

def predict(text):
    return clf(text)[0]["label"]

gr.Interface(fn=predict, inputs="text", outputs="label").launch()
