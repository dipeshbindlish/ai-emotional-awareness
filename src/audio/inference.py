import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv()

AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH")
TARGET_SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
model.to(device)
model.eval()

id2label = model.config.id2label


def load_audio(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


@torch.no_grad()
def predict_emotion(file_path):
    audio = load_audio(file_path)

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    confidence, pred_id = torch.max(probs, dim=0)

    return {
        "emotion": id2label[pred_id.item()],
        "confidence": round(confidence.item(), 4)
    }