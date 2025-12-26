from pathlib import Path
import soundfile as sf
import librosa
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# import intel_extension_for_pytorch as ipex
from dotenv import load_dotenv
import os

load_dotenv()

AUDIO_DATASET = os.getenv("AUDIO_DATASET")
DATASET_ROOT = Path(AUDIO_DATASET)
audio_files = list(DATASET_ROOT.rglob("*.wav"))

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_emotion(path: Path):
    parts = path.stem.split("-")
    emotion_code = parts[2]
    return emotion_map[emotion_code]


samples = []
for f in audio_files:
    samples.append({"path": str(f), "emotion": parse_emotion(f)})

waveform, sr = sf.read(samples[0]["path"])
if sr != 16000:
    waveform_16k = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    sr = 16000
else:
    waveform_16k = waveform

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

inputs = processor(waveform_16k, sampling_rate=sr, return_tensors="pt", padding=True)

labels = sorted(set(s["emotion"] for s in samples))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

dataset = Dataset.from_list(samples)


def encode_labels(example):
    example["label"] = label2id[example["emotion"]]
    return example


dataset = dataset.map(encode_labels)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


def preprocess_audio(batch):
    waveforms = []
    labels = []

    for path, label in zip(batch["path"], batch["label"]):
        waveform, sr = sf.read(path)

        # convert stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # resample
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

        waveforms.append(waveform)
        labels.append(label)

    inputs = processor(
        waveforms, sampling_rate=16000, padding=True, return_tensors="pt"
    )

    inputs["labels"] = labels
    return inputs


dataset = dataset.map(
    preprocess_audio, batched=True, batch_size=8, remove_columns=["path", "emotion"]
)
dataset.set_format("torch")

num_labels = len(label2id)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

model.freeze_feature_extractor()
# for param in model.wav2vec2.feature_extractor.parameters():
#     param.requires_grad = False


# Set device to CPU
# device = torch.device("cpu")
# model.to(device)
# model = ipex.optimize(model, dtype=torch.float32) # Optimize the model for your CPU hardware

training_args = TrainingArguments(
    output_dir="audio_out",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    # CPU:
    # use_cpu=True,  # set to True to force CPU training
    # # bf16=True,  # Use ONLY if you have Intel Extension for PyTorch (IPEX)
    # # use_ipex=True,  # Use ONLY if you have Intel Extension for PyTorch (IPEX)
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    # gradient_accumulation_steps=1,
    # dataloader_num_workers=4,  # speed up data loading
    # dataloader_pin_memory=False,  # speed up data transfer to GPU
    # fp16=False,  # set True only if GPU supports it
    # GPU:
    fp16=False,  # set True only if GPU supports it
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    dataloader_num_workers=2,  # speed up data loading
    dataloader_pin_memory=True,  # speed up data transfer to GPU
    num_train_epochs=10,
    logging_steps=10,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


data_collator = DataCollatorWithPadding(
    tokenizer=processor, padding=True, return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
