from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

model_name = "distilbert-base-uncased"

ds = load_dataset("emotion")
label_names = ds["train"].features["label"].names

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Model and Tokenizer to begin training from scratch
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_names)
)

# Load Pre-trained model from previous training
model = AutoModelForSequenceClassification.from_pretrained("text_out/checkpoint-2000")


def tok(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=64
    )


ds = ds.map(tok, batched=True)

ds = ds.remove_columns(["text"])
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch")

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)


args = TrainingArguments(
    output_dir="text_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    report_to=[],  # Disable logging to WandB
    # no_cuda=True  # Force CPU for Training
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

# trainer.train()
# trainer.train()
results = trainer.evaluate(ds["test"])
print(results)
