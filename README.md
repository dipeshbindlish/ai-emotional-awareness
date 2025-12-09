AI Emotional Awareness Model (Text + Audio MVP)

Install:
pip install -r requirements.txt

Train text:
python src/training/train_text.py
python -m src.training.train_text

Train audio:
python src/training/train_audio.py
python -m src.training.train_audio

Run API:
python src/api/app.py

WSL:
source ~/mlenv/bin/activate
cd /mnt/e/ai-emotional-awareness/
python -m src.training.train_text