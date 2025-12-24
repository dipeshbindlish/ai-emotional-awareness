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
python app.py

WSL:
source ~/mlenv/bin/activate
cd /mnt/e/ai-emotional-awareness/
python -m src.training.train_text



Jupyter Notebook Install and Run in WSL:
pip install notebook ipykernel
python -m ipykernel install --user --name wsl-ml --display-name "WSL ML"
jupyter notebook --no-browser

Select kernel → WSL ML

watch -n 1 nvidia-smi