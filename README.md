AI Emotional Awareness Model (Text + Audio MVP)

Install:
pip install -r requirements.txt

Text:
python -m src.text.train_text
python -m src.text.app

Audio:
python -m src.audio.train_audio
python -m src.audio.app

Run APP:
python app.py

WSL:
source ~/mlenv/bin/activate
cd /mnt/e/ai-emotional-awareness/
watch -n 1 nvidia-smi

Jupyter Notebook Install and Run in WSL:
pip install notebook ipykernel
python -m ipykernel install --user --name wsl-ml --display-name "WSL ML"
jupyter notebook --no-browser

Select kernel → WSL ML