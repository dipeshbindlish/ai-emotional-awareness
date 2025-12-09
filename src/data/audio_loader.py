import torchaudio

def load_audio(path):
    wav, sr = torchaudio.load(path)
    return wav, sr
