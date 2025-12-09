import torch.nn as nn

class LateFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_text, logits_audio):
        return (logits_text + logits_audio) / 2
