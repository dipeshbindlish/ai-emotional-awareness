import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,16,3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.cnn(x).squeeze(-1)
        return self.fc(x)
