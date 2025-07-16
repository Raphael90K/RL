import torch
import torch.nn as nn
from torchsummary import summary


class QNetwork(nn.Module):

    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=3, padding=1),  # (C, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )

        dummy_input = torch.zeros(1, c, h, w)
        with torch.no_grad():
            encoded = self.encoder(dummy_input)
            encoded_size = encoded.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(encoded_size, 8),
            nn.ReLU(),
            nn.Linear(8, num_actions)
        )

    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        return self.head(x)

    def summary(self):
        print("Model Summary:")
        summary(self, (20, 7, 7))
