import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2), nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 16), nn.ReLU(),
            nn.Linear(16, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

    def save_checkpoint(self):
        torch.save(self.state_dict(), 'q_network.pth')
        print("Checkpoint saved to q_network.pth")

    def show_network(self):
        print("Network Architecture:")
        print(self.conv)
        print(self.fc)

    def summary(self):
        print("Model Summary:")
        summary(self, (3, 56, 56))
