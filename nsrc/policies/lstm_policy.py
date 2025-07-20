import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, lstm_hidden_size=256):
        super().__init__(observation_space, features_dim)
        obs_size = observation_space.shape[0]
        self.lstm = nn.LSTM(input_size=obs_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self._lstm_hidden_size = lstm_hidden_size
        self.hidden_state = None

    def forward(self, observations):
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        lstm_out, self.hidden_state = self.lstm(observations, self.hidden_state)
        out = lstm_out[:, -1, :]
        return self.linear(out)
