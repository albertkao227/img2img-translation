import torch
import torch.nn as nn
import torch.nn.functional as F


class Palette(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Palette, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.out(x)
        return x, hidden

