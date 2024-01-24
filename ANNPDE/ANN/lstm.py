from ..PDE.objects import ReverseChauchyPDE
from torch import nn
import torch


class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_layer_sizes):
        super(LSTM, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.hidden_layers.append(
                    nn.LSTM(input_size, hidden_layer_sizes[i], batch_first=True)
                )
            else:
                self.hidden_layers.append(
                    nn.LSTM(
                        hidden_layer_sizes[i-1], 
                        hidden_layer_sizes[i]
                    )
                )
        
        self.linear = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, x):
        for lstm in self.hidden_layers:
            x, _ = lstm(x)
        x = self.linear(x)  # Take the last time step's output
        return x
