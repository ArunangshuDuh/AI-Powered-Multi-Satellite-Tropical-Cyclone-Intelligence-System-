
import torch
import torch.nn as nn
import config_reference as config

class TrendGRU(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_size = 1285 # 1280 (CNN) + 5 (wind, pressure, lat, lon, hour)
        self.hidden_size = 128
        
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.head_6h = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.head_12h = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.head_24h = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, sequence):
        # sequence: [B, T, 1285]
        output, hidden = self.gru(sequence)
        
        # Final hidden state
        final_state = output[:, -1, :] # [B, 128]
        
        res = {
            'plus_6h': self.head_6h(final_state),
            'plus_12h': self.head_12h(final_state),
            'plus_24h': self.head_24h(final_state)
        }
        
        return res
