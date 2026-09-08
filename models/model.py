"""
Multi-task tropical cyclone model.

TCModel: takes a sequence of past observations (lat, lon, wind, pressure, month, hour)
and outputs:
  - current category classification (7-way)
  - current wind/pressure regression (mostly a sanity-check head, since these come
    from the input; the real value is the shared representation feeding the GRU)
  - +6h/+12h/+24h wind/pressure trend prediction (GRU decoder)

This numeric-only version stands on its own today (trained on real HURDAT2 data).
It is deliberately structured so an image-CNN branch (trained on TCIR once available)
can be added later: its embedding would concatenate into `fused` in TCModel.forward
alongside the GRU's numeric embedding, without changing the two output heads or the
prediction decoder.
"""
import torch
import torch.nn as nn


NUM_CATEGORIES = 7
FEATURES_PER_STEP = 6  # lat, lon, wind, pressure, month, hour


class TCModel(nn.Module):
    def __init__(self, hidden_size=64, gru_layers=2, dropout=0.2):
        super().__init__()

        # Encodes the history sequence (24h of 6-hourly observations)
        self.encoder_gru = nn.GRU(
            input_size=FEATURES_PER_STEP,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Metadata fusion (month/hour of the *current* observation, cyclical-encoded outside)
        self.meta_fc = nn.Sequential(
            nn.Linear(4, 16),  # sin(month), cos(month), sin(hour), cos(hour)
            nn.ReLU(),
        )

        fused_size = hidden_size + 16

        self.fusion = nn.Sequential(
            nn.Linear(fused_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: classification (current category)
        self.classifier_head = nn.Linear(128, NUM_CATEGORIES)

        # Head 2: regression (current wind, pressure) - sanity/auxiliary head
        self.regressor_head = nn.Linear(128, 2)

        # Head 3: trend prediction decoder -> (+6h, +12h, +24h) x (wind, pressure) = 6 outputs
        self.trend_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, history, meta):
        """
        history: (batch, seq_len, FEATURES_PER_STEP) float tensor
        meta: (batch, 4) float tensor -> [sin(month), cos(month), sin(hour), cos(hour)]
        """
        _, h_n = self.encoder_gru(history)
        seq_embedding = h_n[-1]  # (batch, hidden_size) - last layer's final hidden state

        meta_embedding = self.meta_fc(meta)

        fused = torch.cat([seq_embedding, meta_embedding], dim=1)
        fused = self.fusion(fused)

        category_logits = self.classifier_head(fused)
        regression_out = self.regressor_head(fused)     # (batch, 2) -> wind, pressure
        trend_out = self.trend_head(fused)               # (batch, 6) -> 3 horizons x 2

        return {
            "category_logits": category_logits,
            "regression": regression_out,
            "trend": trend_out,
        }


def get_persistence_baseline(current_wind, current_pressure, n_horizons=3):
    """
    Persistence baseline: predict no change from the current value.
    Used to compute a defensible 'beats baseline' comparison metric.
    """
    import numpy as np
    wind_pred = np.tile(current_wind[:, None], (1, n_horizons))
    pressure_pred = np.tile(current_pressure[:, None], (1, n_horizons))
    return wind_pred, pressure_pred
