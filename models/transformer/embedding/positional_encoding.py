import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pos = torch.arange(0, seq_len)
        pos = pos * torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)

        pe = torch.zeros(seq_len, d_model).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos)
        pe[:, 1::2] = torch.cos(pos)
        pe.unsqueeze(0) # size: (1, seq_len, d_model), dim=0 for batch
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:x.size(1), :].requires_grad_(False)
