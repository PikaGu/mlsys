import torch.nn as nn

class SubLayer(nn.Module):
    def __init__(self, size, drop_prob):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sublayer, pre_ln=True):
        if pre_ln:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
