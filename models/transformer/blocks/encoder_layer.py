from layers import *
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, drop_prob):
        super().__init__()
        self.sub_layers = nn.ModuleList(SubLayer(d_model, drop_prob) for _ in range(2))
        self.ffn = FFN(d_model, d_ff, drop_prob)
        self.mha = MultiHeadAttention(d_model, n_head, drop_prob)

    def forward(self, x, mask):
        x = self.sub_layers[0](x, lambda x: self.mha(x, x, x, mask))
        x = self.sub_layers[1](x, self.ffn)
