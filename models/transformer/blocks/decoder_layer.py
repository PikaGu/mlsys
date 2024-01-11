from layers import *
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, drop_prob):
        super().__init__()
        self.sub_layers = nn.ModuleList(SubLayer(d_model, drop_prob) for _ in range(3))
        self.ffn = FFN(d_model, d_ff, drop_prob)
        self.src_attn = MultiHeadAttention(d_model, n_head, drop_prob)
        self.self_attn = MultiHeadAttention(d_model, n_head, drop_prob)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sub_layers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sub_layers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sub_layers[2](x, self.ffn)
