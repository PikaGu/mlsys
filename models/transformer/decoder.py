from blocks import DecoderLayer
from embedding import TransformerEmbedding

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, ffn_hidden, n_head, N, drop_prob):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, seq_len, d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(N))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return x
