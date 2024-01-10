from embedding import TransformerEmbedding
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, N, drop_prob):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, seq_len, d_model)

    def forward(self, x):
        pass
