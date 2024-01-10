from embedding import PositionalEncoding, TokenEmbedding
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, drop_prob=0.5):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = PositionalEncoding(seq_len=seq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        return self.dropout(self.pos_emb(x) + self.tok_emb(x))
