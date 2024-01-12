from embedding import TransformerEmbedding
from encoder import Encoder
from decoder import Decoder
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        pass