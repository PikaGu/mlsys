import copy
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, drop_prob):
        super().__init__()
        linear = nn.Linear(d_model, d_model)
        self.linears = nn.ModuleList(copy.deepcopy(linear) for _ in range(4))
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = torch.unsqueeze(mask, 1) # (batch_size, seq_len) -> (batch_size, 1, seq_len)
        batch_size = query.size(0)

        # (batch_size, seq_len, d_model)
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x = self.attention(query, key, value, mask, self.dropout)        
        x = x.transpose().contiguous().view(batch_size, -1, self.h * self.d_k)

        del query
        del key
        del value
        return self.linears[-1](x)

    def attention(self, query, key, value, mask, dropout):
        scores = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = torch.masked_fill(scores, mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, value)
