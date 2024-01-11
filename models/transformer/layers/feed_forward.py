import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
