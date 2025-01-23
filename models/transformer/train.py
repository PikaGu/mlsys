from layers.multi_head_attention import MultiHeadAttention

import torch

def train():
    d_model = 512   # 嵌入维度
    h = 8           # 注意力头数
    drop_prob = 0.1 # Dropout 概率
    seq_len = 10    # 序列长度
    batch_size = 4  # 批大小

    mha = MultiHeadAttention(d_model, h, drop_prob)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

    output = mha(query, key, value, mask)
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    train()
