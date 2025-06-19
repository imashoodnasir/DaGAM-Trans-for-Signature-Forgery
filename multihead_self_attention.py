import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = embed_dim ** 0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: t.view(B, N, self.heads, C // self.heads).transpose(1, 2), qkv)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn = scores.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        return self.fc_out(out)
