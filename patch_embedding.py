import torch.nn as nn

class PatchEmbedder(nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 196, embed_dim))  # Assuming 14x14 patches

    def forward(self, x):  # x: [B, N, P^2]
        x = self.projection(x)
        x = x + self.positional_encoding[:, :x.size(1)]
        return x
