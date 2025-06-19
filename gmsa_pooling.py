class GMSAPool(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = heads
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.virtual_node = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        x = torch.cat([self.virtual_node.expand(x.size(0), -1, -1), x], dim=1)
        scores = torch.matmul(x, x.transpose(1, 2))  # self-attention
        scores = scores.softmax(dim=-1)
        out = torch.matmul(scores, x)
        return self.linear(out[:, 0])  # return virtual node representation
