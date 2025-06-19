class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.gat = GraphAttentionLayer(channels // groups, channels // groups)

    def forward(self, x):  # x: [B, C, 1, 1]
        x = x.view(x.size(0), self.groups, -1)  # [B, g, C/g]
        out = torch.stack([self.gat(group) for group in x], dim=1)
        return out.view(x.size(0), -1)
