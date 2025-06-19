class DaGAMTrans(nn.Module):
    def __init__(self, patch_dim, embed_dim, heads, num_classes):
        super().__init__()
        self.embedding = PatchEmbedder(patch_dim, embed_dim)
        self.transformer = MultiHeadAttention(embed_dim, heads)
        self.gat_node = GraphAttentionLayer(embed_dim, embed_dim)
        self.cam = ChannelAttentionModule(embed_dim, groups=4)
        self.pool = GMSAPool(embed_dim, heads)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.gat_node(x)
        cam_out = self.cam(x.mean(dim=1, keepdim=True))  # simplified input to CAM
        x = x + cam_out.unsqueeze(1)
        pooled = self.pool(x)
        return self.classifier(pooled)
