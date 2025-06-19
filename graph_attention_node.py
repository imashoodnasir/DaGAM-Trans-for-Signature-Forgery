class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data)

    def forward(self, h):
        Wh = self.W(h)
        a_input = torch.cat([Wh.repeat(1, 1, Wh.size(1)).view(Wh.size(0), -1, Wh.size(2)),
                             Wh.repeat(1, Wh.size(1), 1)], dim=2)
        e = torch.matmul(a_input, self.attn).squeeze(2)
        attn = torch.softmax(e, dim=1)
        return torch.bmm(attn.unsqueeze(1), Wh).squeeze(1)
