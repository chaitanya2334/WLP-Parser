from torch import nn, cat, unsqueeze, mul, squeeze


class AttNet(nn.Module):
    def __init__(self, x_size, m_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(x_size + m_size, out_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(out_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        xm = cat([x, m], dim=2)
        batch_size, seq_size, emb_size = xm.size()
        xm = xm.view(-1, emb_size)

        z = self.linear1(xm)
        z = self.tanh(z)
        z = self.linear2(z)
        z = self.sigmoid(z)
        z = z*x + (1-z)*m

        out = z.view(batch_size, seq_size, -1)
        return out
