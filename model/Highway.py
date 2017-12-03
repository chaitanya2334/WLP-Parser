from torch import nn, cat, unsqueeze, mul, squeeze


class HighwayNet(nn.Module):
    def __init__(self, x_size, m_size):
        super().__init__()
        self.T = nn.Linear(x_size + m_size, x_size + m_size)
        self.relu = nn.ReLU()
        self.H = nn.Linear(x_size + m_size, x_size + m_size)
        self.relu = nn.ReLU()

    def forward(self, x, m):
        xm = cat([x, m], dim=2)
        batch_size, seq_size, emb_size = xm.size()
        xm = xm.view(-1, emb_size)

        h = self.H(xm)
        h = self.relu(h)

        t = self.T(xm)
        t = self.relu(t)

        #z = self.linear2(z)
        #z = self.sigmoid(z)
        z = t*h + (1-t)*xm

        out = z.view(batch_size, seq_size, -1)
        return out
