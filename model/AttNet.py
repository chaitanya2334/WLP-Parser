from torch import nn, cat, unsqueeze


class AttNet(nn.Module):
    def __init__(self, x_size, m_size, out_size):
        super().__init__()
        print(x_size + m_size)
        self.linear1 = nn.Linear(x_size + m_size, out_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(out_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        xm = cat([x, m], dim=2)
        xm = xm.squeeze()
        z = self.linear1(xm)
        z = self.tanh(z)
        z = self.linear2(z)
        z = self.sigmoid(z)

        out = z*x + (1-z)*m
        out = unsqueeze(out, 0)
        return out
