from torch import nn, cuda
from torch.autograd import Variable
import config as cfg


class LMnet(nn.Module):
    def __init__(self, input_size, out_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.tanh = nn.Tanh()

        self.linear2 = nn.Linear(self.hidden_size, self.out_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, inp_features):
        cfg.ver_print("Inp features", inp_features)
        # inp_features is of size (seq_len x EMB_DIM)

        linear1_out = self.linear1(inp_features)
        tanh_out = self.tanh(linear1_out)
        linear2_out = self.linear2(tanh_out)
        soft_out = self.log_softmax(linear2_out)

        cfg.ver_print("FINAL OUT", soft_out)

        return soft_out
