from torch import nn, zeros, cat, stack, cuda, unsqueeze
from torch.autograd import Variable
import config as cfg


class CharNet(nn.Module):
    def __init__(self, emb_size, recurr_size, out_size):
        super().__init__()
        self.num_layers = 1
        self.num_dir = 2
        self.batch_size = 1
        self.hidden_size = recurr_size
        self.emb = nn.Embedding(cfg.CHAR_VOCAB, embedding_dim=emb_size, scale_grad_by_freq=True)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=self.hidden_size * self.num_dir, out_features=out_size)
        self.tanh = nn.Tanh()
        self.hidden_state = None
        self.init_state()
        self.init_weights()

    def init_state(self):
        h0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_size))
        c0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_size))
        self.hidden_state = (h0_encoder_bi.cuda(), c0_encoder_bi.cuda())

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, chars):
        out_stack = []

        for word in chars:

            out = self.emb(Variable(cuda.LongTensor(word)))
            out = unsqueeze(out, dim=0)

            out, hidden_state = self.rnn(out, self.hidden_state)

            # TODO verify that this is indeed the last outputs of both forward rnn and backward rnn
            # and that we are concatenating correctly
            out = cat([hidden_state[0][0], hidden_state[0][1]], dim=1)
            cfg.ver_print("Hidden state concat", out)
            out = self.linear(out)
            out = self.tanh(out)
            out_stack.append(out)

        final_out = stack(out_stack, dim=1)
        return final_out
