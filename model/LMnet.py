from torch import nn, cuda
from torch.autograd import Variable
import config as cfg

class LMnet(nn.Module):
    def __init__(self):
        super().__init__()
        assert self.emb_mat_tensor.size(1) == cfg.EMBEDDING_DIM
        self.vocab_size = self.emb_mat_tensor.size(0)
        self.emb_dim = self.emb_mat_tensor.size(1)
        self.hidden_size = cfg.LSTM_HIDDEN_SIZE
        self.batch_size = 1  # we can only do batch size 1.
        self.num_layers = 1
        self.num_dir = 2
        self.out_size = cfg.CATEGORIES
        self.pf_dim = cfg.PF_EMBEDDING_DIM

        self.linear = nn.Linear(self.hidden_size * self.num_dir,
                                self.out_size)

        # self.time_linear = TimeDistributed(self.linear, batch_first=True)

        self.hidden_state = self.init_state()

        self.log_softmax = nn.LogSoftmax()

    def forward(self, sent_idx_seq):
        cfg.ver_print("Sent Index sequence", sent_idx_seq)

        linear_out = self.linear(lstm_out.view(seq_len, -1))
        soft_out = self.log_softmax(linear_out)

        cfg.ver_print("LINEAR OUT", linear_out)
        cfg.ver_print("FINAL OUT", soft_out)

        return soft_out
