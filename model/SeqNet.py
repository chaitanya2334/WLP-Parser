from torch import nn, zeros, FloatTensor, cat, cuda
import torch
from torch.autograd import Variable
import numpy as np

import config as cfg
from model.utils import to_scalar


class SeqNet(nn.Module):
    def __init__(self, emb_mat):
        super().__init__()
        self.emb_mat_tensor = Variable(cuda.FloatTensor(emb_mat))
        assert self.emb_mat_tensor.size(1) == cfg.EMBEDDING_DIM
        self.vocab_size = self.emb_mat_tensor.size(0)
        self.emb_dim = self.emb_mat_tensor.size(1)
        self.hidden_size = cfg.LSTM_HIDDEN_SIZE
        self.batch_size = 1  # we can only do batch size 1.
        self.num_layers = 1
        self.num_dir = 2
        self.out_size = cfg.CATEGORIES
        self.pf_dim = cfg.PF_EMBEDDING_DIM

        # init embedding layer, with pre-trained embedding matrix : emb_mat
        self.emb_lookup = Embedding(self.emb_mat_tensor)

        self.lstm = nn.LSTM(input_size=self.emb_dim, batch_first=True, num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * self.num_dir,
                                self.out_size)

        # self.time_linear = TimeDistributed(self.linear, batch_first=True)

        self.hidden_state = self.init_state()

        self.log_softmax = nn.LogSoftmax()

    def init_state(self):
        """Get cell states and hidden states."""
        h0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_size))
        c0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_size))

        return h0_encoder_bi.cuda(), c0_encoder_bi.cuda()

    def forward(self, sent_idx_seq):
        cfg.ver_print("Sent Index sequence", sent_idx_seq)

        seq_len = sent_idx_seq.size(1)

        emb = self.emb_lookup(sent_idx_seq)

        # emb is now of size(1 x seq_len x EMB_DIM)
        cfg.ver_print("Embedding for the Sequence", emb)

        lstm_out, self.hidden_state = self.lstm(emb, self.hidden_state)
        # lstm_out is of size (1 x seq_len x 2*EMB_DIM)

        lstm_forward, lstm_backward = lstm_out[:, :, :cfg.LSTM_HIDDEN_SIZE], lstm_out[:, :, -cfg.LSTM_HIDDEN_SIZE:]

        # making sure that you got the correct lstm_forward and lstm_backward.
        assert to_scalar(torch.sum(lstm_forward[:, seq_len-1, :] - self.hidden_state[0][0, :, :])) == 0
        assert to_scalar(torch.sum(lstm_backward[:, 0, :] - self.hidden_state[0][1, :, :])) == 0



        linear_out = self.linear(lstm_out.view(seq_len, -1))
        soft_out = self.log_softmax(linear_out)

        cfg.ver_print("LINEAR OUT", linear_out)
        cfg.ver_print("FINAL OUT", soft_out)

        return soft_out

        # My shame :(
        # for x in range(lstm_out.size(1)):
        #    linear_out = self.linear(lstm_out[0][x].view(1, self.hidden_size*self.num_dir))
        #    soft_out = softmax(linear_out)
        #    out_list.append(soft_out)

        # final_out = cat(out_list)
        # return final_out


class Embedding(nn.Module):
    def __init__(self, emb_mat):
        super().__init__()
        self.emb_mat = emb_mat

    def forward(self, x):
        v_l = []
        cfg.ver_print("input to embedding layer", x)
        # x is of size (1 x seq_len)
        seq_len = x.size(1)

        n = x[0].cpu().data.numpy()

        for i in n.tolist():
            v = self.emb_mat[i]
            # v is of size (EMB_DIM)
            # cfg.ver_print("v", v)

            v_l.append(v)
        v = torch.stack(v_l, dim=0)
        # v is of size (seq_len x EMB_DIM)

        v = v.view(1, seq_len, -1)
        # v is now of size(1 x seq_len x EMB_DIM)

        # cfg.ver_print("Embedding out", v)

        return v
