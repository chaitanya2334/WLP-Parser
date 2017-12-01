from itertools import chain

from torch import nn, zeros, FloatTensor, cat, cuda
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import rnn

import config as cfg
from model.AttNet import AttNet
from model.CharNet import CharNet
from model.LMnet import LMnet
from model.multi_batch.MultiBatchCharNet import MultiBatchCharNet
from model.utils import to_scalar, TimeDistributed


class MultiBatchSeqNet(nn.Module):
    def __init__(self, emb_mat, batch_size, isCrossEnt=True, char_level="None", pos_feat="No", dep_rel_feat="No", dep_word_feat="No"):
        super().__init__()
        self.emb_mat_tensor = Variable(cuda.FloatTensor(emb_mat))
        assert self.emb_mat_tensor.size(1) == cfg.EMBEDDING_DIM
        self.vocab_size = self.emb_mat_tensor.size(0)
        self.emb_dim = self.emb_mat_tensor.size(1)
        self.hidden_size = cfg.LSTM_HIDDEN_SIZE
        self.batch_size = batch_size
        self.num_layers = 1
        self.num_dir = 2
        self.out_size = cfg.CATEGORIES
        self.pf_dim = cfg.PF_EMBEDDING_DIM
        self.char_emb_dim = cfg.EMBEDDING_DIM
        self.pos_feat = pos_feat
        self.dep_rel_feat = dep_rel_feat
        self.dep_word_feat = dep_word_feat
        self.char_level = char_level

        # init embedding layer, with pre-trained embedding matrix : emb_mat
        print("word embeddings are being trained using the following strategy: {0}".format(cfg.TRAIN_WORD_EMB))

        if cfg.TRAIN_WORD_EMB == "pre_and_post":
            self.emb_lookup = nn.Embedding(self.vocab_size, self.emb_dim)
            self.emb_lookup.weight = nn.Parameter(cuda.FloatTensor(emb_mat))

        elif cfg.TRAIN_WORD_EMB == "pre_only":
            self.emb_lookup = Embedding(self.emb_mat_tensor)

        elif cfg.TRAIN_WORD_EMB == "random":
            self.emb_lookup = nn.Embedding(self.vocab_size, self.emb_dim)

            tensor = self.__random_tensor(-0.01, 0.01, (self.vocab_size, self.emb_dim))
            self.emb_lookup.weight = nn.Parameter(tensor)

        self.char_net = MultiBatchCharNet(cfg.CHAR_EMB_DIM, cfg.CHAR_RECURRENT_SIZE, out_size=cfg.EMBEDDING_DIM)

        if pos_feat == "Yes":
            self.pos_emb = nn.Embedding(cfg.POS_VOCAB, cfg.POS_EMB_DIM)
            # initialize weights
            tensor = self.__random_tensor(-0.01, 0.01, (cfg.POS_VOCAB, cfg.POS_EMB_DIM))
            self.pos_emb.weight = nn.Parameter(tensor)
        if dep_rel_feat == "Yes":
            self.rel_emb = nn.Embedding(cfg.REL_VOCAB, cfg.REL_EMB_DIM)
            # initialize weights
            tensor = self.__random_tensor(-0.01, 0.01, (cfg.REL_VOCAB, cfg.REL_EMB_DIM))
            self.rel_emb.weight = nn.Parameter(tensor)

        self.att_net = AttNet(cfg.EMBEDDING_DIM, cfg.EMBEDDING_DIM, cfg.EMBEDDING_DIM)

        inp_size = self.emb_dim

        if self.char_level == "Input":
            inp_size += self.char_emb_dim

        if self.pos_feat == "Yes":
            inp_size += cfg.POS_EMB_DIM

        if self.dep_rel_feat == "Yes":
            inp_size += cfg.REL_EMB_DIM

        if self.dep_word_feat == "Yes":
            inp_size += self.emb_dim

        self.lstm = nn.LSTM(input_size=inp_size, batch_first=True,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

        self.lm_forward = TimeDistributed(LMnet(input_size=self.hidden_size,
                                                out_size=min(self.vocab_size + 1, cfg.LM_MAX_VOCAB_SIZE),
                                                hidden_size=cfg.LM_HIDDEN_SIZE), batch_first=True)

        self.lm_backward = TimeDistributed(LMnet(input_size=self.hidden_size,
                                                 out_size=min(self.vocab_size + 1, cfg.LM_MAX_VOCAB_SIZE),
                                                 hidden_size=cfg.LM_HIDDEN_SIZE), batch_first=True)

        self.lstm_linear = nn.Linear(self.hidden_size * 2, cfg.LSTM_OUT_SIZE)

        self.linear = nn.Linear(cfg.LSTM_OUT_SIZE,
                                self.out_size)

        # self.time_linear = TimeDistributed(self.linear, batch_first=True)
        self.init_state(self.batch_size)
        if not isCrossEnt:
            self.log_softmax = nn.LogSoftmax()

        self.isCrossEnt = isCrossEnt

    @staticmethod
    def __random_tensor(r1, r2, size):
        return (r1 - r2) * torch.rand(size) + r2

    def init_state(self, batch_size):
        """Get cell states and hidden states."""
        h0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size))
        c0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size))

        self.hidden_state = (h0_encoder_bi.cuda(), c0_encoder_bi.cuda())

    def pad(self, minibatch, pad_first=False, fix_length=None, include_lengths=True):
        """Pad a batch of examples.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        #if len(minibatch) < self.batch_size:
            # add empty lists to fill the batch size
            # so that later those samples are of size zero and get fully padded
        #    minibatch.extend([[]*(self.batch_size - len(minibatch))])

        pad_token = 0
        if fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = fix_length + (self.init_token, self.eos_token).count(None) - 2

        padded, lengths = [], []
        for x in minibatch:
            if pad_first:
                padded.append(
                    [pad_token] * max(0, max_len - len(x)) +
                    list(x[:max_len]))
            else:
                padded.append(
                    list(x[:max_len]) +
                    [pad_token] * max(0, max_len - len(x)))

            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

        if include_lengths:
            return padded, lengths

        return padded

    @staticmethod
    def unpad(padded, seq_lengths, skip_start=0, skip_end=0):
        # for every sequence in the batch, skip from the beginning skip_start no. of elements
        # same goes for skip_end, except from the end of every sequence.
        max_seq_len = padded.size(1)
        mask = [[0]*skip_start + [1]*(seq_length - skip_end - skip_start) + [0]*(max_seq_len - seq_length + skip_end)
                for seq_length in seq_lengths]

        mask = list(chain.from_iterable(mask))
        mask_tensor = torch.ByteTensor(mask).cuda()
        index = torch.nonzero(mask_tensor)
        # padded of size (batch_size x seq_len x emb_dim)
        index = index.squeeze(dim=1)
        emb_dim = padded.size(2)
        padded = padded.contiguous().view(-1, emb_dim)
        # padded of size (batch_size*len x emb_dim)

        unpadded = torch.index_select(padded, dim=0, index=Variable(index))
        return unpadded

    def mb_lstm_forward(self, padded, seq_lengths):
        # multi batch lstm forward run.
        # input is a batch

        pked_pdded_seq = rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True)

        pked_pdded_out, hidden_state = self.lstm(pked_pdded_seq, self.hidden_state)

        pdded_out, lens = rnn.pad_packed_sequence(pked_pdded_out, batch_first=True)

        return pdded_out, hidden_state

    def forward(self, sent_idx_seq, char_idx_seq, pos, rel, dep_word):
        cfg.ver_print("Sent Index sequence", sent_idx_seq)
        padded_seq, seq_lengths = self.pad(sent_idx_seq)
        padded_seq = Variable(torch.LongTensor(padded_seq)).cuda()

        emb = self.emb_lookup(padded_seq)

        if self.char_level == "Input":
            char_emb = self.char_net(char_idx_seq)
            inp = cat([emb, char_emb], dim=2)

        elif self.char_level == "Attention":
            char_emb = self.char_net(char_idx_seq)
            inp = self.att_net(emb, char_emb)
        else:
            inp = emb

        if self.pos_feat == "Yes":
            padded_pos, seq_len_pos = self.pad(pos)
            padded_pos = Variable(torch.LongTensor(padded_pos)).cuda()
            pos_emb = self.pos_emb(padded_pos)
            inp = cat([inp, pos_emb], dim=2)
        if self.dep_rel_feat == "Yes":
            rel_emb = self.rel_emb(rel)
            inp = cat([inp, rel_emb], dim=2)
        if self.dep_word_feat == "Yes":
            dep_emb = self.emb_lookup(dep_word).data.clone()
            dep_emb = Variable(dep_emb)
            inp = cat([inp, dep_emb], dim=2)

        # emb is now of size(1 x seq_len x EMB_DIM)
        cfg.ver_print("Embedding for the Sequence", inp)
        lstm_out, hidden_state = self.mb_lstm_forward(inp, seq_lengths)
        # lstm_out is of size (batch_size x seq_len x 2*EMB_DIM)
        unrolled_lstm_out = self.unpad(lstm_out, seq_lengths)
        # unrolled_lstm_out is of size(label_size x 2*EMB_DIM); where label_size is the number of words in the batch.
        lstm_forward, lstm_backward = lstm_out[:, :, :cfg.LSTM_HIDDEN_SIZE], lstm_out[:, :, -cfg.LSTM_HIDDEN_SIZE:]
        # lstm_forward of size (batch x max_seq x emb_dim)
        # making sure that you got the correct lstm_forward and lstm_backward.
        for i, seq_len in enumerate(seq_lengths):
            assert to_scalar(torch.sum(lstm_forward[i, seq_len - 1, :] - hidden_state[0][0, i, :])) == 0
            assert to_scalar(torch.sum(lstm_backward[i, 0, :] - hidden_state[0][1, i, :])) == 0

        lm_f_out = self.lm_forward(self.unpad(lstm_forward, seq_lengths, skip_start=0, skip_end=1))
        lm_b_out = self.lm_backward(self.unpad(lstm_backward, seq_lengths, skip_start=1, skip_end=0))
        # size of lm_f_out = (batch_size*seq_len x emb_size)
        cfg.ver_print("Language Model Forward pass out", lm_f_out)
        cfg.ver_print("Language Model Backward pass out", lm_b_out)

        lstm_out = self.lstm_linear(unrolled_lstm_out.squeeze())

        lstm_out = torch.sigmoid(lstm_out)

        lstm_out = lstm_out.unsqueeze(dim=0)

        label_out = lstm_out

        linear_out = self.linear(label_out.view(-1, cfg.LSTM_OUT_SIZE))
        if self.isCrossEnt:
            out = linear_out
        else:
            out = self.log_softmax(linear_out)

        cfg.ver_print("LINEAR OUT", linear_out)
        cfg.ver_print("FINAL OUT", out)

        if self.char_level == "Attention":
            unrolled_emb = self.unpad(emb, seq_lengths)
            unrolled_char_emb = self.unpad(char_emb, seq_lengths)
            return lm_f_out, lm_b_out, out, seq_lengths, unrolled_emb, unrolled_char_emb
        else:
            return lm_f_out, lm_b_out, out, seq_lengths


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
