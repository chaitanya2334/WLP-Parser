from itertools import chain

import torch
from torch import nn, zeros, cat, stack, cuda, unsqueeze
from torch.autograd import Variable
from torch.nn.utils import rnn

import config as cfg


class MultiBatchCharNet(nn.Module):
    def __init__(self, emb_size, recurr_size, out_size):
        super().__init__()
        self.num_layers = 1
        self.num_dir = 2
        self.batch_size = 1
        self.out_size = out_size
        self.hidden_size = recurr_size
        self.emb = nn.Embedding(cfg.CHAR_VOCAB, embedding_dim=emb_size)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=self.hidden_size * self.num_dir, out_features=out_size)
        self.tanh = nn.Tanh()
        self.hidden_state = None
        self.init_state(self.batch_size)
        self.init_weights()

    def init_state(self, batch_size):
        h0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size))
        c0_encoder_bi = Variable(zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size))
        self.hidden_state = (h0_encoder_bi.cuda(), c0_encoder_bi.cuda())

    def init_weights(self):
        initrange = 0.01
        self.emb.weight.data.uniform_(-initrange, initrange)

    def pad(self, minibatch, pad_token, pad_first=False, fix_length=None, include_lengths=True):
        """Pad a batch of examples.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)

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

    def split_and_pad(self, tensor1d, seq_lens):
        start = 0
        max_seq_len = seq_lens[0]
        out_stack = []
        # print(seq_lens)
        for seq_len in seq_lens:
            out = torch.index_select(tensor1d, dim=0,
                                     index=Variable(torch.arange(start, start + seq_len).long()).cuda())
            # print(out.size())
            if max_seq_len - seq_len > 0:
                out_stack.append(
                    cat(
                        [out, Variable(torch.zeros((max_seq_len - seq_len, self.out_size))).cuda()], dim=0))
            else:
                out_stack.append(out)
            start += seq_len

        out = torch.stack(out_stack, dim=0)
        return out

    @staticmethod
    def len_sort(seq):
        # sorts the list by length, also returns arg-index so that you can revert back
        idx, sent = zip(*sorted(enumerate(seq), key=lambda x: len(x[1]), reverse=True))
        ridx = sorted(range(len(seq)), key=idx.__getitem__)
        return sent, ridx

    def forward(self, minibatch):
        out_stack = []
        minibatch = list(minibatch)
        minibatch_lengths = [len(sent) for sent in minibatch]
        batch_of_words = list(chain.from_iterable(minibatch))

        self.init_state(len(batch_of_words))
        # a hack to get index of the sorted words, so i can unsort them back after they are processed
        # print(batch_of_words)
        sent, ridx = self.len_sort(batch_of_words)
        padded, seq_lengths = self.pad(sent, 0)
        # print(padded)
        out = self.emb(Variable(cuda.LongTensor(padded)))
        # out is of size (all_words x max_len x char_emb_size)
        # print("out size: {0}".format(out.size()))
        out = rnn.pack_padded_sequence(out, seq_lengths, batch_first=True)
        out, hidden_state = self.rnn(out, self.hidden_state)
        # hidden_state[0] is of size: (num_dir x batch_size x lstm_hidden_dim)
        # print("hidden state size: {0}".format(hidden_state[0].size()))

        # TODO verify
        # unsorting IMPORTANT. cos we initially sorted the seq of chars to pass it to rnn.
        hidden_state = torch.index_select(hidden_state[0], dim=1, index=Variable(cuda.LongTensor(ridx)))

        # TODO verify that this is indeed the last outputs of both forward rnn and backward rnn

        out = cat([hidden_state[0], hidden_state[1]], dim=1)
        # print("cat out size: {0}".format(out.size()))
        cfg.ver_print("Hidden state concat", out)
        out = self.linear(out)
        out = self.tanh(out)
        # print("before split and pad function {0}".format(out.size()))
        # this will split 1d tensor of word embeddings, into 2d array of word embeddings based on lengths
        final_out = self.split_and_pad(out, minibatch_lengths)
        # final_out is of size (batch_size x max_seq_len x emb_size)
        # print(final_out.size())
        return final_out
