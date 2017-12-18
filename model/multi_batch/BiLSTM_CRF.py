import torch
from torch import nn, cuda
from torch.autograd import Variable
import config as cfg


class BiLSTM_CRF(nn.Module):

    def __init__(self, emb_mat, tag_idx):
        super(BiLSTM_CRF, self).__init__()
        self.emb_mat_tensor = Variable(cuda.FloatTensor(emb_mat))
        self.embedding_dim = self.emb_mat_tensor.size(1)
        self.hidden_dim = cfg.LSTM_HIDDEN_SIZE
        self.vocab_size = self.emb_mat_tensor.size(0)
        self.tag_idx = tag_idx
        self.tagset_size = len(tag_idx)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).cuda()

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_idx[cfg.SENT_START], :] = -10000
        self.transitions.data[:, tag_idx[cfg.SENT_END]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(2, 1, self.hidden_dim)).cuda(),
                Variable(torch.randn(2, 1, self.hidden_dim)).cuda())

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_idx[cfg.SENT_START]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_idx[cfg.SENT_END]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = Variable(cuda.FloatTensor([0]))

        start_tag = Variable(torch.cuda.LongTensor([self.tag_idx[cfg.SENT_START]]))
        tags = torch.cat([start_tag, tags], dim=0)
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags.data[i + 1], tags.data[i]]
            score += feat[tags[i + 1]]
        score = score + self.transitions[self.tag_idx[cfg.SENT_END], tags.data[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = cuda.FloatTensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_idx[cfg.SENT_START]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG/SENT_END
        terminal_var = forward_var + self.transitions[self.tag_idx[cfg.SENT_END]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_idx[cfg.SENT_START]  # Sanity check
        best_path.reverse()
        return best_path

    @staticmethod
    def to_scalar(var):
        # returns a python float
        return var.view(-1).data.tolist()[0]

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return self.to_scalar(idx)

    @staticmethod
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        tensor = cuda.LongTensor(idxs)
        return Variable(tensor)

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def neg_log_likelihood(self, sentences, tags):
        total_loss = Variable(cuda.FloatTensor([0]))
        for sentence, tag in zip(sentences, tags):
            sentence = sentence[1:-1]
            tag = tag[1:-1]
            sent_var = Variable(cuda.LongTensor(sentence))
            tag_var = Variable(cuda.LongTensor(tag))

            feats = self._get_lstm_features(sent_var)
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tag_var)
            total_loss += forward_score - gold_score

        return total_loss

    def forward(self, sentences):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        tag_seq = []
        for sentence in sentences:
            sentence = sentence[1:-1]
            sent_var = Variable(cuda.LongTensor(sentence))
            lstm_feats = self._get_lstm_features(sent_var)

            # Find the best path, given the features.
            tag_seq.append([self.tag_idx[cfg.SENT_START]] +
                           self._viterbi_decode(lstm_feats) +
                           [self.tag_idx[cfg.SENT_END]])

        return tag_seq