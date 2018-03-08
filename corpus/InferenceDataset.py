import itertools
from collections import namedtuple

import re
from nltk.tokenize.moses import MosesTokenizer
from torch.utils import data

Data = namedtuple("Data", ['SENT', 'X', 'C'])


class InferenceDataset(data.Dataset):
    def __init__(self, p_txt, word_index, char_index, is_oov, sent_start, sent_end, word_start, word_end, unk):
        self.original_sents = self.tokenize(p_txt)
        self.sents = self.tokenize(p_txt, to_lower=True)
        self.char_index = char_index
        self.word_index = word_index
        # self.pos_index = pos_ids
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.word_start = word_start
        self.word_end = word_end
        # self.tag_idx = tag_idx
        self.is_oov = is_oov
        self.unk_token = unk
        self.words = list(itertools.chain.from_iterable([[word.lower() for word in sent] for sent in self.sents]))
        self.vocab = set(self.words)
        self.arg = self.arg_sort(self.sents)

    @staticmethod
    def tokenize(txt, to_lower=False):
        assert isinstance(txt, str)
        tokenizer = MosesTokenizer()
        lines = txt.split('\n')
        t = [tokenizer.tokenize(line) for line in lines]
        if to_lower:
            return [[word.lower() for word in line] for line in t]
        else:
            return t

    @staticmethod
    def arg_sort(l):
        r = list(range(len(l)))
        r.sort(key=lambda i: len(l[i]), reverse=True)
        return r

    def undo_sort(self, sents):
        original = [None] * len(sents)
        for i, sent in zip(self.arg, sents):
            original[i] = sent

        return original

    def __getitem__(self, item):

        x = self.__gen_sent_idx_seq(self.sents[item])
        c = self.__prep_char_idx_seq(self.sents[item])
        # f_pos = f['0:pos'].as_matrix()

        # add pos tag for start and end tag
        # f_pos = np.insert(f_pos, 0, self.pos_index['NULL'])
        # f_pos = np.insert(f_pos, f_pos.size, self.pos_index['NULL'])
        # f_pos = list(f_pos.tolist())

        # assert len(x) == len(f) + 2, (len(x), len(f), pno)
        # assert len(x) == len(f_pos)
        return Data(self.original_sents[item], x, c)

    def __len__(self):
        return len(self.sents)

    def __gen_sent_idx_seq(self, sent):
        sent_idx_seq = self.__to_idx_seq(sent, start=self.sent_start, end=self.sent_end,
                                         index=self.word_index, oov=self.is_oov)

        return sent_idx_seq

    def __to_idx_seq(self, list1d, start, end, index, oov=None):
        row_idx_seq = [index[start]]

        for item in list1d:
            item = re.sub(r'\d', '0', item)
            if item not in index or (oov and oov[index[item]] == 1):
                row_idx_seq.append(index[self.unk_token])
            else:
                row_idx_seq.append(index[item])

        row_idx_seq.append(index[end])

        return row_idx_seq

    def __prep_char_idx_seq(self, sent):
        char_idx_seq = [self.__to_idx_seq([self.sent_start], start=self.word_start, end=self.word_end,
                                          index=self.char_index)] + \
                       [self.__to_idx_seq(list(word), start=self.word_start, end=self.word_end,
                                          index=self.char_index)
                        for word in sent] + \
                       [self.__to_idx_seq([self.sent_end], start=self.word_start, end=self.word_end,
                                          index=self.char_index)]

        return char_idx_seq
