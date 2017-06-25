from enum import Enum

import logging

import nltk
from nltk import sent_tokenize
import re


class TextFile(object):
    class Status(Enum):

        # if text file is composed of each sentence on each line
        SENT = 1

        # if text file is composed of a heading on the first line, and the full text in the next line.
        FULL = 2

        # if text file is empty (only heading or no heading)
        EMPTY = 3

    def __init__(self, filename):

        self.txt_fname = filename + '.txt'
        with open(self.txt_fname, 'r', encoding='utf-8') as t_f:
            self.text = t_f.readlines()
            self.type = self.get_text_type()

    def get_tokenized_sents(self, to_lowercase):
        sents = self.get_sents()
        if not sents:
            yield None
        else:
            for sent in sents:
                yield self._word_tokenizer(sent, to_lowercase)

    def get_sents(self):
        if self.type == self.Status.FULL:
            return sent_tokenize(self.text[1])
        if self.type == self.Status.SENT:
            return self.text[1:]
        return None

    def get_text_type(self):
        if len(self.text) < 2:
            logging.debug('{0} file is empty'.format(self.txt_fname))
            return self.Status.EMPTY
        if len(self.text) == 2:
            logging.debug('{0} file does not split sentences'.format(self.txt_fname))
            return self.Status.FULL
        if len(self.text) > 2:
            logging.debug('{0} file splits sentences'.format(self.txt_fname))
            return self.Status.SENT

    @staticmethod
    def _word_tokenizer(sent, to_lowercase):

        words = nltk.word_tokenize(sent)
        ret_words = []
        for word in words:
            if re.match('[.,/#!$%^&*;:{}=\-_`~()]', word) is None:
                if to_lowercase:
                    ret_words.append(word.lower())
                else:
                    ret_words.append(word)

        return ret_words
