from unittest import TestCase
from corpus.Manager import Manager
from preprocessing.text_processing import prepare_embeddings
import config as cfg


class TestManager(TestCase):
    def test_gen_train(self):
        self.fail()
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print(word_index['<s>'])
        self.corpus = Manager(word_index, char_index)
        self.corpus.set_per((1, 1, 98))
        train = self.corpus.gen_train()
        c = 0
        for sample in train:
            print(self.corpus.to_words(sample.X), sample.F[0])
            c += 1
        self.assertEqual(c, int(0.01 * self.corpus.size(True)))

    def test_gen_dev(self):
        self.fail()
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print(word_index['<s>'])
        self.corpus = Manager(word_index, char_index)
        self.corpus.set_per((1, 1, 98))
        dev = self.corpus.gen_dev()
        c = 0
        for sample in dev:
            print(self.corpus.to_words(sample.X), sample.P, sample.F[0])
            c += 1
        self.assertEqual(c, int(0.01 * self.corpus.size(True)))

    def test_window_empty(self):
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print(word_index['<s>'])
        self.corpus = Manager(word_index, char_index)
        self.corpus.set_per((60, 20, 20))

        train = self.corpus.split_data(2933, 2944, replace_num=True, to_filter=True)
        c = 0
        for sample in train:
            print(self.corpus.to_words(sample.X), sample.F[0])
            c += 1
        self.assertEqual(c, int(0.01 * self.corpus.size(True)))

    def test_gen_test(self):
        self.fail()
