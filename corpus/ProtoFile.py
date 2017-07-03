# TODO Do a code review
import os
from collections import namedtuple, Counter

import nltk
from nltk.tokenize import sent_tokenize
import config as cfg

import re
import io
import logging

from corpus.TextFile import TextFile

Tag = namedtuple("Tag", "tag_id, tag_name, tag_name_bio, start, end, word")
Link = namedtuple("Link", "l_id, l_name, arg1, arg2")


class ProtoFile(TextFile):
    def __init__(self, filename, bio_encoding=True):
        super().__init__(filename)
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.protocol_name = self.basename
        self.text_file = self.filename + '.txt'
        self.ann_file = self.filename + '.ann'

        with io.open(self.text_file, 'r', encoding='utf-8', newline='') as t_f, io.open(self.ann_file, 'r',
                                                                                        encoding='utf-8',
                                                                                        newline='') as a_f:
            self.text = t_f.readlines()
            print(self.text)
            self.full_text = "".join(self.text)
            self.ann = a_f.readlines()
            self.status = self.__pretest()
            self.links = []
        if self.status:
            self.tags = self.__parse_tags()
            self.unique_tags = set([tag.tag_name for tag in self.tags])

            self.sents = self.get_sents()
            self.nsents = len(self.sents)
            self.__std_index()
            self.__parse_links()
            self.tag_0_id = 'T0'
            self.tag_0_name = 'O'
            self.word_tag_per_sent, self.tokens, self.words, self.token_sents = self.populate_tokens(bio_encoding)

    def cnt_sent(self):
        if len(self.text) == 2:
            sents = sent_tokenize(self.text[1])
            return len(sents)
        else:
            return 0

    def cnt_words(self):
        if len(self.text) == 2:
            w = self.text[1].split()
            return len(w)

    def count_tags(self, headers):
        tag_counts = dict()
        if self.status:
            for tag in self.tags:
                if tag.tag_name in headers:
                    if tag.tag_name in tag_counts:
                        tag_counts[tag.tag_name] += 1
                    else:
                        tag_counts[tag.tag_name] = 1
            for link in self.links:
                if link.l_name in headers:
                    if link.l_name in tag_counts:
                        tag_counts[link.l_name] += 1
                    else:
                        tag_counts[link.l_name] = 1
        return tag_counts

    def count_span_len(self):
        s_len = dict()
        if self.status:
            for tag in self.tags:
                if tag.tag_name in s_len:
                    s_len[tag.tag_name] += len(tag.word.split())
                else:
                    s_len[tag.tag_name] = len(tag.word.split())
        return s_len

    def __std_index(self):
        def get_tag(e_id):
            for it, _line in enumerate(self.ann):
                if _line.find(e_id) == 0:
                    logging.info(_line.rstrip())
                    spl = _line.split()

                    return spl[1].split(':')[1]

        def replace_Es(string):
            if string[0] == 'E' or string[0] == 'R':
                sp_res = string.split()
                for sp in sp_res[1:]:

                    rel = sp.split(':')
                    if len(rel) == 2 and rel[1][0] == 'E':
                        logging.info(["Before:", string])
                        e_id = rel[1]

                        # e_id = E23
                        t_id = get_tag(e_id)

                        # t_id = T73

                        string = string.replace(e_id, t_id)
                        logging.info(["After:", string])
                        logging.info("\n")
            return string

        for i, line in enumerate(self.ann):
            self.ann[i] = replace_Es(line)

    def __pretest(self):
        """
        Returns false if annotation file or text file is empty
        :return:
        """
        if len(self.text) < 2:
            logging.debug(self.text)
            return False
        if len(self.ann) < 1:
            logging.debug(self.ann)
            return False
        return True

    def __parse_links(self):
        if self.links:
            logging.error("Already parsed, I am not parsing again")
            return
        for line in [t for t in self.ann if (t[0] == 'E' or t[0] == 'R')]:
            if line[0] == 'E':
                e = self.__parse_e(line)
                self.links.extend(e)
            elif line[0] == 'R':
                r = self.__parse_r(line)
                self.links.append(r)

    def __parse_e(self, e):
        links = []

        def get_tag(arg_id):

            l = [tag for tag in self.tags if tag.tag_id == arg_id]
            return l[0]

        temp = e.rstrip()
        temp = temp.split()
        e_id = temp[0]
        arg1_id = temp[1].split(':')[1]

        arg1_tag = get_tag(arg1_id)
        if temp[2:]:
            for rel in temp[2:]:
                r_name, arg2_id = rel.split(':')
                arg2_tag = get_tag(arg2_id)
                links.append(Link(e_id, r_name, arg1_tag, arg2_tag))

        return links

    def __parse_r(self, r):
        def get_tag(arg):
            tag_id = arg.split(':')[1]

            if tag_id[0] == 'T':
                l = [tag for tag in self.tags if tag.tag_id == tag_id]
            else:
                l = [_link.arg1 for _link in self.links if _link.l_id == tag_id]
            return l[0]

        temp = r.rstrip()
        temp = temp.split()
        r_id = temp[0]
        r_name = temp[1]
        arg1 = temp[2]
        arg2 = temp[3]
        arg1_tag = get_tag(arg1)
        arg2_tag = get_tag(arg2)
        link = Link(r_id, r_name, arg1_tag, arg2_tag)
        return link

    def __parse_tags(self):
        tags = []
        only_tags = [t for t in self.ann if t[0] == 'T']
        for tag in only_tags:
            tag = tag.rstrip()
            temp = tag.split('\t')

            if len(temp[1].split()) == 3:
                tag_name, start, end = temp[1].split()
            elif len(temp[1].split()) == 4:
                tag_name, start, _, end = temp[1].split()
            else:
                tag_name, start, _, _, end = temp[1].split()

            t = Tag(tag_id=temp[0], tag_name=tag_name, tag_name_bio='', start=int(start), end=int(end), word=temp[2])
            tags.append(t)
        return tags

    @staticmethod
    def __contain(s1, e1, s2, e2):
        if s2 <= s1 and e1 <= e2:
            return True
        elif not (s2 >= s1 and e2 >= e1 or s2 <= s1 and e2 <= e1):
            logging.debug("partial overlap: {0} {1} {2} {3}".format(s1, e1, s2, e2))
            return False
        return False

    def get_tag(self, word, start, end):
        # self.tag = [(tag_id, tag_name, start, end, word)] named tuple
        for tag in self.tags:

            if word in tag.word and self.__contain(start, end, tag.start, tag.end):
                if tag.start == start:
                    tag_name_bio = 'B-'
                else:
                    tag_name_bio = 'I-'
                return Tag(tag_id=tag.tag_id, tag_name=tag.tag_name, tag_name_bio=tag_name_bio, start=start, end=end,
                           word=word)

            elif self.__contain(start, end, tag.start, tag.end):
                logging.debug("odd result:{0}, {1}, {2}, {3}".format(word, start, end, tag))

        return Tag(tag_id=self.tag_0_id, tag_name=self.tag_0_name, tag_name_bio='', start=start, end=end, word=word)

    def extract_data_per_sent(self, with_bio=False):
        # extract a pair of list of list of words and tags
        # eg text:    This is a Sentence. This is also a sentence.
        # eg output:  (words, tags)
        # eg output:  ([[This, is, a, Sentence],
        #              [This, is, also, a, sentence]],
        #              [[Tag(), Tag(), Tag(), Tag()],
        #              [Tag(), Tag(), Tag(), Tag(), Tag()]])

        start = len(self.text[0])
        print(start)
        end = start
        # words = self.text[1].split()
        sents = self.sents
        sent_tags = []
        print(self.full_text)
        for sent in sents:
            print(sent)

            words = self._word_tokenizer(sent, to_lowercase=False)
            word_tag = []
            tags_name_only = []
            for word in words:

                start = self.full_text.find(word, end)
                end = start + len(word)
                tag = self.get_tag(word, start, end)
                print(tag)
                if with_bio:
                    tag_name = tag.tag_name_bio + tag.tag_name

                else:
                    tag_name = tag.tag_name

                word_tag.append((word, tag_name))

            sent_tags.append(word_tag)

        # split word_tag
        col_words = []
        col_tags = []
        for sent in sent_tags:
            if sent:
                words, tags = zip(*sent)
                col_words.append(words)
                col_tags.append(tags)

        return col_words, col_tags

    def extract_tags_per_sent(self):
        # extract a list of list of tags.
        # eg text:    This is a Sentence. This is also a sentence.
        # eg output:  [[Tag(), Tag(), Tag(), Tag()],
        #              [Tag(), Tag(), Tag(), Tag(), Tag()]]

        start = len(self.text[0])
        end = start
        # words = self.text[1].split()
        sents = self.sents
        sent_tags = []
        for sent in sents:
            words = self._word_tokenizer(sent, to_lowercase=False)
            word_tag = []
            tags_name_only = []
            for word in words:
                start = (self.text[0] + self.text[1]).find(word, end)
                end = start + len(word)

                tag = self.get_tag(word, start, end)
                word_tag.append((word, tag))
                tags_name_only.append(tag.tag_name)
            sent_tags.append(tags_name_only)
        return sent_tags

    def sent_spans(self):
        start = len(self.text[0])
        end = start
        # words = self.text[1].split()
        sents = self.sents
        res = []
        for sent in sents:
            ss = end
            words = self._word_tokenizer(sent)
            for word in words:
                start = (self.text[0] + self.text[1]).find(word, end)
                end = start + len(word)
            se = end
            res.append((ss, se))
        return res

    def extract_word_tags_per_sent(self):
        start = len(self.text[0])
        end = start
        # words = self.text[1].split()
        sents = self.sents
        sent_tags = []
        for sent in sents:
            words = self._word_tokenizer(sent, to_lowercase=True)
            word_tag = []

            for word in words:
                start = (self.text[0] + self.text[1]).find(word, end)
                end = start + len(word)

                tag = self.get_tag(word, start, end)

                word_tag.append((word, tag))
            sent_tags.append(word_tag)
        return sent_tags

    def extract_tags(self):
        start = len(self.text[0])
        end = start
        # words = self.text[1].split()
        words = self._word_tokenizer(self.text[1], to_lowercase=True)
        word_tag = []
        tags_name_only = []

        for word in words:
            start = (self.text[0] + self.text[1]).find(word, end)
            end = start + len(word)

            tag = self.get_tag(word, start, end)

            word_tag.append((word, tag))
            tags_name_only.append(tag.tag_name)

        # ("Add", Tag("Action-Verb", ...)) , "Action-Verb"
        return word_tag, tags_name_only

    # ###################interface from Article#############################################
    # TODO clean this up
    def populate_tokens(self, bio_encoding):
        word_tags, _ = self.extract_tags()
        word_tag_per_sent = self.extract_word_tags_per_sent()
        tokens = [Token(word, (tag.tag_name + tag.tag_name_bio if bio_encoding else tag.tag_name))
                  for word, tag in word_tags if tag.tag_name in cfg.LABELS + [cfg.NO_NE_LABEL]]
        sents = self.extract_word_tags_per_sent()

        token_sents = [[Token(word, (tag.tag_name_bio + tag.tag_name if bio_encoding else tag.tag_name))
                  for word, tag in sent if tag.tag_name in cfg.LABELS + [cfg.NO_NE_LABEL]] for sent in sents]

        words = ([word for word, tag in word_tags])

        return word_tag_per_sent, tokens, words, token_sents

    def get_content_as_string(self):
        """Returns the article's content as string.
        This is not neccessarily identical to the original text content, because multi-whitespaces
        are replaced by single whitespaces.

        Returns:
            string (article/document content).
        """
        return " ".join([token.word for token in self.tokens])

    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.

        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear. (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        if add_no_ne_label:
            counts = Counter([token.label for token in self.tokens])
        else:
            counts = Counter([token.label for token in self.tokens \
                              if token.label != cfg.NO_NE_LABEL])
        return counts.most_common()

    def count_labels(self, add_no_ne_label=False):
        """Returns how many named entity tokens appear in the article/document.

        Args:
            add_no_ne_label: Whether to also count unlabeled words. (Default is False.)
        Returns:
            Count of all named entity tokens (integer).
        """
        return sum([count[1] for count in self.get_label_counts(add_no_ne_label=add_no_ne_label)])


class Token(object):
    """Encapsulates a token/word.
    Members:
        token.original: The original token, i.e. the word _and_ the label, e.g. "John/PER".
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See Window.apply_features().)
    """

    def __init__(self, word, label=cfg.NO_NE_LABEL):
        """Initialize a new Token object.
        Args:
            original: The original word as found in the text document, including the label,
                e.g. "foo", "John/PER".
        """
        # self.original = original

        self.word = word
        self.label = label
        # self._word_ascii = None
        self.feature_values = None


if __name__ == '__main__':
    protofile = ProtoFile("simple_input/protocol_31")
    sents, labels = protofile.extract_data_per_sent(True)
