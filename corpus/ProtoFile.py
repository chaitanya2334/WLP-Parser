import os
import pickle
from collections import namedtuple, Counter

import nltk
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tokenize.moses import MosesTokenizer
from tqdm import tqdm

import config as cfg
import features_config as feat_cfg

import io
import logging
from builtins import any as b_any
import re

import itertools as it
from preprocessing.feature_engineering.pos import PosTagger
import html

logger = logging.getLogger(__name__)

Tag = namedtuple("Tag", "tag_id, tag_name, start, end, words")
Link = namedtuple("Link", "l_id, l_name, arg1, arg2")


# its a good idea to keep a datastructure like
# list of sentences, where each sentence is a list of words : [[word1, word2, word3,...], [word1, word2...]]

class ProtoFile:
    def __init__(self, filename, genia, gen_features, lowercase, replace_digits, to_filter):
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.protocol_name = self.basename
        self.text_file = self.filename + '.txt'
        self.ann_file = self.filename + '.ann'

        with io.open(self.text_file, 'r', encoding='utf-8', newline='') as t_f, io.open(self.ann_file, 'r',
                                                                                        encoding='utf-8',
                                                                                        newline='') as a_f:
            self.tokenizer = MosesTokenizer()
            self.lines = []
            for line in t_f.readlines():
                self.lines.append(html.unescape(line))

            self.text = "".join(self.lines)  # full text
            self.ann = a_f.readlines()
            self.status = self.__pretest()
            self.links = []

        if self.status:
            sents = [self.tokenizer.tokenize(line) for line in self.lines]  # generate list of list of words
            self.heading = sents[0]
            self.sents = sents[1:]
            self.tags = self.__parse_tags()
            self.unique_tags = set([tag.tag_name for tag in self.tags])
            self.__std_index()
            self.__parse_links()
            self.tag_0_id = 'T0'
            self.tag_0_name = 'O'
            self.tokens2d = self.gen_tokens(labels_allowed=cfg.LABELS, lowercase=lowercase,
                                            replace_digits=replace_digits)
            self.tokens2d = [[self.clean_html_tag(token) for token in token1d] for token1d in self.tokens2d]

            self.word_cnt = sum(len(tokens1d) for tokens1d in self.tokens2d)
            self.f_df = None
            if gen_features:
                if genia:
                    self.pos_tags = self.__gen_pos_genia(genia)
                else:
                    self.pos_tags = self.__gen_pos_stanford()

                self.conll_deps = self.__gen_dep()
                self.parse_trees = self.__gen_parse_trees()

            if to_filter:
                self.filter()

            self.relations = self.gen_relations()

    @staticmethod
    def clean_html_tag(token):
        token.word = html.unescape(token.word)
        return token

    def filter(self):
        # tokens2d, pos_tags, and conll_deps are filtered if a sentence was not tagged
        new_tokens2d = []
        new_pos_tags = []
        new_conll_deps = []
        for tokens1d, pos_tag1d, deps1d in zip(self.tokens2d, self.pos_tags, self.conll_deps):
            # here tokens1d is a sentence of word sequences, and label is a sequence of labels for a sentence.

            # check if any of the labels in this sentence have POSITIVE_LABEL in them, if they do, then consider that
            # sentence, else discard that sentence.
            if b_any(cfg.POSITIVE_LABEL in token.label for token in tokens1d):
                new_tokens2d.append(tokens1d)
                new_pos_tags.append(pos_tag1d)
                new_conll_deps.append(deps1d)

        self.tokens2d = new_tokens2d
        self.pos_tags = new_pos_tags
        self.conll_deps = new_conll_deps

    def get_deps(self):
        return [nltk.DependencyGraph(conll_dep, top_relation_label='root') for conll_dep in self.conll_deps]

    def __gen_parse_trees(self):
        p_cache = os.path.join(cfg.PARSE_PICKLE_DIR, self.protocol_name + '.p')
        try:
            parse_trees = pickle.load(open(p_cache, 'rb'))

        except(pickle.UnpicklingError, EOFError, FileNotFoundError):

            parser = StanfordParser(path_to_jar=feat_cfg.STANFORD_PARSER_JAR,
                                    path_to_models_jar=feat_cfg.STANFORD_PARSER_MODEL_JAR,
                                    java_options="-mx3000m")
            temp_trees = list(parser.raw_parse_sents(self.lines[1:]))
            parse_trees = [next(trees) for trees in temp_trees]
            os.makedirs(os.path.dirname(p_cache), exist_ok=True)
            pickle.dump(parse_trees, open(p_cache, 'wb'))

        return parse_trees

    def __gen_dep(self):
        d_cache = os.path.join(cfg.DEP_PICKLE_DIR, self.protocol_name + '.p')
        try:
            # loading saved dep parsers
            conll_deps = pickle.load(open(d_cache, 'rb'))

        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            dep = StanfordDependencyParser(path_to_jar=feat_cfg.STANFORD_PARSER_JAR,
                                           path_to_models_jar=feat_cfg.STANFORD_PARSER_MODEL_JAR,
                                           java_options="-mx3000m")

            # adding pos data to dep parser speeds up dep generation even further
            dep_graphs = [sent_dep for sent_dep in dep.tagged_parse_sents(self.pos_tags)]

            # save dependency graph in conll format
            conll_deps = [next(deps).to_conll(10) for deps in dep_graphs]
            os.makedirs(os.path.dirname(d_cache), exist_ok=True)
            pickle.dump(conll_deps, open(d_cache, 'wb'))

        return conll_deps

    def __gen_pos_genia(self, pos_tagger):
        p_cache = os.path.join(cfg.POS_GENIA_DIR, self.protocol_name + '.p')
        try:
            pos_tags = pickle.load(open(p_cache, 'rb'))
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pos_tags = pos_tagger.parse_through_file([" ".join(sent) for sent in self.sents])
            os.makedirs(os.path.dirname(p_cache), exist_ok=True)
            pickle.dump(pos_tags, open(p_cache, 'wb'))
        return pos_tags

    def __gen_pos_stanford(self):
        pos = PosTagger(feat_cfg.STANFORD_POS_JAR_FILEPATH, feat_cfg.STANFORD_MODEL_FILEPATH,
                        cache_filepath=None)
        p_cache = os.path.join(cfg.POS_PICKLE_DIR, self.protocol_name + '.p')
        try:
            pos_tags = pickle.load(open(p_cache, 'rb'))
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pos_tags = pos.tag_sents(self.sents)
            # for some reason stanford parser deletes words that are just underscores, and
            # dependency parser cannot deal with an empty text in pos tagger, so the below hack.
            pos_tags = [[pos_tag if pos_tag[0] else ('_', pos_tag[1]) for pos_tag in p1d] for p1d in pos_tags]
            pickle.dump(pos_tags, open(p_cache, 'wb'))

        return pos_tags

    def cnt_words(self):
        if self.status:
            w = sum([len(sent) for sent in self.sents[1:]])
            return w

            # generic counter of entities, supported by a function callback fn that depends on tag's properties

    # ent_counter returns a dict = {'ENTITY1' : summation for all tags of 'ENTITY1'(fn(tag))}
    def __ent_counter(self, ent_types, fn):
        tag_cnts = dict()
        if self.status:
            for tag in self.tags:
                if tag.tag_name in ent_types:
                    if tag.tag_name in tag_cnts:
                        tag_cnts[tag.tag_name] += fn(tag)
                    else:
                        tag_cnts[tag.tag_name] = fn(tag)

        return tag_cnts

    def ent_cnt(self, ent_types):
        tag_cnts = self.__ent_counter(ent_types, lambda x: 1)

        return tag_cnts

    def ent_w_cnt(self, ent_types):
        def cnt_words(tag):
            string = tag.word
            words = nltk.word_tokenize(string)
            return len(words)

        tag_cnts = self.__ent_counter(ent_types, cnt_words)

        return tag_cnts

    # calculates the total no of chars (including spaces) for each entity in a protocol file
    def ent_span_len(self, ent_types):
        tag_cnts = self.__ent_counter(ent_types, lambda x: len(x.word))

        return tag_cnts

    def __std_index(self):
        # modifies the ann text such that all
        # Exx Action:Txx Using:Exx convert to
        # Exx Action:Txx Using:Tyy
        # so that they are easier to resolve later
        # given that Txx can be independently resolved,
        # whereas Exx sometimes have forward and backward dependencies
        def search_tag(e_id):
            if e_id[0] == 'E':
                for _line in self.ann:
                    if _line.find(e_id) == 0:
                        logging.info(_line.rstrip())
                        spl = _line.split()
                        return spl[1].split(':')[1]
            else:
                return e_id

        def replace_Es(string):
            if string[0] == 'E':
                sp_res = string.split()
                front_half = sp_res[0]
                args = [tuple(sp.split(':')) for sp in sp_res[1:]]
            elif string[0] == 'R':
                sp_res = string.split()
                r_id = sp_res[0]
                r_name = sp_res[1]
                front_half = " ".join([r_id, r_name])
                args = [tuple(sp.split(':')) for sp in sp_res[2:]]
            else:
                # nothing to replace
                return string

            # args = [(Action, Txx), (Using, Exx)]
            replaced_args = [(rel_name, search_tag(tid)) for rel_name, tid in args]
            # replaced_args = [(Action, Txx), (Using, Txx)]
            args_str = " ".join([":".join(item) for item in replaced_args])
            # args_str = "Action:Txx Using:Txx"

            string = " ".join([front_half, args_str])
            # string = "Exx Action:Txx Using:Txx"

            return string

        for i, line in enumerate(self.ann):
            self.ann[i] = replace_Es(line)

    def __pretest(self):
        """
        Returns false if annotation file or text file is empty
        :return:
        """
        if len(self.lines) < 2:
            logger.debug(self.sents)
            return False
        if len(self.ann) < 1:
            logger.debug(self.ann)
            return False
        return True

    def __parse_links(self):
        if self.links:
            logger.error("Already parsed, I am not parsing again")
            return
        for line in [t for t in self.ann if (t[0] == 'E' or t[0] == 'R')]:
            if line[0] == 'E':
                e = self.__parse_e(line)
                self.links.extend(e)
            elif line[0] == 'R':
                r = self.__parse_r(line)
                self.links.append(r)

    def get_tag_by_id(self, tid):
        if tid[0] == 'T':
            ret = [tag for tag in self.tags if tag.tag_id == tid]
        else:
            ret = [link.arg1 for link in self.links if link.l_id == tid]
        return ret[0]

    # TODO test
    def get_wb(self, tag1, tag2):
        # we want tag1 to appear before tag2
        assert isinstance(tag1, Tag)
        assert isinstance(tag2, Tag)
        if tag1.end > tag2.start:
            tag1, tag2 = tag2, tag1

        return self.text[tag1.end:tag2.start]

    def get_tb(self, tag1, tag2):
        assert isinstance(tag1, Tag)
        assert isinstance(tag2, Tag)
        if tag1.end > tag2.start:
            tag1, tag2 = tag2, tag1

        idx = tag1.start
        while idx < tag2.end:
            tag = self.get_tag_by_start(idx)

            idx = tag.end

    # TODO test
    def surr_words(self, tag, n):
        fore_words = self.text[tag.end:].split()[:2]
        back_words = self.text[:tag.start].split()[:-2]
        return [fore_words + back_words]

    # TODO test
    def __parse_e(self, e):
        links = []
        temp = e.rstrip()
        temp = temp.split()
        e_id = temp[0]
        arg1_id = temp[1].split(':')[1]

        arg1_tag = self.get_tag_by_id(arg1_id)
        if temp[2:]:
            for rel in temp[2:]:
                r_name, arg2_id = rel.split(':')
                arg2_tag = self.get_tag_by_id(arg2_id)
                links.append(Link(e_id, r_name, arg1_tag, arg2_tag))

        return links

    # TODO test
    def __parse_r(self, r):
        r_id, r_name, arg1, arg2 = r.rstrip().split()
        arg1_id = arg1.split(':')[1]
        arg2_id = arg2.split(':')[1]
        arg1_tag = self.get_tag_by_id(arg1_id)
        arg2_tag = self.get_tag_by_id(arg2_id)
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

            t = Tag(tag_id=temp[0],
                    tag_name=tag_name,
                    start=int(start),
                    end=int(end),
                    words=self.tokenizer.tokenize(temp[2]))

            tags.append(t)
        return tags

    @staticmethod
    def __contain(s1, e1, s2, e2):
        if s2 <= s1 and e1 <= e2:
            return True
        elif not (s2 >= s1 and e2 >= e1 or s2 <= s1 and e2 <= e1):
            logger.debug("partial overlap: {0} {1} {2} {3}".format(s1, e1, s2, e2))
            return False
        return False

    @staticmethod
    def make_bio(tag):
        # returns [(word, label), (word, label)]
        # where the label is encoded with B, I, or O based on its position in the tag
        # tag = Tag(tag_id, tag_name, start, end, words)

        labels = ['B-' + tag.tag_name]
        labels += ['I-' + tag.tag_name for _ in tag.words[1:]]
        return list(zip(tag.words, labels))

    def get_tag_by_start(self, start):
        for tag in self.tags:
            if tag.start == start:
                return tag

        logging.debug("Protocol={0}: No tag found with start == {1}".format(self.protocol_name, start))
        return None

    def gen_tokens(self, labels_allowed=None, lowercase=False, replace_digits=False):
        # for a list of list of words returns a list of list of tokens
        # [[Token(word, label), Token(word, label)], [Token(word, label), Token(word, label)]]
        # BIO encoding

        start = len(self.sents[0])
        ret = []
        for sent in self.sents:
            word_label_pairs = []
            wi = 0
            sent = [html.unescape(word) for word in sent]

            while wi < len(sent):
                word = sent[wi]
                start = self.text.find(word, start)

                tag = self.get_tag_by_start(start)
                # we dont want tags that are not allowed
                if labels_allowed:
                    if tag and tag.tag_name not in labels_allowed:
                        tag = None

                if tag:
                    # tag was found
                    # make bio returns a list of (word, label) pairs which we extend to the word_label_pairs list.
                    logging.debug("Protocol={0}: Tag was found = {1}".format(self.protocol_name, tag))
                    word_label_pairs.extend(self.make_bio(tag))
                    start = tag.end
                    wi += len(tag.words)

                if not tag:
                    # its likely that there is no tag for this word
                    logging.debug("Protocol={0}: Tag was not found at word = {1}".format(self.protocol_name, word))
                    word_label_pairs.append((word, 'O'))
                    start += len(word)
                    wi += 1

            tokens = [Token(html.unescape(word), label, lowercase=lowercase, replace_digits=replace_digits) for
                      word, label in
                      word_label_pairs]

            ret.append(tokens)

        return ret

    def get_token_idx(self, tag):
        def get_sentence_by_tag(t, lines, p):
            # find the sentence number based on tag.
            s = t.start
            e = t.end
            assert s < e, p.basename

            sent_start = len(lines[0])
            sent_end = sent_start
            for i, sent in enumerate(lines[1:]):
                sent_end += len(sent)
                if sent_start <= s and e <= sent_end:
                    return i

                sent_start = sent_end

            return None

        sent_idx = get_sentence_by_tag(tag, self.lines, self)

        assert sent_idx is not None

        tokens1d = self.tokens2d[sent_idx]
        offset = sum([len(sent) for sent in self.lines[:sent_idx + 1]])
        start = offset
        s_idx = 0

        while start < tag.start:
            # did not find s_idx, increment start to the next token's start
            start = offset + html.unescape(self.lines[sent_idx + 1]).find(tokens1d[s_idx + 1].word,
                                                                          start - offset + len(tokens1d[s_idx].word))
            s_idx += 1

        end = start
        e_idx = s_idx
        while end < tag.end:
            if e_idx + 1 == len(tokens1d):
                e_idx += 1
                break
            end = offset + html.unescape(self.lines[sent_idx + 1]).find(tokens1d[e_idx + 1].word,
                                                                        end - offset + len(tokens1d[e_idx].word))
            e_idx += 1

        return sent_idx, (s_idx, e_idx)

    def gen_relations(self):
        r_cache = os.path.join(cfg.REL_PICKLE_DIR, self.protocol_name + '.p')
        try:
            relations = pickle.load(open(r_cache, 'rb'))
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            relations = self.__gen_relations()
            os.makedirs(os.path.dirname(r_cache), exist_ok=True)
            pickle.dump(relations, open(r_cache, 'wb'))
        return relations

    # based on the assumption that a link is always between two arguments, both being in the same sentence.
    def __gen_relations(self):
        ret = []
        arg_perm = list(it.permutations(self.tags, 2))

        for link in self.links:
            # remove all the links that do exist from this list
            arg_perm = [(_arg1, _arg2) for _arg1, _arg2 in arg_perm
                        if _arg1.words != link.arg1.words or _arg2.words != link.arg2.words]

            # assumption that both args are in the same sentence
            sent_idx1, arg1 = self.get_token_idx(link.arg1)
            sent_idx2, arg2 = self.get_token_idx(link.arg2)

            if sent_idx1 == sent_idx2:  # verify that both arg1 and arg2 have the same sent idx

                token_arg1 = [token.original for token in self.tokens2d[sent_idx1][arg1[0]:arg1[1]]]
                token_arg2 = [token.original for token in self.tokens2d[sent_idx2][arg2[0]:arg2[1]]]

                # TODO fix all these errors
                if self.basename == "protocol_371":
                    if sent_idx1 == 18 and token_arg1[-1] == "tubes":
                        token_arg1[-1] = "tube"

                    if sent_idx1 == 18 and token_arg2[-1] == "tubes":
                        token_arg2[-1] = "tube"

                if self.basename == "protocol_577":
                    if sent_idx1 == 53 and token_arg1[-1] == "RPE":
                        token_arg1[-1] = "AW2"
                    if sent_idx1 == 53 and token_arg2[-1] == "RPE":
                        token_arg2[-1] = "AW2"

                # protocol 600 gave problem too
                # (many protocols like this that have text in Tag different from text in self.sents fail too.)

                # assert [html.unescape(word) for word in link.arg1.words] == token_arg1, (link.arg1.words, token_arg1)
                # assert [html.unescape(word) for word in link.arg2.words] == token_arg2, (link.arg2.words, token_arg2)

                ret.append(Relation(self, link.l_name, sent_idx1, self.parse_trees[sent_idx1], arg1, arg2, link.arg1, link.arg2))

        for arg1, arg2 in tqdm(arg_perm, desc="arg_perm " + self.protocol_name):
            sent_idx1, arg1_idx = self.get_token_idx(arg1)
            sent_idx2, arg2_idx = self.get_token_idx(arg2)
            if sent_idx1 == sent_idx2:
                ret.append(Relation(self, 'O', sent_idx1, self.parse_trees[sent_idx1], arg1_idx, arg2_idx, arg1, arg2))

        return ret

    # ###################interface from Article#############################################
    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.

        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear. (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        if add_no_ne_label:
            counts = Counter([token.label for token in self.tokens2d])
        else:
            counts = Counter([token.label for token in self.tokens2d \
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


class Relation(object):
    def __init__(self, protocol, l_name, sent_idx, sent_parse_tree, arg1, arg2, arg1_tag, arg2_tag):

        """

        :param protocol: reference to the parent protocol object
        :param l_name: label of the relation
        :param sent_idx: The sentence index. The first index on the tokens2d data structure in the parent protocol object.
        :param arg1: a tuple (start_idx, end_idx) the start and end index on the sentence where arg1 is
        :param arg2: a tuple (start_idx, end_idx) the start and end index on the sentence where arg2 is
        """

        self.sent_idx = sent_idx
        self.arg1 = arg1
        self.arg2 = arg2
        self.p = protocol
        self.label = l_name
        self.arg1_tag = arg1_tag
        self.arg2_tag = arg2_tag
        self.parse_tree = sent_parse_tree

        self.feature_values = None

        # type checks
        assert isinstance(self.arg1, tuple)
        assert isinstance(self.arg2, tuple)
        assert isinstance(self.label, str)
        assert isinstance(self.sent_idx, int)
        assert isinstance(self.p, ProtoFile)

    def sameNP(self):
        c_type = self.__is_same_chunk()
        return c_type == "NP"

    def sameVP(self):
        c_type = self.__is_same_chunk()
        return c_type == "VP"

    def samePP(self):
        c_type = self.__is_same_chunk()
        return c_type == "PP"

    def arg1_deps(self):
        return self.__arg_deps(self.arg1)

    def arg2_deps(self):
        return self.__arg_deps(self.arg2)

    def get_arg1_tokens(self):
        return self.__get_tokens(self.arg1)

    def get_arg2_tokens(self):
        return self.__get_tokens(self.arg2)

    def is_1_before_2(self):
        return self.arg1[1] < self.arg2[0]

    def get_tokens_bet(self):
        return self.__get_bet(self.p.tokens2d)

    def get_b_tokens(self, no):
        return self.__get_b(self.p.tokens2d, no)

    def get_a_tokens(self, no):
        return self.__get_a(self.p.tokens2d, no)

    def get_bet_chunks(self):
        pos = self.__get_bet(self.p.pos_tags)
        return [p[2] for p in pos]

    def get_b_chunks(self, no):
        pos = self.__get_b(self.p.pos_tags, no)
        return [p[2] for p in pos]

    def get_a_chunks(self, no):
        pos = self.__get_a(self.p.pos_tags, no)
        return [p[2] for p in pos]

    def __get_bet(self, list2d):
        if self.is_1_before_2():
            return list2d[self.sent_idx][self.arg1[1]:self.arg2[0]]
        else:
            return list2d[self.sent_idx][self.arg2[1]:self.arg1[0]]

    def __get_b(self, list2d, no):
        if self.is_1_before_2():
            return list2d[self.sent_idx][self.arg1[0] - no:self.arg1[0]]
        else:
            return list2d[self.sent_idx][self.arg2[0] - no:self.arg2[0]]

    def __get_a(self, list2d, no):
        if self.is_1_before_2():
            return list2d[self.sent_idx][self.arg2[1]:self.arg2[1] + no]
        else:
            return list2d[self.sent_idx][self.arg1[1]:self.arg1[1] + no]

    def __arg_deps(self, arg):
        def get_deps(triples, word):
            dep = (0, 0)
            for g, r, d in triples:
                if g[0] == word:
                    dep = d
                    return dep

            return dep

        dep_graph = self.p.get_deps()[self.sent_idx]
        t = dep_graph.triples()
        tokens = self.__get_tokens(arg)
        deps = [get_deps(t, token.word) for token in tokens]

        return deps

    def __get_tokens(self, arg):
        # TODO undo hack
        if arg[0] == arg[1]:
            arg = (arg[0], arg[1] + 1)

        return self.p.tokens2d[self.sent_idx][arg[0]:arg[1]]

    def __get_all(self, list2d):
        if self.is_1_before_2():
            return list2d[self.sent_idx][self.arg1[0]:self.arg2[1]]
        else:
            return list2d[self.sent_idx][self.arg2[0]:self.arg1[1]]

    def __is_same_chunk(self):
        # checks if arg1 and arg2 are in the same chunk. if so, it will return the chunk type.
        if self.is_1_before_2():
            pos = self.p.pos_tags[self.sent_idx][self.arg1[0]:self.arg2[1]]
        else:
            pos = self.p.pos_tags[self.sent_idx][self.arg2[0]:self.arg1[1]]

        if not pos:
            return False

        start_chunk = pos[0][2]
        if start_chunk != 'O':
            start_chunk = start_chunk[2:]
        for p in pos:
            if p[2][0] == 'O' or p[2][0] == 'B':
                return False

        return start_chunk




class Token(object):
    """Encapsulates a token/word.
    Members:
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See EntityWindow.apply_features().)
    """

    def __init__(self, word, label=cfg.NO_NE_LABEL, lowercase=False, replace_digits=False):
        """Initialize a new Token object.
        Args:
            original: The original word as found in the text document, including the label,
                e.g. "foo", "John/PER".
        """
        # self.original = original

        self.word = word

        if lowercase:
            self.word = self.word.lower()

        if replace_digits:
            self.word = re.sub(r'\d', '0', self.word)

        self.label = label
        self.original = word
        # self._word_ascii = None
        self.feature_values = None


if __name__ == '__main__':
    pro = ProtoFile("./simple_input/protocol_235")
    [print([(token.word, token.label) for token in tokens1d]) for tokens1d in pro.tokens2d]
