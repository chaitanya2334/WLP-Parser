import glob
from shutil import copyfile, copy2

from conlleval import start_of_chunk, end_of_chunk
import os


class BratFile(object):
    def __init__(self, path, name):
        self.pno = name
        self.txt_fname = os.path.join(path, name + ".txt")
        self.ann_fname = os.path.join(path, name + ".ann")

        self.__clear_files()
        self.__start = 0
        self.__end = 0
        self.__tag_id = 0

    def __clear_files(self):
        open(self.txt_fname, "w").close()
        open(self.ann_fname, "w").close()

    def writer(self, words, labels, pno, ignore_label):
        with open(self.txt_fname, 'a', encoding='utf-8') as txt, \
                open(self.ann_fname, 'a', encoding='utf-8') as ann:
            # write the text file
            txt.write(" ".join(words) + " [" + pno + ']\n')
            sent_pad = len(" [" + pno + ']')

            # write the ann file
            self.__write_ann(ann, words, labels,
                             ignore_label,
                             sent_pad)

    def __write_ann(self, ann, words, labels, ignore_label, sent_pad):
        word_span = []
        start = self.__start
        end = self.__end
        tag_id = self.__tag_id

        for x in range(len(words)):
            prev_tag, prev_label = self.__split_tag_label(labels[x - 1] if x - 1 >= 0 else ignore_label)
            tag, label = self.__split_tag_label(labels[x])
            next_tag, next_label = self.__split_tag_label(labels[x + 1] if x + 1 < len(labels) else ignore_label)

            if start_of_chunk(prev_tag, tag, prev_label, label):
                word_span = []
                start = end
            word_span.append(words[x])
            end = end + len(words[x])

            if end_of_chunk(tag, next_tag, label, next_label):

                if label != ignore_label:
                    ann.write('T' + str(tag_id) + '\t' +
                              label + ' ' + str(start) + ' ' + str(end) + '\t' +
                              " ".join(word_span) + '\n')

                tag_id += 1
                start = end + 1
                word_span = []

            end += 1
        end = end + sent_pad

        self.__start = start
        self.__end = end
        self.__tag_id = tag_id

    @staticmethod
    def __split_tag_label(label):
        if len(label) < 2:
            return label, label

        return label[:1], label[2:]

    @staticmethod
    def __partition(a_list, lengths):
        i = 0
        ret = []
        for le in lengths:
            ret.append(a_list[i:i + le])
            i = i + le
        return ret


class Writer(object):
    def __init__(self, conf_path, save_path, name, label2id):
        # public

        self.true_path = os.path.join(save_path, name, "true")
        self.pred_path = os.path.join(save_path, name, "pred")

        self.check_dir(self.true_path)
        self.check_dir(self.pred_path)

        self.__clear_dir(self.true_path)
        self.__clear_dir(self.pred_path)

        self.gen_conf(conf_path, self.true_path)
        self.gen_conf(conf_path, self.pred_path)

        self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}

        self.true_brat_files = dict()
        self.pred_brat_files = dict()

        # private
        self.__start_true = 0
        self.__start_pred = 0
        self.__end_true = 0
        self.__end_pred = 0
        self.__tag_id_true = 0
        self.__tag_id_pred = 0

    #  ############################################# PUBLIC METHODS ####################################################

    @staticmethod
    def gen_conf(conf_path, save_path):
        conf_files = glob.glob(os.path.join(conf_path, "*.conf"))
        for conf_file in conf_files:
            copy2(conf_file, save_path)

    @staticmethod
    def check_dir(_dir):
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    @staticmethod
    def __clear_dir(_dir):
        txts = glob.glob(os.path.join(_dir, "*.txt"))
        anns = glob.glob(os.path.join(_dir, "*.ann"))

        if txts and not anns:
            # if anns is empty but there are .txt files then you are likely deleting stuff in the wrong dir
            raise RuntimeError(
                "dude.. i was about to delete stuff in the wrong file (probably). check dir: {0}".format(_dir))

        for txt in txts:
            open(txt, "w").close()

        for ann in anns:
            open(ann, "w").close()

    def from_file(self, path):
        pass

    # appends a line in txt and also tags words in that line.
    # start and end are relative positions with respect to this line and not the whole file.
    def writer(self, words, true_labels, pred_labels, pno, ignore_label):
        if pno not in self.true_brat_files:
            self.true_brat_files[pno] = BratFile(self.true_path, pno)

        if pno not in self.pred_brat_files:
            self.pred_brat_files[pno] = BratFile(self.pred_path, pno)

        self.true_brat_files[pno].writer(words, true_labels, pno, ignore_label)
        self.pred_brat_files[pno].writer(words, pred_labels, pno, ignore_label)

    def from_labels(self, sents, true, pred, ignore_label="O", doFull=True):
        # sents = [(sent, p), ... ]
        # words = self.__partition(words, sent_counts)
        # true = self.__partition(true, sent_counts)
        # pred = self.__partition(pred, sent_counts)
        true = self.convert_2_text(true)
        pred = self.convert_2_text(pred)

        for sent, true_labels, pred_labels in zip(sents, true, pred):
            sent_words, pno = sent
            assert len(sent_words) == len(true_labels) == len(pred_labels), \
                "Sentence data not of equal length. " \
                "({0})sent_words={1}, ({2})sent_true={3}, ({4})sent_pred={5}".format(len(sent_words), sent_words,
                                                                                     len(true_labels), true_labels,
                                                                                     len(pred_labels), pred_labels)

            if not doFull:
                if pred_labels != true_labels:
                    self.writer(sent_words, true_labels, pred_labels, pno, ignore_label)
            else:
                self.writer(sent_words, true_labels, pred_labels, pno, ignore_label)

    def convert_2_text(self, list2d):
        return [[self.id2label[idx] for idx in list1d] for list1d in list2d]
