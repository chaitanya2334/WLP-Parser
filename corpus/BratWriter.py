from conlleval import start_of_chunk, end_of_chunk


class BratFile(object):
    def __init__(self, pred_name, true_name):
        # public
        self.true_txt_fname = true_name + '.txt'
        self.true_ann_fname = true_name + '.ann'

        self.pred_txt_fname = pred_name + '.txt'
        self.pred_ann_fname = pred_name + '.ann'

        self.__clear_file(self.true_txt_fname)
        self.__clear_file(self.true_ann_fname)
        self.__clear_file(self.pred_txt_fname)
        self.__clear_file(self.pred_ann_fname)

        # private
        self.__start_true = 0
        self.__start_pred = 0
        self.__end_true = 0
        self.__end_pred = 0
        self.__tag_id_true = 0
        self.__tag_id_pred = 0

    ############################################## PUBLIC METHODS ######################################################

    def __clear_file(self, filename):
        open(filename, 'w').close()

    def from_file(self, path):
        pass

    # appends a line in txt and also tags words in that line.
    # start and end are relative positions with respect to this line and not the whole file.
    def writer(self, words, true, pred, pno, ignore_label):
        with open(self.true_txt_fname, 'a', encoding='utf-8') as true_txt, \
                open(self.true_ann_fname, 'a', encoding='utf-8') as true_ann, \
                open(self.pred_txt_fname, 'a', encoding='utf-8') as pred_txt, \
                open(self.pred_ann_fname, 'a', encoding='utf-8') as pred_ann:
            # write the text file
            true_txt.write(" ".join(words) + " [" + pno + ']\n')
            pred_txt.write(" ".join(words) + " [" + pno + ']\n')

            sent_pad = len(" [" + pno + ']')

            # write the ann file
            self.__start_true, self.__end_true, self.__tag_id_true = self.__write_ann(true_ann, words, true,
                                                                                      ignore_label,
                                                                                      self.__start_true,
                                                                                      self.__end_true,
                                                                                      sent_pad,
                                                                                      self.__tag_id_true)

            self.__start_pred, self.__end_pred, self.__tag_id_pred = self.__write_ann(pred_ann, words, pred,
                                                                                      ignore_label,
                                                                                      self.__start_pred,
                                                                                      self.__end_pred,
                                                                                      sent_pad,
                                                                                      self.__tag_id_pred)

    def from_labels(self, words, p_list, true, pred,  ignore_label="O", doFull=True):
        # words = self.__partition(words, sent_counts)
        # true = self.__partition(true, sent_counts)
        # pred = self.__partition(pred, sent_counts)

        true = self.convert_2_text(true)
        pred = self.convert_2_text(pred)

        for sent_words, sent_true, sent_pred, pno in zip(words, true, pred, p_list):
            print(sent_words)
            assert len(sent_words) == len(sent_true) == len(sent_pred), \
                "Sentence data not of equal length. " \
                "sent_words={0}, sent_true={1}, sent_pred={2}".format(sent_words, sent_true, sent_pred)

            if not doFull:
                if sent_pred != sent_true:
                    self.writer(sent_words, sent_true, sent_pred, pno, ignore_label)
            else:
                self.writer(sent_words, sent_true, sent_pred, pno, ignore_label)

    @staticmethod
    def convert_2_text(list2d):
        res = []
        for row in list2d:
            ret = []
            for bin_label in row:
                if bin_label == 0:
                    ret.append('B-Action')
                elif bin_label == 1:
                    ret.append('I-Action')
                elif bin_label == 2:
                    ret.append('O')
                else:
                    raise ValueError("Incorrect label number")

            res.append(ret)

        return res

    ###################################### PRIVATE METHODS #############################################################
    def __write_ann(self, ann, words, labels, ignore_label, start, end, sent_pad, tag_id):
        word_span = []
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
        return start, end, tag_id

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
