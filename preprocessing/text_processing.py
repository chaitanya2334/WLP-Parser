# TODO figure out the right name for this python file
import re
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from torch import LongTensor, cuda
import config as cfg
import numpy as np
import itertools
import collections


def gen_list2id_dict(list_, min_freq=-1, insert_words=None, lowercase=False, replace_digits=False):
    """
    Iterates over texts and creates a word2id mapping.
    """
    counter = collections.Counter()
    for item in list_:
        if lowercase:
            item = item.lower()
        if replace_digits:
            item = re.sub(r'\d', '0', item)
        counter.update(item.strip().split())

    list2id = collections.OrderedDict()
    if insert_words is not None:
        for word in insert_words:
            list2id[word] = len(list2id)

    word_count_list = counter.most_common()

    for (word, count) in word_count_list:
        if min_freq <= 0 or count >= min_freq:
            list2id[word] = len(list2id)

    return list2id


# this will convert a batch of variable length samples into padded tensor.
# the batch variable is a list of list of words,
# where the inner list is of variable length (represents a single sentence)
def torchify(dataset, padding=True):
    assert isinstance(dataset, Data)

    batch = dataset.X
    label = dataset.Y

    # sort samples by length in descending order
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    if padding:
        batch = pad(batch)
    batch = LongTensor(batch)

    return Data(batch, label)


def pad(batch):
    max_len = len(batch[0])
    for sample in batch:
        assert isinstance(sample, list)

        for x in range(max_len - len(sample)):
            sample.append(0)

    return batch


def batchify(dataset, batch_size=cfg.BATCH_SIZE):
    assert isinstance(dataset, Data)

    batch = dataset.X
    label = dataset.Y

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = batch.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = batch.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    if cuda.is_available():
        data = data.cuda()
    return data
