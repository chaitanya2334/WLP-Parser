# TODO figure out the right name for this python file
import re
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from torch import LongTensor, cuda

from corpus.Manager import Manager, Data
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
        if lowercase == True:
            item = item.lower()
        if replace_digits == True:
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


def prepare_embeddings(use_norm=False, replace_digit=True, load_bin=True, support_start_stop=True):
    corpus = Manager()

    # get all the sentences each sentence is a sequence of words (list of words)
    sent_iter = corpus.load_tokenized_sents(corpus.load_textfiles(cfg.ARTICLES_FOLDERPATH))

    if replace_digit:
        sent_iter = corpus.replace_num(sent_iter)

    # train a skip gram model to generate word vectors. Vectors will be of dimension given by 'size' parameter.
    print("         Loading Word2Vec ...")
    if load_bin:
        print("                     Loading a Massive File ...")
        skip_gram_model = KeyedVectors.load_word2vec_format(cfg.PUBMED_AND_PMC_W2V_BIN, binary=True)
    else:
        skip_gram_model = Word2Vec(sentences=sent_iter, size=cfg.EMBEDDING_DIM, sg=1, window=10, min_count=1, workers=4)

    cfg.ver_print("word2vec emb size", skip_gram_model.vector_size)
    # TODO fix tokenizer. Incorrect values.

    # this is fine. this is a hack. i took away the tokenizing part of this class. i use it just to create the word index.
    tokenizer = Tokenizer(char_level=True)
    sent_iter_flat = list(itertools.chain.from_iterable(sent_iter))

    list_of_chars = list(itertools.chain.from_iterable([list(word) for word in sent_iter_flat]))

    tokenizer.fit_on_texts(sent_iter)

    word_index = tokenizer.word_index

    char_index = gen_list2id_dict(list_of_chars, insert_words=['<w>', '</w>', '<s>', '</s>'])

    print(char_index)

    cfg.CHAR_VOCAB = len(char_index.items())

    if support_start_stop:
        word_index['<s>'] = len(word_index)+1
        word_index['</s>'] = len(word_index)+1

    with open('test_tokenizer.txt', 'w', encoding='utf-8') as out:
        out.writelines([item + ' ' + str(word_index[item]) + '\n' for item in sent_iter_flat])

    embedding_matrix = np.zeros((len(word_index) + 1, cfg.EMBEDDING_DIM))
    print("         Populating Embedding Matrix ...")
    with open(cfg.OOV_FILEPATH, 'w') as f:
        f.write("Out of Vocabulary words\n")

    for word, i in word_index.items():
        try:
            embedding_vector = skip_gram_model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # not found in vocab
            # words not found in embedding index will be all-zeros.
            with open(cfg.OOV_FILEPATH, 'a') as f:
                f.write('{0}\n'.format(word))
            cfg.ver_print('out of vocabulary word', word)

    return embedding_matrix, word_index, char_index


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
