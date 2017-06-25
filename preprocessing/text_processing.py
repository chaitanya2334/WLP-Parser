# TODO figure out the right name for this python file

from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from torch import LongTensor, cuda

from corpus.Manager import Manager, Dataset
import config as cfg
import numpy as np
import itertools


def prepare_embeddings(use_norm=False, load_bin=True):
    corpus = Manager(corpus_path=cfg.CORPUS_FOLDERPATH, common_path=cfg.COMMON_FOLDERPATH)

    # get all the sentences each sentence is a sequence of words (list of words)
    sent_iter = corpus.load_tokenized_sents(corpus.load_textfiles())

    # train a skip gram model to generate word vectors. Vectors will be of dimension given by 'size' parameter.
    print("         Loading Word2Vec ...")
    if load_bin:
        print("                     Loading a Massive File ...")
        skip_gram_model = KeyedVectors.load_word2vec_format(cfg.PUBMED_AND_PMC_W2V_BIN, binary=True)
    else:
        skip_gram_model = Word2Vec(sentences=sent_iter, size=cfg.EMBEDDING_DIM, sg=1, window=10, min_count=1, workers=4)

    texts = corpus.load_sents(corpus.load_textfiles())
    cfg.ver_print("word2vec emb size", skip_gram_model.vector_size)
    # TODO fix tokenizer. Incorrect values.
    tokenizer = Tokenizer(char_level=True)
    sent_iter_flat = list(itertools.chain.from_iterable(sent_iter))

    tokenizer.fit_on_texts(sent_iter)

    word_index = tokenizer.word_index

    with open('test_tokenizer.txt', 'w', encoding='utf-8') as out:
        out.writelines([item + ' ' + str(word_index[item]) + '\n' for item in sent_iter_flat])

    embedding_matrix = np.zeros((len(word_index) + 1, cfg.EMBEDDING_DIM))
    print("         Populating Embedding Matrix ...")
    for word, i in word_index.items():
        try:
            embedding_vector = skip_gram_model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # not found in vocab
            # words not found in embedding index will be all-zeros.
            with open(cfg.OOV_FILEPATH, 'w') as f:
                f.write('{0}\n'.format(word))
            cfg.ver_print('out of vocabulary word', word)

    return embedding_matrix, word_index


# this will convert a batch of variable length samples into padded tensor.
# the batch variable is a list of list of words,
# where the inner list is of variable length (represents a single sentence)
def torchify(dataset, padding=True):
    assert isinstance(dataset, Dataset)

    batch = dataset.X
    label = dataset.Y

    # sort samples by length in descending order
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    if padding:
        batch = pad(batch)
    batch = LongTensor(batch)

    return Dataset(batch, label)


def pad(batch):
    max_len = len(batch[0])
    for sample in batch:
        assert isinstance(sample, list)

        for x in range(max_len - len(sample)):
            sample.append(0)

    return batch


def batchify(dataset, batch_size=cfg.BATCH_SIZE):
    assert isinstance(dataset, Dataset)

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
