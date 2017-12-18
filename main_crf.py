import argparse
import random
import os

import time
import logging
import sys
from itertools import chain

from torch import nn, optim, max, LongTensor, cuda, sum, transpose, torch, stack, tensor
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from corpus.BratWriter import BratFile, Writer
from corpus.WLPDataset import WLPDataset
import config as cfg
from model.SeqNet import SeqNet
from model.multi_batch.BiLSTM_CRF import BiLSTM_CRF
from model.multi_batch.MultiBatchSeqNet import MultiBatchSeqNet
from model.utils import to_scalar
from postprocessing.evaluator import Evaluator
import numpy as np
import pickle
import pandas as pd

from preprocessing.utils import quicksave, quickload, touch

logger = logging.getLogger(__name__)

plt.ion()
plt.legend(loc='upper left')


def argmax(var):
    assert isinstance(var, Variable)
    _, preds = torch.max(var.data, 1)

    preds = preds.cpu().numpy().tolist()
    return preds


def train_a_epoch(name, data, tag_idx, model, optimizer):
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=tag_idx, conll_eval=True)
    t = tqdm(data, total=len(data))

    for SENT, X, Y, P in t:
        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        np.set_printoptions(threshold=np.nan)
        nnl = model.neg_log_likelihood(X, Y)
        logger.debug("tensor X variable: {0}".format(X))
        nnl.backward()
        preds = model(X)
        for pred, x, y in zip(preds, X, Y):
            evaluator.append_data(to_scalar(nnl), pred, x, y)

        if cfg.CLIP is not None:
            clip_grad_norm(model.parameters(), cfg.CLIP)

        optimizer.step()

    evaluator.classification_report()

    return evaluator, model


def plot_curve(x, y1, y2, xlabel, ylabel, title, savefile):
    plt.plot(range(x + 1), y1, 'r', label='dev')
    plt.plot(range(x + 1), y2, 'b', label='test')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(savefile)
    plt.pause(0.05)


def build_model(train_dataset, dev_dataset, test_dataset,
                collate_fn, tag_idx, is_oov, embedding_matrix, model_save_path, plot_save_path):
    # init model
    model = BiLSTM_CRF(embedding_matrix, tag_idx)

    # Turn on cuda
    model = model.cuda()

    # verify model
    print(model)

    # remove paramters that have required_grad = False

    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9)
    optimizer.zero_grad()
    model.zero_grad()

    # init loss criteria
    best_res_val_0 = 0.0
    best_epoch = 0
    dev_eval_history = []
    test_eval_history = []
    for epoch in range(cfg.MAX_EPOCH):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)

        random.seed(epoch)
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.RANDOM_TRAIN,
                                  num_workers=28, collate_fn=collate_fn)

        train_eval, model = train_a_epoch(name="train", data=train_loader, tag_idx=tag_idx,
                                          model=model, optimizer=optimizer)

        dev_loader = DataLoader(dev_dataset, batch_size=cfg.BATCH_SIZE, num_workers=28, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=28, collate_fn=collate_fn)

        dev_eval, _, _ = test("dev", dev_loader, tag_idx, model)
        test_eval, _, _ = test("test", test_loader, tag_idx, model)

        dev_eval.verify_results()
        test_eval.verify_results()
        dev_eval_history.append(dev_eval.results[cfg.BEST_MODEL_SELECTOR[0]])
        test_eval_history.append(test_eval.results['test_conll_f'])
        plot_curve(epoch, dev_eval_history, test_eval_history, "epochs", "fscore", "epoch learning curve",
                   plot_save_path)
        pickle.dump((dev_eval_history, test_eval_history), open("plot_data.p", "wb"))
        # pick the best epoch
        if epoch < cfg.MIN_EPOCH_IMP or (dev_eval.results[cfg.BEST_MODEL_SELECTOR[0]] > best_res_val_0):
            best_epoch = epoch
            best_res_val_0 = dev_eval.results[cfg.BEST_MODEL_SELECTOR[0]]

            torch.save(model, model_save_path)

        print("current dev micro_score: {0}".format(dev_eval.results[cfg.BEST_MODEL_SELECTOR[0]]))
        print("current dev macro_score: {0}".format(dev_eval.results[cfg.BEST_MODEL_SELECTOR[1]]))
        print("best dev micro_score: {0}".format(best_res_val_0))
        print("best_epoch: {0}".format(str(best_epoch)))

        # if the best epoch model outperforms MA
        if 0 < cfg.MAX_EPOCH_IMP <= (epoch - best_epoch):
            break
    print("Loading Best Model ...")

    model = torch.load(model_save_path)
    return model


def test(name, data, tag_idx, model):
    pred_list = []
    true_list = []
    full_eval = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=tag_idx, conll_eval=True)
    only_ents_eval = Evaluator("test_ents_only", [0, 1], skip_label=['B-Action', 'I-Action'],
                               main_label_name=cfg.POSITIVE_LABEL, label2id=tag_idx, conll_eval=True)

    for SENT, X, Y, P in tqdm(data, desc=name, total=len(data)):
        np.set_printoptions(threshold=np.nan)
        preds = model(X)

        for pred, x, y in zip(preds, X, Y):
            full_eval.append_data(0, pred, x, y)
            only_ents_eval.append_data(0, pred, x, y)
            pred_list.append(pred[1:-1])
            true_list.append(y[1:-1])

    full_eval.classification_report()
    only_ents_eval.classification_report()

    return full_eval, pred_list, true_list


def dataset_prep(loadfile=None, savefile=None):
    start_time = time.time()

    if loadfile:
        print("Loading corpus ...")
        corpus = pickle.load(open(loadfile, "rb"))
        with open("file_order.txt", 'w') as f:
            f.write("\n".join([p.filename for p in corpus.protocols]))
            print("DONE WRITING FILENAMES")
        corpus.gen_data(cfg.PER)
    else:
        print("Loading Data ...")
        corpus = WLPDataset(gen_feat=True, min_wcount=cfg.MIN_WORD_COUNT,
                            lowercase=cfg.LOWERCASE, replace_digits=cfg.REPLACE_DIGITS)
        corpus.gen_data(cfg.PER)

        if savefile:
            print("Saving corpus and embedding matrix ...")
            pickle.dump(corpus, open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus


def multi_batchify(samples):
    samples = sorted(samples, key=lambda s: len(s.SENT), reverse=True)

    SENT, X, Y, P = zip(*[(sample.SENT, sample.X, sample.Y, sample.P)
                                  for sample in samples])

    return SENT, X, Y, P


def to_variables(X, C, POS, Y):
    if cfg.BATCH_TYPE == "multi":
        x_var = X
        c_var = C
        pos_var = POS
        y_var = list(chain.from_iterable(list(Y)))

        lm_X = [[cfg.LM_MAX_VOCAB_SIZE - 1 if (x >= cfg.LM_MAX_VOCAB_SIZE) else x for x in x1d] for x1d in X]

    else:
        x_var = Variable(cuda.LongTensor([X]))
        c_var = C
        # f_var = Variable(torch.from_numpy(f)).float().unsqueeze(dim=0).cuda()
        pos_var = Variable(torch.from_numpy(POS).cuda()).unsqueeze(dim=0)
        lm_X = [cfg.LM_MAX_VOCAB_SIZE - 1 if (x >= cfg.LM_MAX_VOCAB_SIZE) else x for x in X]
        y_var = Variable(cuda.LongTensor(Y))

    return x_var, c_var, pos_var, y_var, lm_X


def single_run(corpus, index, title, overwrite, only_test=False):
    if cfg.BATCH_TYPE == "multi":
        collate_fn = multi_batchify
    else:
        collate_fn = lambda x: \
            (x[0].X, x[0].C, x[0].POS, x[0].REL, x[0].DEP, x[0].Y)

    model_save_path = os.path.join(cfg.MODEL_SAVE_DIR, title + ".m")
    plot_save_path = os.path.join(cfg.PLOT_SAVE_DIR, title + ".png")
    if not only_test:
        the_model = build_model(corpus.train, corpus.dev, corpus.test,
                                collate_fn, corpus.tag_idx, corpus.is_oov, corpus.embedding_matrix, model_save_path,
                                plot_save_path)
    else:
        the_model = torch.load(model_save_path)

    print("Testing ...")
    test_loader = DataLoader(corpus.test, batch_size=cfg.BATCH_SIZE, num_workers=28, collate_fn=collate_fn)
    test_eval, pred_list, true_list = test("test", test_loader, corpus.tag_idx, the_model)

    print("Writing Brat File ...")
    bratfile_full = Writer(cfg.CONF_DIR, os.path.join(cfg.BRAT_DIR, title), "full_out", corpus.tag_idx)
    bratfile_inc = Writer(cfg.CONF_DIR, os.path.join(cfg.BRAT_DIR, title), "inc_out", corpus.tag_idx)

    # convert idx to label
    test_eval.print_results()
    txt_res_file = os.path.join(cfg.TEXT_RESULT_DIR, title + ".txt")
    csv_res_file = os.path.join(cfg.CSV_RESULT_DIR, title + ".csv")
    test_eval.write_results(txt_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)
    test_eval.write_csv_results(csv_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)

    test_loader = DataLoader(corpus.test, batch_size=cfg.BATCH_SIZE, num_workers=28, collate_fn=collate_fn)
    sents = [(sent, p) for SENT, X, C, POS, Y, P in test_loader for sent, p in zip(SENT, P)]
    bratfile_full.from_labels(sents, true_list, pred_list, doFull=True)
    bratfile_inc.from_labels(sents, true_list, pred_list, doFull=False)

    return test_eval


def main(nrun=1):
    for run in range(nrun):
        dataset = dataset_prep(loadfile=cfg.DB)
        cfg.CATEGORIES = len(dataset.tag_idx.keys()) + 2  # +2 for start and end tags of a seq
        dataset.tag_idx['<s>'] = len(dataset.tag_idx.keys())
        dataset.tag_idx['</s>'] = len(dataset.tag_idx.keys())
        cfg.WORD_VOCAB = len(dataset.word_index.items())
        i = 0

        if cfg.CHAR_LEVEL != "None":
            cfg.CHAR_VOCAB = len(dataset.char_index.items())

        if cfg.POS_FEATURE == "Yes":
            cfg.POS_VOCAB = len(dataset.pos_ids)
        if cfg.DEP_LABEL_FEATURE == "Yes":
            cfg.REL_VOCAB = len(dataset.rel_ids)
        if cfg.DEP_WORD_FEATURE == "Yes":
            cfg.DEP_WORD_VOCAB = dataset.embedding_matrix.shape[0]

        test_ev = single_run(dataset, i, "BiLSTM_CRF", overwrite=False, only_test=False)
        plt.clf()


if __name__ == '__main__':
    # setup_logging()
    main(1)
