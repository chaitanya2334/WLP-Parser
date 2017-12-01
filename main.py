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

from corpus.BratWriter import BratFile
from corpus.WLPDataset import WLPDataset
import config as cfg
from model.SeqNet import SeqNet
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


def train_a_epoch(name, data, tag_idx, is_oov, model, optimizer, seq_criterion, lm_f_criterion, lm_b_criterion, att_loss,
                  gamma):
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=tag_idx, conll_eval=True)
    t = tqdm(data, total=len(data))

    if is_oov[0] == 1:
        print("Yes, UNKNOWN token is out of vocab")
    else:
        print("No, UNKNOWN token is not out of vocab")

    for X, C, POS, REL, DEP, Y in t:

        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        model.init_state(len(X))

        x_var, c_var, pos_var, rel_var, dep_var, y_var, lm_X = to_variables(X, C, POS, REL, DEP, Y)

        np.set_printoptions(threshold=np.nan)

        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, seq_lengths, emb, char_emb = model(x_var, c_var, pos_var, rel_var, dep_var)
            unrolled_x_var = list(chain.from_iterable(x_var))

            not_oov_seq = [-1 if is_oov[idx] else 1 for idx in unrolled_x_var]
            char_att_loss = att_loss(emb.squeeze(), char_emb.squeeze(), Variable(torch.cuda.LongTensor(not_oov_seq)))

        else:
            lm_f_out, lm_b_out, seq_out, seq_lengths = model(x_var, c_var, pos_var, rel_var, dep_var)

        logger.debug("lm_f_out : {0}".format(lm_f_out))
        logger.debug("lm_b_out : {0}".format(lm_b_out))
        logger.debug("seq_out : {0}".format(seq_out))

        logger.debug("tensor X variable: {0}".format(x_var))

        # remove start and stop tags
        pred = argmax(seq_out)

        logger.debug("Predicted output {0}".format(pred))
        seq_loss = seq_criterion(seq_out, Variable(torch.LongTensor(y_var)).cuda())

        # to limit the vocab size of the sample sentence ( trick used to improve lm model)
        # TODO make sure that start and end symbol of sentence gets through this filtering.
        logger.debug("Sample input {0}".format(lm_X))
        if gamma != 0:
            lm_X_f = [x1d[1:] for x1d in lm_X]
            lm_X_b = [x1d[:-1] for x1d in lm_X]
            lm_X_f = list(chain.from_iterable(lm_X_f))
            lm_X_b = list(chain.from_iterable(lm_X_b))
            lm_f_loss = lm_f_criterion(lm_f_out.squeeze(), Variable(cuda.LongTensor(lm_X_f)).squeeze())
            lm_b_loss = lm_b_criterion(lm_b_out.squeeze(), Variable(cuda.LongTensor(lm_X_b)).squeeze())

            if cfg.CHAR_LEVEL == "Attention":
                total_loss = seq_loss + Variable(cuda.FloatTensor([gamma])) * (lm_f_loss + lm_b_loss) + char_att_loss
            else:
                total_loss = seq_loss + Variable(cuda.FloatTensor([gamma])) * (lm_f_loss + lm_b_loss)

        else:
            total_loss = seq_loss

        desc = "total_loss: {0:.4f} = seq_loss: {1:.4f}".format(to_scalar(total_loss),
                                                                to_scalar(seq_loss))
        if gamma != 0:
            desc += " + gamma: {0} * (lm_f_loss: {1:.4f} + lm_b_loss: {2:.4f})".format(gamma,
                                                                                      to_scalar(lm_f_loss),
                                                                                      to_scalar(lm_b_loss))

        if cfg.CHAR_LEVEL == "Attention":
            desc += " + char_att_loss: {0:.4f}".format(to_scalar(char_att_loss))

        t.set_description(desc)

        preds = roll(pred, seq_lengths)
        for pred, x, y in zip(preds, X, Y):
            evaluator.append_data(to_scalar(total_loss), pred, x, y)

        total_loss.backward()
        if cfg.CLIP is not None:
            clip_grad_norm(model.parameters(), cfg.CLIP)

        optimizer.step()

    evaluator.classification_report()

    return evaluator, model


def roll(pred, seq_lengths):
    # converts 1d list to 2d list
    ret = []
    start = 0
    for seq_l in seq_lengths:
        ret.append(pred[start:start + seq_l])
        start += seq_l
    return ret


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
    model = MultiBatchSeqNet(embedding_matrix, batch_size=cfg.BATCH_SIZE, isCrossEnt=False, char_level=cfg.CHAR_LEVEL,
                             pos_feat=cfg.POS_FEATURE,
                             dep_rel_feat=cfg.DEP_LABEL_FEATURE, dep_word_feat=cfg.DEP_WORD_FEATURE)

    # Turn on cuda
    model = model.cuda()

    # verify model
    print(model)

    # init gradient descent optimizer

    optimizer = optim.Adadelta(model.parameters(), lr=cfg.LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9)
    optimizer.zero_grad()
    model.zero_grad()

    # init loss criteria
    seq_criterion = nn.NLLLoss()
    lm_f_criterion = nn.NLLLoss()
    lm_b_criterion = nn.NLLLoss()
    att_loss = nn.CosineEmbeddingLoss(margin=1)
    best_res_val_0 = 0.0
    best_res_val_1 = 0.0
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

        train_eval, model = train_a_epoch(name="train", data=train_loader, tag_idx=tag_idx, is_oov=is_oov,
                                          model=model, optimizer=optimizer, seq_criterion=seq_criterion,
                                          lm_f_criterion=lm_f_criterion, lm_b_criterion=lm_b_criterion,
                                          att_loss=att_loss, gamma=cfg.LM_GAMMA)

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
    correct = 0
    total = 0
    pred_list = []
    true_list = []
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=tag_idx, conll_eval=True)
    for X, C, POS, REL, DEP, Y in tqdm(data, desc=name, total=len(data)):
        np.set_printoptions(threshold=np.nan)
        model.init_state(len(X))
        x_var, c_var, pos_var, rel_var, dep_var, y_var, lm_X = to_variables(X, C, POS, REL, DEP, Y)

        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, seq_lengths, emb, char_emb = model(x_var, c_var, pos_var, rel_var, dep_var)
        else:
            lm_f_out, lm_b_out, seq_out, seq_lengths = model(x_var, c_var, pos_var, rel_var, dep_var)

        pred = argmax(seq_out)
        preds = roll(pred, seq_lengths)
        for pred, x, y in zip(preds, X, Y):
            evaluator.append_data(0, pred, x, y)
            pred_list.append(pred[1:-1])
            true_list.append(y[1:-1])

    evaluator.classification_report()

    return evaluator, pred_list, true_list


# pred and true are lists of numpy arrays. each numpy array represents a sample
def fscore(pred, true):
    assert len(pred) == len(true)
    tp, tn, fn, fp = (0,) * 4

    for p_np, t_np in zip(pred, true):
        p_l, t_l = p_np.tolist(), t_np.tolist()

        for p, t in zip(p_l, t_l):
            if p == t == 0:
                tn += 1

            elif p == t == 1:
                tp += 1

            elif p == 0 and t == 1:
                fn += 1

            elif p == 1 and t == 0:
                fp += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fs = (2 * precision * recall) / (precision + recall)
    print("tp={0}, tn={1}, fn={2}, fp={3}".format(tp, tn, fn, fp))
    print("total = {0}".format(tp + tn + fn + fp))
    print("recall = {0}/{1} = {2}".format(tp, tp + fn, recall))
    print("precision = {0}/{1} = {2}".format(tp, tp + fp, precision))
    print("fscore = {0}/{1} = {2}".format(2 * precision * recall, precision + recall, fs))


def dataset_prep(loadfile=None, savefile=None):
    start_time = time.time()

    if loadfile:
        print("Loading corpus ...")
        corpus = pickle.load(open(loadfile, "rb"))
        corpus.gen_data(cfg.PER)
    else:
        print("Loading Data ...")
        corpus = WLPDataset(gen_feat=True, min_wcount=cfg.MIN_WORD_COUNT)
        corpus.gen_data(cfg.PER)

        if savefile:
            print("Saving corpus and embedding matrix ...")
            pickle.dump(corpus, open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus


def multi_batchify(samples):
    X, C, POS, REL, DEP, Y = zip(*[(sample.X, sample.C, sample.POS, sample.REL, sample.DEP, sample.Y)
                                   for sample in samples])
    X = sorted(X, key=lambda x: len(x), reverse=True)
    Y = sorted(Y, key=lambda x: len(x), reverse=True)
    C = sorted(C, key=lambda x: len(x), reverse=True)
    POS = sorted(POS, key=lambda x: len(x), reverse=True)

    return X, C, POS, REL, DEP, Y


def to_variables(X, C, POS, REL, DEP, Y):
    if cfg.BATCH_TYPE == "multi":
        x_var = X
        c_var = C
        pos_var = POS
        rel_var = REL
        dep_var = DEP
        y_var = list(chain.from_iterable(list(Y)))

        lm_X = [[cfg.LM_MAX_VOCAB_SIZE - 1 if (x >= cfg.LM_MAX_VOCAB_SIZE) else x for x in x1d] for x1d in X]


    else:
        x_var = Variable(cuda.LongTensor([X]))
        c_var = C
        # f_var = Variable(torch.from_numpy(f)).float().unsqueeze(dim=0).cuda()
        pos_var = Variable(torch.from_numpy(POS).cuda()).unsqueeze(dim=0)
        rel_var = Variable(torch.from_numpy(REL).cuda()).unsqueeze(dim=0)
        dep_var = Variable(cuda.LongTensor([DEP]))
        lm_X = [cfg.LM_MAX_VOCAB_SIZE - 1 if (x >= cfg.LM_MAX_VOCAB_SIZE) else x for x in X]
        y_var = Variable(cuda.LongTensor(Y))

    return x_var, c_var, pos_var, rel_var, dep_var, y_var, lm_X


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
                                collate_fn, corpus.tag_idx, corpus.is_oov, corpus.embedding_matrix, model_save_path, plot_save_path)
    else:
        the_model = torch.load(model_save_path)

    print("Testing ...")
    test_loader = DataLoader(corpus.test, batch_size=cfg.BATCH_SIZE, num_workers=28, collate_fn=collate_fn)
    test_eval, pred_list, true_list = test("test", test_loader, corpus.tag_idx, the_model)

    print("Writing Brat File ...")
    bratfile_full = BratFile(cfg.PRED_BRAT_FULL + title, cfg.TRUE_BRAT_FULL + title, corpus.tag_idx)
    bratfile_inc = BratFile(cfg.PRED_BRAT_INC + title, cfg.TRUE_BRAT_INC + title, corpus.tag_idx)

    # convert idx to label
    test_eval.print_results()
    txt_res_file = os.path.join(cfg.TEXT_RESULT_DIR, title + ".txt")
    csv_res_file = os.path.join(cfg.CSV_RESULT_DIR, title + ".csv")
    test_eval.write_results(txt_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)
    test_eval.write_csv_results(csv_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)

    # bratfile_full.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
    #                          [sample.P[1:-1] for sample in corpus.test],
    #                          true_list, pred_list, doFull=True)
    # bratfile_inc.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
    #                         [sample.P for sample in corpus.test],
    #                         true_list, pred_list, doFull=False)

    return test_eval


def build_cmd_parser():
    parser = argparse.ArgumentParser(description='Action Sequence Labeler.')

    parser.add_argument('--train_per', dest='train_per', type=int, required=False,
                        help='Percentage of the train data to be actually used for training. Int between (0-100)')

    parser.add_argument('--train_word_emb', dest='train_word_emb', required=True,
                        choices=["pre_and_post", "random", "pre_only"],
                        help='Pick the word embedding strategy. pre_and_post will use pretrained word embeddings '
                             'and will also train further on those, pre_only will not train further '
                             'and random will initialize the random word embeddings')

    parser.add_argument('--lm_gamma', metavar='G', type=float, required=True,
                        help='If Language model is to be used, gamma is a gating variable that controls '
                             'how important LM should be. A float number between (0 - 1)')

    parser.add_argument('--char_level', required=True, choices=["None", "Input", "Attention"],
                        help='The char level embedding to add on top of the bi LSTM.')

    parser.add_argument('--pos', required=True, choices=["No", "Yes"],
                        help='The feature level to be added on top of the bi LSTM.')

    parser.add_argument('--dep_label', required=True, choices=["No", "Yes"],
                        help='Whether or not to include dependency labels embedding')

    parser.add_argument('--dep_word', required=True, choices=["No", "Yes"],
                        help="whether or not to include dependency word embedding")

    parser.add_argument("filename", metavar="String",
                        help="This is the filename (without ext) "
                             "that will be given to the file where all the results will be stored ")

    args = parser.parse_args()
    return args


def current_config():
    s = "TRAIN_PER = " + str(cfg.TRAIN_PER) + "\n"
    s += "WORD_VOCAB = " + str(cfg.WORD_VOCAB) + "\n"
    s += "TRAIN_WORD_EMB = " + str(cfg.TRAIN_WORD_EMB) + "\n"
    s += "LM_GAMMA = " + str(cfg.LM_GAMMA) + "\n"
    s += "CHAR_LEVEL = " + cfg.CHAR_LEVEL + "\n"
    s += "CHAR_VOCAB = {0}".format(cfg.CHAR_VOCAB) + "\n"
    s += "POS = " + cfg.POS_FEATURE + "\n"
    s += "DEP_LABEL = " + cfg.DEP_LABEL_FEATURE + "\n"
    s += "DEP_WORD = " + cfg.DEP_WORD_FEATURE + "\n"

    return s


def main(nrun=1):
    args = build_cmd_parser()

    if args.train_per is not None:
        cfg.TRAIN_PER = args.train_per

    cfg.TRAIN_WORD_EMB = args.train_word_emb
    cfg.CHAR_LEVEL = args.char_level
    cfg.POS_FEATURE = args.pos
    cfg.DEP_LABEL_FEATURE = args.dep_label
    cfg.DEP_WORD_FEATURE = args.dep_word
    cfg.LM_GAMMA = args.lm_gamma
    for run in range(nrun):
        dataset = dataset_prep(savefile=cfg.DB)
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

        print(current_config())
        test_ev = single_run(dataset, i, args.filename, overwrite=False, only_test=False)
        plt.clf()


if __name__ == '__main__':
    # setup_logging()
    main(1)
