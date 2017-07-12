import random
from decimal import Decimal

import sys

import time
from torch import nn, optim, max, LongTensor, cuda, sum, transpose, torch, stack
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from corpus.BratWriter import BratFile
from corpus.Manager import Manager
import config as cfg
from model.SeqNet import SeqNet
from model.utils import to_scalar, all_zeros, is_batch_zeros
from postprocessing.evaluator import Evaluator
from preprocessing.text_processing import prepare_embeddings
import numpy as np
import pickle

from preprocessing.utils import quicksave, quickload, touch


def argmax(var):
    assert isinstance(var, Variable)
    _, preds = torch.max(var.data, 1)

    preds = preds.cpu().numpy().tolist()

    preds = [pred[0] for pred in preds]
    return preds


def train_a_epoch(name, data, model, optimizer, seq_criterion, lm_f_criterion, lm_b_criterion, att_loss, gamma):
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)

    for sample in data:
        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        model.init_state()
        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, emb, char_emb = model(Variable(cuda.LongTensor([sample.X])), sample.C)

            t = []

            t = is_batch_zeros(emb.squeeze())

            char_att_loss = att_loss(emb.squeeze(), char_emb.squeeze(), t)
        else:
            lm_f_out, lm_b_out, seq_out = model(Variable(cuda.LongTensor([sample.X])), sample.C)

        cfg.ver_print("lm_f_out", lm_f_out)
        cfg.ver_print("lm_b_out", lm_b_out)
        cfg.ver_print("seq_out", seq_out)

        cfg.ver_print("tensor X variable", Variable(cuda.FloatTensor([sample.X])))

        # remove start and stop tags
        seq_out = seq_out[1: -1]
        pred = argmax(seq_out)

        cfg.ver_print("Predicted output", pred)

        seq_loss = seq_criterion(seq_out, Variable(cuda.LongTensor(sample.Y)))

        # to limit the vocab size of the sample sentence ( trick used to improve lm model)
        lm_X = [cfg.LM_MAX_VOCAB_SIZE - 1 if (x >= cfg.LM_MAX_VOCAB_SIZE) else x for x in sample.X]

        cfg.ver_print("Sample input", lm_X)

        lm_f_loss = lm_f_criterion(lm_f_out.squeeze(), Variable(cuda.LongTensor(lm_X)[1:]).squeeze())
        lm_b_loss = lm_b_criterion(lm_b_out.squeeze(), Variable(cuda.LongTensor(lm_X)[:-1]).squeeze())

        if cfg.CHAR_LEVEL == "Attention":
            total_loss = seq_loss + Variable(cuda.FloatTensor([gamma])) * (lm_f_loss + lm_b_loss) + char_att_loss
        else:
            total_loss = seq_loss + Variable(cuda.FloatTensor([gamma])) * (lm_f_loss + lm_b_loss)

        # print('loss = {0}'.format(to_scalar(loss)))

        evaluator.append_data(to_scalar(total_loss), pred, sample.X[1:-1], sample.Y)

        total_loss.backward()

        if cfg.CLIP is not None:
            clip_grad_norm(model.parameters(), cfg.CLIP)
        optimizer.step()
    evaluator.gen_results()

    return evaluator, model


def build_model(train_data, dev_data, embedding_matrix):
    # init model
    model = SeqNet(embedding_matrix, False)

    # Turn on cuda
    model = model.cuda()

    # verify model
    print(model)
    # print(list(model.parameters()))

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
    best_res_val = 0.0
    best_epoch = 0
    for epoch in range(cfg.MAX_EPOCH):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)

        random.seed(epoch)
        data_copy = list(train_data)
        if cfg.RANDOM_TRAIN:
            random.shuffle(data_copy)

        train_eval, model = train_a_epoch("train", data_copy, model, optimizer, seq_criterion,
                                          lm_f_criterion, lm_b_criterion, att_loss, cfg.LM_GAMMA)
        train_eval.print_results()

        dev_eval, _, _ = test("dev", dev_data, model)

        dev_eval.verify_results()

        # pick the best epoch
        if epoch == 0 or (dev_eval.results[cfg.BEST_MODEL_SELECTOR] > best_res_val):
            best_epoch = epoch
            best_res_val = dev_eval.results[cfg.BEST_MODEL_SELECTOR]
            torch.save(model, cfg.MODEL_SAVE_FILEPATH)

        print("current dev score: {0}".format(dev_eval.results[cfg.BEST_MODEL_SELECTOR]))
        print("best dev score: {0}".format(best_res_val))
        print("best_epoch: {1}".format(cfg.BEST_MODEL_SELECTOR, str(best_epoch)))

        # if the best epoch model outperforms MA
        if 0 < cfg.MAX_EPOCH_IMP <= (epoch - best_epoch):
            break
    print("Loading Best Model ...")
    model = torch.load(cfg.MODEL_SAVE_FILEPATH)
    return model


def test(name, data, model):
    correct = 0
    total = 0
    pred_list = []
    true_list = []
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)
    for sample in data:
        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, emb, char_emb = model(Variable(cuda.LongTensor([sample.X])), sample.C)
        else:
            lm_f_out, lm_b_out, seq_out = model(Variable(cuda.LongTensor([sample.X])), sample.C)

        # remove start and stop tags
        seq_out = seq_out[1:-1]

        pred = argmax(seq_out)
        evaluator.append_data(0, pred, sample.X[1:-1], sample.Y)

        # print(predicted)
        pred_list.append(pred)
        true_list.append(sample.Y)

    evaluator.gen_results()
    evaluator.print_results()

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
        print("Loading corpus and Embedding Matrix ...")
        corpus, embedding_matrix = pickle.load(open(loadfile, "rb"))
    else:
        print("Preparing Embedding Matrix ...")
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print("Loading Data ...")
        corpus = Manager(load_pos=True, word_index=word_index, char_index=char_index)
        corpus.gen_data((cfg.TRAIN_PERCENT, cfg.DEV_PERCENT, cfg.TEST_PERCENT), replace_digit=cfg.REPLACE_DIGITS,
                        to_filter=cfg.FILTER_ALL_NEG)

        if savefile:
            pickle.dump((corpus, embedding_matrix), open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus, embedding_matrix


def single_run(corpus, embedding_matrix, index):
    the_model = build_model(corpus.train, corpus.dev, embedding_matrix)

    print("Testing ...")
    test_eval, pred_list, true_list = test("test", corpus.test, the_model)

    print("Writing Brat File ...")
    bratfile_full = BratFile(cfg.PRED_BRAT_FULL + str(index), cfg.TRUE_BRAT_FULL + str(index))
    bratfile_inc = BratFile(cfg.PRED_BRAT_INC + str(index), cfg.TRUE_BRAT_INC + str(index))

    bratfile_full.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
                              [sample.P for sample in corpus.test],
                              true_list, pred_list, doFull=True)
    bratfile_inc.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
                             [sample.P for sample in corpus.test],
                             true_list, pred_list, doFull=False)

    test_eval.print_results()

    return test_eval


if __name__ == '__main__':
    dataset, emb_mat = dataset_prep(savefile=cfg.DB_WITH_FEATURES)

    i = 400
    # touch(cfg.RESULT_FILE)
    # LSTM + SOFTMAX
    # cfg.LM_GAMMA = 0
    # cfg.CHAR_LEVEL = None
    # # lr = 0.03
    # i = 0
    # lrs = [0.03, 0.3, 1]
    # for lr in lrs:
    #     cfg.LEARNING_RATE = lr
    #     print("LSTM + SOFTMAX; lr={0}".format(lr))
    #     test_ev = single_run(corpus, emb_mat, i)
    #     test_ev.write_results(cfg.RESULT_FILE, "LSTM + SOFTMAX; lr={0}".format(lr))
    #     i += 1
    #
    # # LSTM + SOFTMAX + LM
    # cfg.CHAR_LEVEL = None
    #
    # lrs = [0.03, 0.3, 1]
    # gamma = [0.1, 0.3, 0.5]
    #
    # for lr in lrs:
    #     for g in gamma:
    #         cfg.LEARNING_RATE = lr
    #         cfg.LM_GAMMA = g
    #         print("LSTM + SOFTMAX + LM; lr={0}; g={1}".format(lr, g))
    #         test_ev = single_run(corpus, emb_mat, i)
    #         test_ev.write_results(cfg.RESULT_FILE, "LSTM + SOFTMAX + LM; lr={0}; g={1}".format(lr, g))
    #         i += 1

    # LSTM + SOFTMAX + LM + CHAR_INPUT
    cfg.CHAR_LEVEL = "Attention"
    cfg.CHAR_VOCAB = len(dataset.char_index.items())

    lrs = [0.03, 0.3, 1]
    gamma = [0.1, 0.3, 0.5]

    for lr in lrs:
        for g in gamma:
            cfg.LEARNING_RATE = lr
            cfg.LM_GAMMA = g
            print("LSTM + SOFTMAX + LM + CHAR_ATTENTION; lr={0}; g={1}".format(lr, g))
            test_ev = single_run(dataset, emb_mat, i)
            test_ev.write_results(cfg.RESULT_FILE, "LSTM + SOFTMAX + LM + CHAR_ATTENTION; lr={0}; g={1}".format(lr, g))
            i += 1

