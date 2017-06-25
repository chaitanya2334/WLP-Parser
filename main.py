import random
from decimal import Decimal

import sys
from torch import nn, optim, max, LongTensor, cuda, sum, transpose, torch
from torch.autograd import Variable

from corpus.Manager import Manager
import config as cfg
from model.SeqNet import SeqNet
from postprocessing.evaluator import Evaluator
from preprocessing.text_processing import prepare_embeddings
import numpy as np





def argmax(var):
    assert isinstance(var, Variable)
    _, preds = torch.max(var.data, 1)

    preds = preds.cpu().numpy().tolist()

    preds = [pred[0] for pred in preds]
    return preds


def train_a_epoch(name, data, model, optimizer, criterion):
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)

    for sample in data:
        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        model.hidden_state = model.init_state()
        out = model(Variable(cuda.LongTensor([sample.X])))

        cfg.ver_print("out", out[:, 1])
        cfg.ver_print("tensor Y", Variable(cuda.FloatTensor(sample.Y)))

        pred = argmax(out)

        cfg.ver_print("Predicted output", pred)

        loss = criterion(out, Variable(cuda.LongTensor(sample.Y)))

        # print('loss = {0}'.format(to_scalar(loss)))

        evaluator.append_data(to_scalar(loss), pred, sample.X, sample.Y)

        loss.backward()
        optimizer.step()
    evaluator.gen_results()

    return evaluator, model


def build_model(train_data, dev_data, embedding_matrix):
    # init model
    model = SeqNet(embedding_matrix)

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
    criterion = nn.NLLLoss()
    best_res_val = 0.0
    best_epoch = 0
    for epoch in range(cfg.MAX_EPOCH):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)

        random.seed(epoch)
        data_copy = list(train_data)
        random.shuffle(data_copy)

        print(train_data)
        print(data_copy)
        train_eval, model = train_a_epoch("train", data_copy, model, optimizer, criterion)
        train_eval.print_results()

        dev_eval = test("dev", dev_data, model)

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
        outputs = model(Variable(cuda.LongTensor([sample.X])))
        _, predicted = max(outputs.data, 1)

        pred = argmax(outputs)
        evaluator.append_data(0, pred, sample.X, sample.Y)

        predicted = transpose(predicted, 0, 1)
        predicted = predicted.view(predicted.size(1))
        # print(predicted)
        total += Variable(cuda.LongTensor(sample.Y)).size(0)
        correct += (predicted == cuda.LongTensor(sample.Y)).sum()
        pred_list.append(predicted.cpu().numpy())
        true_list.append(np.array(sample.Y))

    evaluator.gen_results()

    return evaluator


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


if __name__ == '__main__':
    corpus = Manager(cfg.ARTICLES_FOLDERPATH, cfg.COMMON_FOLDERPATH)
    print("Preparing Embedding Matrix ...")
    embedding_matrix, word_index = prepare_embeddings()
    print("Loading Data ...")
    corpus.gen_data(cfg.TRAIN_PERCENT, cfg.DEV_PERCENT, cfg.TEST_PERCENT, word_index)

    print("Training ...")
    the_model = build_model(corpus.train, corpus.dev,  embedding_matrix)

    print("Testing ...")
    test_eval = test("test", corpus.test, the_model)

    test_eval.print_results()
