import argparse
import random
import os

import time
from torch import nn, optim, max, LongTensor, cuda, sum, transpose, torch, stack, tensor
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

from corpus.BratWriter import BratFile
from corpus.Manager import Manager, CustomDataset
import config as cfg
from model.SeqNet import SeqNet
from model.utils import to_scalar, all_zeros, is_batch_zeros
from postprocessing.evaluator import Evaluator
from preprocessing.text_processing import prepare_embeddings
import numpy as np
import pickle
import pandas as pd

from preprocessing.utils import quicksave, quickload, touch


def argmax(var):
    assert isinstance(var, Variable)
    _, preds = torch.max(var.data, 1)

    preds = preds.cpu().numpy().tolist()

    preds = [pred[0] for pred in preds]
    return preds


def train_a_epoch(name, data, model, optimizer, seq_criterion, lm_f_criterion, lm_b_criterion, att_loss, gamma):
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)

    for sample in tqdm(data, desc=name, total=len(data)):

        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()
        model.init_state()

        # padding cos features are not designed for start and end sentence tags
        # f = np.lib.pad(sample.F, [(1, 1), (0, 0)], 'constant', constant_values=(0, 0))

        np.set_printoptions(threshold=np.nan)

        x_var = Variable(cuda.LongTensor([sample.X]))
        c_var = sample.C
        #f_var = Variable(torch.from_numpy(f)).float().unsqueeze(dim=0).cuda()
        pos_var = Variable(torch.from_numpy(sample.POS).cuda()).unsqueeze(dim=0)
        rel_var = Variable(torch.from_numpy(sample.REL).cuda()).unsqueeze(dim=0)
        dep_var = Variable(cuda.LongTensor([sample.DEP]))

        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, emb, char_emb = model(x_var, c_var, pos_var, rel_var, dep_var)
            t = is_batch_zeros(emb.squeeze())
            char_att_loss = att_loss(emb.squeeze(), char_emb.squeeze(), t)

        else:
            lm_f_out, lm_b_out, seq_out = model(x_var, c_var, pos_var, rel_var, dep_var)

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


def build_model(train_dataset, dev_dataset, embedding_matrix, model_save_path):
    # init model
    model = SeqNet(embedding_matrix, isCrossEnt=False, char_level=cfg.CHAR_LEVEL, pos_feat=cfg.POS_FEATURE,
                   dep_rel_feat=cfg.DEP_LABEL_FEATURE, dep_word_feat=cfg.DEP_WORD_FEATURE)

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
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=cfg.RANDOM_TRAIN,
                                  num_workers=28, collate_fn=lambda x: x[0])

        train_eval, model = train_a_epoch("train", train_loader, model, optimizer, seq_criterion,
                                          lm_f_criterion, lm_b_criterion, att_loss, cfg.LM_GAMMA)
        train_eval.print_results()

        dev_loader = DataLoader(dev_dataset, batch_size=1, num_workers=28, collate_fn=lambda x: x[0])

        dev_eval, _, _ = test("dev", dev_loader, model)

        dev_eval.verify_results()

        # pick the best epoch
        if epoch == 0 or (dev_eval.results[cfg.BEST_MODEL_SELECTOR] > best_res_val):
            best_epoch = epoch
            best_res_val = dev_eval.results[cfg.BEST_MODEL_SELECTOR]
            torch.save(model, model_save_path)

        print("current dev score: {0}".format(dev_eval.results[cfg.BEST_MODEL_SELECTOR]))
        print("best dev score: {0}".format(best_res_val))
        print("best_epoch: {1}".format(cfg.BEST_MODEL_SELECTOR, str(best_epoch)))

        # if the best epoch model outperforms MA
        if 0 < cfg.MAX_EPOCH_IMP <= (epoch - best_epoch):
            break
    print("Loading Best Model ...")
    model = torch.load(model_save_path)
    return model


def test(name, data, model):
    correct = 0
    total = 0
    pred_list = []
    true_list = []
    evaluator = Evaluator(name, [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)
    for sample in tqdm(data, desc=name, total=len(data)):
        # f = np.lib.pad(sample.F, [(1, 1), (0, 0)], 'constant', constant_values=(0, 0))
        np.set_printoptions(threshold=np.nan)

        x_var = Variable(cuda.LongTensor([sample.X]))
        c_var = sample.C
        # f_var = Variable(torch.from_numpy(f)).float().unsqueeze(dim=0).cuda()
        pos_var = Variable(torch.from_numpy(sample.POS).cuda()).unsqueeze(dim=0)
        rel_var = Variable(torch.from_numpy(sample.REL).cuda()).unsqueeze(dim=0)
        dep_var = Variable(cuda.LongTensor([sample.DEP]))

        if cfg.CHAR_LEVEL == "Attention":
            lm_f_out, lm_b_out, seq_out, emb, char_emb = model(x_var, c_var, pos_var, rel_var, dep_var)
        else:
            lm_f_out, lm_b_out, seq_out = model(x_var, c_var, pos_var, rel_var, dep_var)

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
        corpus.gen_data(cfg.PER, cfg.TRAIN_PER)
        zero_ids = np.where(~embedding_matrix.any(axis=1))[0]
        embedding_matrix[zero_ids] = np.random.uniform(-0.01, 0.01, (1, embedding_matrix.shape[1]))

    else:
        print("Preparing Embedding Matrix ...")
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print("Loading Data ...")
        corpus = Manager(load_feat=True, word_index=word_index, char_index=char_index)
        corpus.gen_data(cfg.PER, cfg.TRAIN_PER)

    if savefile:
        print("Saving corpus and embedding matrix ...")
        pickle.dump((corpus, embedding_matrix), open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus, embedding_matrix


def single_run(corpus, embedding_matrix, index, title, overwrite, only_test=False):
    model_save_path = os.path.join(cfg.MODEL_SAVE_DIR, title + ".m")
    if not only_test:
        the_model = build_model(corpus.train, corpus.dev, embedding_matrix, model_save_path)
    else:
        the_model = torch.load(model_save_path)

    print("Testing ...")
    test_loader = DataLoader(corpus.test, batch_size=1, num_workers=28, collate_fn=lambda x: x[0])
    test_eval, pred_list, true_list = test("test", test_loader, the_model)

    print("Writing Brat File ...")
    bratfile_full = BratFile(cfg.PRED_BRAT_FULL + title, cfg.TRUE_BRAT_FULL + title)
    bratfile_inc = BratFile(cfg.PRED_BRAT_INC + title, cfg.TRUE_BRAT_INC + title)

    bratfile_full.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
                              [sample.P for sample in corpus.test],
                              true_list, pred_list, doFull=True)
    bratfile_inc.from_labels([corpus.to_words(sample.X[1:-1]) for sample in corpus.test],
                             [sample.P for sample in corpus.test],
                             true_list, pred_list, doFull=False)

    test_eval.print_results()
    txt_res_file = os.path.join(cfg.TEXT_RESULT_DIR, title + ".txt")
    csv_res_file = os.path.join(cfg.CSV_RESULT_DIR, title + ".csv")
    test_eval.write_results(txt_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)
    test_eval.write_csv_results(csv_res_file, title + "g={0}".format(cfg.LM_GAMMA), overwrite)

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
    s += "TRAIN_WORD_EMB = " + str(cfg.TRAIN_WORD_EMB) + "\n"
    s += "LM_GAMMA = " + str(cfg.LM_GAMMA) + "\n"
    s += "CHAR_LEVEL = " + cfg.CHAR_LEVEL + "\n"
    s += "POS = " + cfg.POS_FEATURE + "\n"
    s += "DEP_LABEL = " + cfg.DEP_LABEL_FEATURE + "\n"
    s += "DEP_WORD = " + cfg.DEP_WORD_FEATURE + "\n"

    return s

if __name__ == '__main__':
    # args = build_cmd_parser()

    # if args.train_per is not None:
    #     cfg.TRAIN_PER = args.train_per
    #
    # cfg.TRAIN_WORD_EMB = args.train_word_emb
    # cfg.CHAR_LEVEL = args.char_level
    # cfg.POS_FEATURE = args.pos
    # cfg.DEP_LABEL_FEATURE = args.dep_label
    # cfg.DEP_WORD_FEATURE = args.dep_word
    # cfg.LM_GAMMA = args.lm_gamma

    dataset, emb_mat = dataset_prep(loadfile=cfg.DB_WITH_POS_DEP)
    # i = 0
    #
    # if cfg.CHAR_LEVEL != "None":
    #     cfg.CHAR_VOCAB = len(dataset.char_index.items())
    #
    # if cfg.POS_FEATURE == "Yes":
    #     cfg.POS_VOCAB = len(dataset.pos_ids)
    # if cfg.DEP_LABEL_FEATURE == "Yes":
    #     cfg.REL_VOCAB = len(dataset.rel_ids)
    # if cfg.DEP_WORD_FEATURE == "Yes":
    #     cfg.DEP_WORD_VOCAB = emb_mat.shape[0]
    #
    # print(current_config())
    # test_ev = single_run(dataset, emb_mat, i, args.filename, overwrite=False)
