import argparse
import pickle

import os
import torch
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from corpus.BratWriter import Writer, BratFile
from corpus.InferenceDataset import InferenceDataset
from corpus.WLPDataset import WLPDataset
from model.multi_batch.MultiBatchSeqNet import MultiBatchSeqNet


def multi_batchify(samples):
    samples = sorted(samples, key=lambda s: len(s.SENT), reverse=True)
    SENT, X, C = zip(*[(sample.SENT, sample.X, sample.C) for sample in samples])

    return SENT, X, C


def argmax(var):
    assert isinstance(var, Variable)
    _, preds = torch.max(var.data, 1)

    preds = preds.cpu().numpy().tolist()
    return preds


def write_brat(sents, pred_list, save_path):
    print("Writing Brat File ...")
    bratfile = BratFile(save_path, "brat")
    for sent in sents:
        bratfile.writer(sent, pred_list, "brat", ignore_label=[])


def to_variables(X, C, lm_vocab_size):
    x_var = X
    c_var = C
    lm_x = [[lm_vocab_size - 1 if (x >= lm_vocab_size) else x for x in x1d] for x1d in X]

    return x_var, c_var, lm_x


def roll(pred, seq_lengths):
    # converts 1d list to 2d list
    ret = []
    start = 0
    for seq_l in seq_lengths:
        ret.append(pred[start:start + seq_l])
        start += seq_l
    return ret


def test(name, data, tag_idx, model, lm_vocab_size, char_level):
    pred_list = []
    sents = []
    for SENT, X, C in tqdm(data, desc=name, total=len(data)):
        np.set_printoptions(threshold=np.nan)
        model.init_state(len(X))
        x_var, c_var, lm_x = to_variables(X=X, C=C, lm_vocab_size=lm_vocab_size)

        if char_level == "Attention":
            lm_f_out, lm_b_out, seq_out, seq_lengths, emb, char_emb = model(x_var, c_var)
        else:
            lm_f_out, lm_b_out, seq_out, seq_lengths = model(x_var, c_var)

        pred = argmax(seq_out)
        preds = roll(pred, seq_lengths)
        for pred, sent in zip(preds, SENT):
            pred_list.append(pred[1:-1])
            sents.append(sent)

    return sents, pred_list


def main(p_txt, cfg):
    corpus, the_model = load_model_and_corpus(cfg)
    inference(the_model, corpus, p_txt, cfg)


def load_model_and_corpus(cfg):
    model_save_path = cfg['MODEL_SAVE_PATH']
    print("Loading Dataset ...")
    corpus = pickle.load(open(cfg["CORPUS_FILE"], "rb"))
    print("Loading Model ...")
    # init model
    the_model = MultiBatchSeqNet(emb_mat=corpus.embedding_matrix,
                                 categories=len(corpus.tag_idx.keys()) + 2,  # +2 for start and end tags of a seq
                                 batch_size=cfg['BATCH_SIZE'],
                                 isCrossEnt=False,
                                 char_level=cfg['CHAR_LEVEL'],
                                 pos_feat="No",
                                 use_cuda=False,
                                 dep_rel_feat="No",
                                 dep_word_feat="No")

    the_model.load_state_dict(torch.load(model_save_path, map_location=lambda storage, loc: storage))
    return corpus, the_model


def inference(the_model, corpus, p_txt, cfg):
    dataset = InferenceDataset(p_txt=p_txt,
                               word_index=corpus.word_index,
                               char_index=corpus.char_index,
                               is_oov=corpus.is_oov,
                               sent_start=cfg['SENT_START'],
                               sent_end=cfg['SENT_END'],
                               word_start=cfg['WORD_START'],
                               word_end=cfg['WORD_END'],
                               unk=cfg['UNK'])

    data_loader = DataLoader(dataset, batch_size=cfg['BATCH_SIZE'], num_workers=8, collate_fn=multi_batchify)
    print("Testing ...")
    sents, pred_list = test("test", data_loader, corpus.tag_idx, the_model, cfg['LM_VOCAB_SIZE'], cfg['CHAR_LEVEL'])

    brat_writer = Writer(cfg['CONF_DIR'], cfg['BRAT_SAVE_PATH'], "full_out", corpus.tag_idx)

    sents = dataset.undo_sort(sents)
    pred_list = dataset.undo_sort(pred_list)
    brat_writer.gen_one_file(sents, pred_list, cfg['BRAT_SAVE_PATH'], "brat")


def init_args():
    parser = argparse.ArgumentParser(description='Sequence labeler.')
    parser.add_argument('--yaml', type=str,
                        help='config file path')
    args = parser.parse_args()
    return args


def parse_yaml(cfg_path):
    with open(cfg_path, 'r') as stream:
        return yaml.load(stream)


if __name__ == '__main__':
    args = init_args()
    yaml_cfg = parse_yaml(args.yaml)

    with open(yaml_cfg['SAMPLE_PROTOCOL_FILE'], 'r') as p_f:
        p_text = p_f.read()
    main(p_text, yaml_cfg)
