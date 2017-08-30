import pickle

import torch




def model_read_pos(model_save_file, corpus_file):
    model = torch.load(model_save_file)
    emb_matrix = model.pos_emb.weight.data.cpu().numpy()
    corpus, init_emb_mat = pickle.load(open(corpus_file, "rb"))
    pos_ids = corpus.pos_ids
    id2word = {v: k for k, v in pos_ids.items()}
    print(id2word)
    y = [id2word[i] for i in range(emb_matrix.shape[0])]
    X = emb_matrix
    return X, y


def model_read_word_emb(model_save_file, corpus_file):
    model = torch.load(model_save_file)
    #emb_matrix = model.emb_lookup.emb_mat.data.cpu().numpy()
    emb_matrix = model.emb_lookup.weight.data.cpu().numpy()
    corpus, init_emb_mat = pickle.load(open(corpus_file, "rb"))
    word_index = corpus.word_index
    id2word = {v: k for k, v in word_index.items()}
    print(id2word)
    y = [id2word[i] for i in range(1, emb_matrix.shape[0])]
    X = emb_matrix[1:]
    print(emb_matrix[word_index['onetaq']])
    return X, y