import os

from visualization.model_read import model_read_word_emb, model_read_pos
from visualization.run import viz_tsne, simple_vis_tsne

import config as cfg
import numpy as np
import pandas as pd
import matplotlib2tikz

model_save_path_pnp = os.path.join(cfg.MODEL_SAVE_DIR,
                                   "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.9_CH_Input_POS_Yes_DEP_L_No_DEP_W_No.m")

model_save_path_po = os.path.join(cfg.MODEL_SAVE_DIR,
                                  "LSTM_WE_POS_R_0_WE_False_G_0.8_CH_None_F_None.m")

model_save_path_rand = os.path.join(cfg.MODEL_SAVE_DIR,
                                    "LSTM_WE_RUN_T_SIZE_100_R_0_WE_random_G_0.8_CH_None_F_None.m")

model_save_path_pos_input = os.path.join(cfg.MODEL_SAVE_DIR,
                                         "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.3_CH_Input_POS_Yes_DEP_L_No_DEP_W_No.m")
model_save_path_pos_attention = os.path.join(cfg.MODEL_SAVE_DIR,
                                             "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.6_CH_Attention_POS_Yes_DEP_L_No_DEP_W_No.m")
model_save_path_pos_none = os.path.join(cfg.MODEL_SAVE_DIR,
                                        "ONLY_POS_T_SIZE_100_R_0_WE_pre_and_post_G_0.4_CH_None_POS_Yes_DEP_L_No_DEP_W_No.m")

model_save_path_pos = os.path.join(cfg.MODEL_SAVE_DIR,
                                   "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.9_CH_None_POS_Yes_DEP_L_No_DEP_W_No.m")


def my_tsne(save_path, labels):
    X, y = model_read_word_emb(save_path, cfg.DB_WITH_POS)
    print(X.shape)
    simple_vis_tsne(X, y, labels)


def write_labels(save_path):
    X, y = model_read_word_emb(save_path, cfg.DB_WITH_POS)

    zero_ids = np.where(~X.any(axis=1))[0]

    oov_words = [y[i] for i in zero_ids.tolist()]

    labels = [[word, "OOV"] if word in oov_words else [word, "Not OOV"] for word in y]

    df = pd.DataFrame(labels, columns=["WordFeatureGroup", "Vocab"])

    df.to_csv("visualization/data/oov_words_pnp.csv", sep='\t', index=False)

if __name__ == '__main__':
    gen_image = True
    model_path = model_save_path_pnp

    if gen_image:
        df = pd.read_csv("visualization/data/oov_words.csv", sep='\t')
        oov_labels = df["Vocab"].tolist()
        my_tsne(model_path, oov_labels)

    else:
        X, y = model_read_word_emb(model_path, cfg.DB_WITH_POS_DEP)

        write_labels(model_path)
        df = pd.DataFrame(X)

        df.to_csv("visualization/data/PNP_none.csv", sep='\t', header=False, index=False)

        with open("visualization/data/label_PNP_none.txt", 'w') as f:
            f.writelines([w + "\n" for w in y])
