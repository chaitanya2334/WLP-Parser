import os

from visualization.model_read import model_read_word_emb
from visualization.run import viz_tsne, simple_vis_tsne

import config as cfg
import numpy as np
import pandas as pd

model_save_path_pnp = os.path.join(cfg.MODEL_SAVE_DIR,
                                   "FEAT_3_S_TRUN_T_SIZE_100_R_0_WE_pre_and_post_G_0.1_CH_None_POS_Yes_DEP_L_No_DEP_W_No.m")

model_save_path_po = os.path.join(cfg.MODEL_SAVE_DIR,
                                  "LSTM_WE_POS_R_0_WE_False_G_0.8_CH_None_F_None.m")

model_save_path_rand = os.path.join(cfg.MODEL_SAVE_DIR,
                                    "LSTM_WE_RUN_T_SIZE_100_R_0_WE_random_G_0.8_CH_None_F_None.m")


def my_tsne(save_path):
    X, y = model_read_word_emb(save_path, cfg.DB_WITH_POS)
    print(X.shape)
    simple_vis_tsne(X, y)


if __name__ == '__main__':
    gen_image = False
    model_path = model_save_path_pnp

    if gen_image:
        my_tsne(model_path)
    else:
        X, y = model_read_word_emb(model_path, cfg.DB_WITH_POS)

        df = pd.DataFrame(X)

        df.to_csv("RAND.csv", sep='\t', header=False, index=False)

        with open("label_RAND.txt", 'w') as f:
            f.writelines([w + "\n" for w in y])
