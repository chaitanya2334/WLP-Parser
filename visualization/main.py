import os

from visualization.model_read import model_read_word_emb, model_read_pos
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

model_save_path_pos_input = os.path.join(cfg.MODEL_SAVE_DIR,
                                         "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.3_CH_Input_POS_Yes_DEP_L_No_DEP_W_No.m")
model_save_path_pos_attention = os.path.join(cfg.MODEL_SAVE_DIR,
                                             "ONLY_POS_T_SIZE_100_R_2_WE_pre_and_post_G_0.6_CH_Attention_POS_Yes_DEP_L_No_DEP_W_No.m")
model_save_path_pos_none = os.path.join(cfg.MODEL_SAVE_DIR,
                                        "ONLY_POS_T_SIZE_100_R_0_WE_pre_and_post_G_0.4_CH_None_POS_Yes_DEP_L_No_DEP_W_No.m")


def my_tsne(save_path):
    X, y = model_read_word_emb(save_path, cfg.DB_WITH_POS)
    print(X.shape)
    simple_vis_tsne(X, y)


if __name__ == '__main__':
    gen_image = False
    model_path = model_save_path_pos_none

    if gen_image:
        my_tsne(model_path)
    else:
        X, y = model_read_pos(model_path, cfg.DB_WITH_POS)

        df = pd.DataFrame(X)

        df.to_csv("visualization/data/POS_none.csv", sep='\t', header=False, index=False)

        with open("visualization/data/label_POS_none.txt", 'w') as f:
            f.writelines([w + "\n" for w in y])
