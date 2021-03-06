# -*- coding: utf-8 -*-
"""File containing various constants used throughout the program."""
import os

# directory of the config file
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# filepath to the corpus
# The corpus must have one article/document per line.
# Each named entity must be tagged in the form word/LABEL, e.g.
#   John/PER Doe/PER did something yesterday. Then he did something else.
#   Washington/LOC D.C./LOC is the capital of the U.S.
#   ....
ARTICLES_FOLDERPATH = os.path.join(CURRENT_DIR, "simple_input/")


# filepath to a unigrams file generated by the script in preprocessing/collect_unigrams.py
UNIGRAMS_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/unigrams.txt")

UNIGRAMS_FOLDERPATH = os.path.join(CURRENT_DIR, "preprocessing/labels")

# filepath to a unigrams file (unigrams of person names) generated by the script
# in preprocessing/collect_unigrams.py
# UNIGRAMS_PERSON_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/unigrams_per.txt")

# number of words to skip in the list of all unigrams for the CRF training,
# e.g. a value of 100 means that during feature generation no feature will be generated
# for the 100 most common words (except for "not in unigrams list" feature)
UNIGRAMS_SKIP_FIRST_N = 150

# maximum number of words to use from the list of all unigrams during CRF training,
# e.g. a value of 100 means that the unigrams list will be filled with the 100 most common words
# (assuming UNIGRAMS_SKIP_FIRST_N was set to 0). All other words will not be part of the unigrams
# list and will get the feature "not in unigrams list".
UNIGRAMS_MAX_COUNT_WORDS = 4200

# name of the file containing the LDA's dictionary/vocabulary, as generated by
# preprocessing/lda.py
LDA_DICTIONARY_FILENAME = "lda_dictionary"

# filepath to the file containing the LDA's dictionary/vocabulary, as generated by
# preprocessing/lda.py
LDA_DICTIONARY_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/" + LDA_DICTIONARY_FILENAME)

# name of the file containing the LDA's trained model, as generated by preprocessing/lda.py
LDA_MODEL_FILENAME = "lda_model"

# filepath to the file containing the LDA's trained model, as generated by preprocessing/lda.py
LDA_MODEL_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/" + LDA_MODEL_FILENAME)

# filepath to the file containing the LDA's cache, generated during the CRF training
LDA_CACHE_FILEPATH = os.path.join(CURRENT_DIR, "lda.cache")

# window size used during the LDA training and during the feature generation (left size of window)
LDA_WINDOW_LEFT_SIZE = 5

# window size used during the LDA training and during the feature generation (right size of window)
LDA_WINDOW_RIGHT_SIZE = 5

# window size used during the LDA training and during the feature generation (total size of window)
LDA_WINDOW_SIZE = LDA_WINDOW_LEFT_SIZE + 1 + LDA_WINDOW_RIGHT_SIZE

# number of topics of the LDA
LDA_COUNT_TOPICS = 100

# filepath to the directory containing the stanford POS tagger
STANFORD_POS_DIR = os.path.join(CURRENT_DIR, "preprocessing/feature_engineering/stanford-postagger-full-2017-06-09")

STANFORD_PARSER_PATH = os.path.join(CURRENT_DIR, "preprocessing/feature_engineering/stanford-parser-full-2017-06-09")

STANFORD_PARSER_JAR = os.path.join(STANFORD_PARSER_PATH, "stanford-parser.jar")

STANFORD_PARSER_MODEL_JAR = os.path.join(STANFORD_PARSER_PATH, "stanford-parser-3.8.0-models.jar")

# filepath to the jar of the stanford POS tagger
STANFORD_POS_JAR_FILEPATH = os.path.join(STANFORD_POS_DIR, "stanford-postagger-3.8.0.jar")

# filepath to the used model of the stanford POS tagger (in subdirectory "models/" in the stanford
# pos tagger's directory)
STANFORD_MODEL_FILEPATH = os.path.join(STANFORD_POS_DIR, "models/english-bidirectional-distsim.tagger")

GENIA_TAGGER_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/feature_engineering/geniatagger-3.0.2/geniatagger")

# filepath to the cache to use for the pos tagger during training of the CRF
POS_TAGGER_CACHE_FILEPATH = os.path.join(CURRENT_DIR, "pos.cache")

# filepath to the w2v clusters file as genreated by the word2vec tool
W2V_CLUSTERS_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/output_word2vec.txt")

# filepath to a 'paths' file generated by Percy Liang's brown clustering tool
BROWN_CLUSTERS_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/input-c1000-p1.out/paths")

# window size of each example to train on
WINDOW_SIZE = 20

# how many words to the left of a word will be part of the feature set of a word,
# e.g. if set to >=1 and the word 1 left of a word W has the feature "w2v=123" then W will get a
# featur "-1:w2v=123".
SKIPCHAIN_LEFT = 0

# see SKIPCHAIN_LEFT, just to the right
SKIPCHAIN_RIGHT = 0

# maximum number of optimizer iterations during training of the CRF (if set to None the optimizer
# will decide when to quit)
MAX_ITERATIONS = None

# Number of windows to use during training (offset is COUNT_WINDOWS_TEST, i.e. test windows will
# be loaded first)
COUNT_WINDOWS_TRAIN = 10000

# Number of windows to use during testing
COUNT_WINDOWS_TEST = 0

FEATURES_PICKLE = os.path.join(CURRENT_DIR, "preprocessing/features_pickle.p")

# Label for any word that has no named entity label
NO_NE_LABEL = "O"

# labels to accept when parsing data, all other labels will be treated as normal text
# e.g. in "Manhatten/NY" the "NY" will not be treated as a label and the full token
# "Manhatten/NY" will be loaded as one word
LABELS = ["Action-Verb", "Reagent", "Location", "Device", "Mention", "Method", "Seal", "Modifier", "Numerical", "Measure-Type",
          "Unit", "Quantity", "Concentration", "Time", "Tool", "Temperature", "Rpm"]

# Whether to remove parts of the BIO encoding, specifically whether to remove the "B-" and "I-"
# parts, e.g. "B-PER" or "I-LOC" will become "PER" and "LOC" if set to True.
# This happens before checking whether a label is contained in LABELS.
REMOVE_BIO_ENCODING = True

OUTPUT_CONLL_FILE = os.path.join(CURRENT_DIR, "postprocessing/conll_output.txt")

OUTPUT_BRAT_TRUE_FILEPATH = os.path.join(CURRENT_DIR, "output/true/brat_out")
OUTPUT_BRAT_PRED_FILEPATH = os.path.join(CURRENT_DIR, "output/pred/brat_out")

OUTPUT_BRAT_TRUE_INC_FILEPATH = os.path.join(CURRENT_DIR, "output/true/brat_inc_out")
OUTPUT_BRAT_PRED_INC_FILEPATH = os.path.join(CURRENT_DIR, "output/pred/brat_inc_out")

TRAIN_FILEPATH = os.path.join(CURRENT_DIR, "output/train.txt")
DEV_FILEPATH = os.path.join(CURRENT_DIR, "output/dev.txt")
TEST_FILEPATH = os.path.join(CURRENT_DIR, "output/test.txt")

POSITIVE_LABEL = 'Action-Verb'

PICKLE_SAVE_FILEPATH = os.path.join(CURRENT_DIR, "save.p")
