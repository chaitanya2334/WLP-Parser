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
ARTICLES_FOLDERPATH = os.path.join(CURRENT_DIR, "input/Chaitanya")

CORPUS_FOLDERPATH = os.path.join(CURRENT_DIR, "input")

# Access to all protocols (unannotated)
COMMON_FOLDERPATH = os.path.join(CURRENT_DIR, "input/Common")

PUBMED_AND_PMC_W2V_BIN = os.path.join(CURRENT_DIR, "preprocessing/PubMed-and-PMC-w2v.bin")

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100
PF_EMBEDDING_DIM = 5

NUMBER_OF_VALIDATION_SAMPLES = 200

LEARNING_RATE = 0.3

LSTM_HIDDEN_SIZE = 200

MAX_EPOCH = 100

VERBOSE = False

BATCH_SIZE = 1

POSITIVE_LABEL = 'Action-Verb'
NEG_LABEL = 'O'

CATEGORIES = 3

OOV_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/oov.txt")

MODEL_SAVE_FILEPATH = os.path.join(CURRENT_DIR, "save.m")

FILTER_ALL_NEG = True

MAX_EPOCH_IMP = 20

TRAIN_PERCENT = 60

DEV_PERCENT = 20

TEST_PERCENT = 20

LM_HIDDEN_SIZE = 50

LM_MAX_VOCAB_SIZE = 7500

LM_GAMMA = 0.1

REPLACE_DIGITS = True

NUM_TO_D = True

PUBMED_VOCAB_FILE = os.path.join(CURRENT_DIR, "preprocessing/pubmed_vocab.txt")

CHAR_EMB_DIM = 50

CHAR_RECURRENT_SIZE = 200

CHAR_VOCAB = 0
CHAR_LEVEL = "Input"

WORD_START = "<w>"
WORD_END = "</w>"
SENT_START = "<s>"
SENT_END = "</s>"

UNK = "<unk>"

RANDOM_TRAIN = False

CLIP = 20

BEST_MODEL_SELECTOR = "dev_conll_f"

def ver_print(string, value):
    if VERBOSE:
        print(string + ':\n {0}'.format(value))
