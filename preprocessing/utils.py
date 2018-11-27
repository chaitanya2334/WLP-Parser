import io
import math
import zipfile

import numpy as np
import pickle
import os

import requests
from nltk.parse.stanford import StanfordDependencyParser
from tqdm import tqdm

from utils import download


def generate_pf_mat(n):
    def gen_row(i, length):
        row = [abs(j - i) for j in range(length)]
        return row

    pf_mat = []
    for x in range(n):
        pf_mat.append(gen_row(x, n))

    pf_mat = np.array(pf_mat)
    return pf_mat


def quicksave(items, file):
    pickle.dump(items, open(file, "wb"))


def quickload(file):
    return pickle.load(open(file, "rb"))


def touch(fname, times=None):
    with open(fname, 'w'):
        os.utime(fname, times)


def get_stanford_dep_parser(path_to_jar, path_to_models_jar):
    def decorator(fn):
        def decorated(*args, **kwargs):
            try:
                dep = StanfordDependencyParser(path_to_jar=path_to_jar,
                                               path_to_models_jar=path_to_models_jar,
                                               java_options="-mx3000m")
            except(pickle.UnpicklingError, EOFError, FileNotFoundError, TypeError, LookupError):
                print("Downloading Stanford Parser ...")
                url = "https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip"
                dirpath = os.path.dirname(os.path.dirname(path_to_jar))
                download(url, dirpath)

                dep = StanfordDependencyParser(path_to_jar=path_to_jar,
                                               path_to_models_jar=path_to_models_jar,
                                               java_options="-mx3000m")

            kwargs['dep_parser'] = dep
            return fn(*args, **kwargs)

        return decorated

    return decorator
