import io
import math
import zipfile

import numpy as np
import pickle
import os

import requests
from nltk.parse.stanford import StanfordDependencyParser
from tqdm import tqdm


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
                r = requests.get(url, stream=True)
                total_size = int(r.headers.get('content-length', 0))
                block_size = 1024
                pbar = tqdm(r.iter_content(chunk_size=block_size),
                            total=total_size, unit_divisor=1024,
                            unit='B', unit_scale=True)
                with io.BytesIO() as buf:
                    for chunk in pbar:
                        buf.write(chunk)
                        buf.flush()
                        pbar.update(block_size)

                    buf.seek(0, 0)

                    z = zipfile.ZipFile(buf)
                    dirpath = os.path.dirname(os.path.dirname(path_to_jar))
                    z.extractall(dirpath)
                    z.close()

                dep = StanfordDependencyParser(path_to_jar=path_to_jar,
                                               path_to_models_jar=path_to_models_jar,
                                               java_options="-mx3000m")

            kwargs['dep_parser'] = dep
            return fn(*args, **kwargs)

        return decorated

    return decorator
