import numpy as np
import pickle
import os


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
