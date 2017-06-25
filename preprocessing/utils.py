import numpy as np


def generate_pf_mat(n):
    def gen_row(i, length):
        row = [abs(j - i) for j in range(length)]
        return row

    pf_mat = []
    for x in range(n):
        pf_mat.append(gen_row(x, n))

    pf_mat = np.array(pf_mat)
    return pf_mat


