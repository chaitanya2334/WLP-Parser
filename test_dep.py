import csv

import os
from nltk.parse.stanford import StanfordDependencyParser

from corpus.Manager import Manager
from preprocessing.feature_engineering.pos import PosTagger

import features_config as feat_cfg


def write_csv_results(csv_filepath, title, overwrite=True):
    # if file is empty or you want to overwrite the file, write the headers.
    if os.stat(csv_filepath).st_size == 0 or overwrite:
        with open(csv_filepath, 'w') as csvfile:
            fieldnames = ['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_filepath, 'a') as csvfile:
        fieldnames = ['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Method': title,
                         'Accuracy': 1,
                         'Precision': 2,
                         'Recall': 3,
                         'F1-Score': 4})


if __name__ == '__main__':
    write_csv_results("test.csv", "asdf", overwrite=True)
    write_csv_results("test.csv", "dddd", overwrite=True)
