import glob
import os

from corpus.Manager import WLPDataset
import config as cfg
import fileinput

def rename_tag(fpaths, ext=".ann", replace_from="Action-Verb", replace_to="Action", times=None):
    for fpath in fpaths:
        p = os.path.dirname(fpath)
        title, _ = os.path.splitext(os.path.basename(fpath))
        full_path = os.path.join(p, title + ext)
        print(full_path)
        with fileinput.FileInput(full_path, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(replace_from, replace_to), end='')


if __name__ == '__main__':
    corpus = WLPDataset()

    files = glob.glob(cfg.ARTICLES_FOLDERPATH + "/*.ann")
    rename_tag(files)
