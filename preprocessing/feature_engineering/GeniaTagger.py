import io
import math
import os
import subprocess
import tarfile

import psutil
import requests
from tqdm import tqdm

import utils


class Tag(object):
    def __init__(self, t):
        if len(t) == 5:
            self.word = t[0]
            self.base = t[1]
            self.pos = t[2]
            self.chunk = t[3]
            self.ner = t[4]


class GeniaTagger(object):
    """
    """

    def __init__(self, path_to_tagger):
        """

        Arguments:
        - `path_to_tagger`:
        """
        self._path_to_tagger = path_to_tagger
        self._dir_to_tagger = os.path.dirname(path_to_tagger)

        if not os.path.isfile(self._path_to_tagger):
            self.dl_and_make()

    def dl_and_make(self):
        print("Downloading genia tagger ...")
        r = requests.get("http://www.nactem.ac.uk/tsujii/GENIA/tagger/geniatagger-3.0.2.tar.gz", stream=True)
        total_size = int(r.headers.get('Content-length'))
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
            z = tarfile.open(fileobj=buf)
            z.extractall(os.path.dirname(self._dir_to_tagger))

        subprocess.call('make',
                        cwd=self._dir_to_tagger,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    @staticmethod
    def kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def parse_through_file(self, sents):
        with open("temp.txt", "w", encoding="utf-8") as f:
            for sent in sents:
                f.write(sent + "\n")

        in_file = open("temp.txt", "r", encoding="utf-8")
        out_file = open("out.txt", "w", encoding="utf-8")
        subprocess.call(self._path_to_tagger, cwd=self._dir_to_tagger,
                        stdin=in_file, stdout=out_file,
                        stderr=subprocess.PIPE)
        ret_sents = []
        sent = []

        with open("out.txt", "r", encoding="utf-8") as out:
            for line in out.readlines():
                if line == '\n':
                    ret_sents.append(sent)
                    sent = []
                else:
                    word_tag = Tag(line.split("\t"))
                    sent.append((word_tag.word, word_tag.pos, word_tag.chunk))

        in_file.close()
        out_file.close()
        return ret_sents

    def parse(self, sents):
        """

        Arguments:
        - `self`:
        - `text`:
        """

        results = list()

        for oneline in tqdm(sents):
            self._tagger.stdin.write((oneline + "\n").encode("utf-8"))
            r = self._tagger.stdout.readline()[:-1]
            if not r:
                break
            results.append(tuple(r.split('\t')))
        return results
