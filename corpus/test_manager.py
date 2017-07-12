from unittest import TestCase

from torch.utils.data import DataLoader

from corpus.Manager import Manager
from preprocessing.text_processing import prepare_embeddings
import config as cfg


class TestManager(TestCase):
    def test_pytorch_dataloader(self):
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        dataset = Manager(word_index, char_index)

        def collate(batch):
            return batch

        data = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=collate)

        for i, item in enumerate(data):
            if i > 10:
                break
            item = item[0]
            print(dataset.to_words(item.X), item.Y, item.P, item.C, item.F)


