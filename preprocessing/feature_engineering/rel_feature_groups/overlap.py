from corpus.ProtoFile import Tag, Relation
from preprocessing.feature_engineering.datasets import RelationWindow
import config as cfg

class OverlapFeatureGroup(object):
    def __init__(self, ):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel, Relation)
            result.append([self.mb(rel),  # combination of mention entity types
                           self.wb(rel),
                           ])

        # print("done")
        return result

    def mb(self, rel):
        tb = rel.get_tokens_bet()
        count = len(tb)
        for token in tb:
            if token.label == cfg.NEG_LABEL:
                count -=1

        return "#mb={0}".format(count)

    def wb(self, rel):
        tb = rel.get_tokens_bet()
        count = len(tb)
        return "#wb={0}".format(count)