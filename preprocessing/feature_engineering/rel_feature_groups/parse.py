from nltk import ParentedTree

from corpus.ProtoFile import Relation
from preprocessing.feature_engineering.datasets import RelationWindow


class ParseFeatureGroup(object):

    def __init__(self):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel, Relation)
            result.append([self.cphbnull(rel),  # combination of mention entity types
                           ])

        # print("done")
        return result

    @staticmethod
    def get_words(tokens):
        return [token.word for token in tokens]

    @staticmethod
    def get_lca_length(location1, location2):
        i = 0
        while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
            i += 1
        return i

    @staticmethod
    def get_labels_from_lca(ptree, lca_len, location):
        labels = []
        for i in range(lca_len, len(location)):
            labels.append(ptree[location[:i]].label())
        return labels

    def find_path(self, ptree, text1, text2):
        leaf_values = ptree.leaves()
        leaf_index1 = leaf_values.index(text1)
        leaf_index2 = leaf_values.index(text2)

        location1 = ptree.leaf_treeposition(leaf_index1)
        location2 = ptree.leaf_treeposition(leaf_index2)

        # find length of least common ancestor (lca)
        lca_len = self.get_lca_length(location1, location2)

        # find path from the node1 to lca

        labels1 = self.get_labels_from_lca(ptree, lca_len, location1)
        # ignore the first element, because it will be counted in the second part of the path
        result = labels1[1:]
        # inverse, because we want to go from the node to least common ancestor
        result = result[::-1]

        # add path from lca to node2
        result = result + self.get_labels_from_lca(ptree, lca_len, location2)
        return result

    def ptp(self, rel):

        ptree = ParentedTree.fromstring(rel.parse_tree)
        print(ptree.pprint())
        arg1_tokens = rel.get_arg1_tokens()
        arg1_words = self.get_words(arg1_tokens)
        arg2_tokens = rel.get_arg2_tokens()
        arg2_words = self.get_words(arg2_tokens)

        return "ptp={0}".format(self.find_path(ptree, arg1_words[-1], arg2_words[-1]))

    def ptph(self, rel):
        ptree = ParentedTree.fromstring(rel.parse_tree)
        print(ptree.pprint())
        arg1_tokens = rel.get_arg1_tokens()
        arg1_words = self.get_words(arg1_tokens)
        arg2_tokens = rel.get_arg2_tokens()
        arg2_words = self.get_words(arg2_tokens)
        return "ptp={0}".format(self.find_path(ptree, arg1_words[-1], arg2_words[-1]))