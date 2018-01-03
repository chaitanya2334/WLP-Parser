from corpus.ProtoFile import Relation
from preprocessing.feature_engineering.datasets import RelationWindow


class DependencyFeatureGroup(object):
    def __init__(self):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel, Relation)
            result.append([self.et1dw1(rel),  # combination of mention entity types
                           self.et2dw2(rel),
                           self.h1dw1(rel),
                           self.h2dw2(rel),
                           self.et12SameNP(rel),
                           self.et12SamePP(rel),
                           self.et12SameVP(rel)
                           ])

        # print("done")
        return result

    @staticmethod
    def get_words(tokens):
        return [token.word for token in tokens]

    def et1dw1(self, rel):
        et = rel.arg1_tag.tag_name
        dep = rel.arg1_deps()

        return "et1dw1={0}{1}".format(et, dep)

    def et2dw2(self, rel):
        et = rel.arg2_tag.tag_name
        dep = rel.arg2_deps()

        return "et2dw2={0}{1}".format(et, dep)

    def h1dw1(self, rel):
        arg1_tokens = rel.get_arg1_tokens()
        words = self.get_words(arg1_tokens)
        h1 = words[-1]
        dep = rel.arg1_deps()

        return "h1dw1={0}{1}".format(h1, dep)

    def h2dw2(self, rel):
        arg2_tokens = rel.get_arg2_tokens()
        words = self.get_words(arg2_tokens)
        h2 = words[-1]
        dep = rel.arg2_deps()

        return "h1dw1={0}{1}".format(h2, dep)

    @staticmethod
    def et12(rel):
        return "et12={0}".format("_".join([rel.arg1_tag.tag_name, rel.arg2_tag.tag_name]))

    def et12SameNP(self, rel):
        et12 = self.et12(rel)

        return "et12SameNP={0}_{1}".format(et12, rel.sameNP())

    def et12SamePP(self, rel):
        et12 = self.et12(rel)

        return "et12SamePP={0}_{1}".format(et12, rel.samePP())

    def et12SameVP(self, rel):
        et12 = self.et12(rel)

        return "et12SameVB={0}_{1}".format(et12, rel.sameVP())



