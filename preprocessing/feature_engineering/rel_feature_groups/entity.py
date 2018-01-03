from corpus.ProtoFile import Tag
from preprocessing.feature_engineering.datasets import RelationWindow


# TODO test class
class EntityFeatureGroup(object):
    def __init__(self):
        pass

    def convert_window(self, window):
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel.arg1_tag, Tag)
            result.append([self.et12(rel),  # combination of mention entity types
                           ])

        # print("done")
        return result

    @staticmethod
    def et12(link):
        return "et12={0}".format("_".join([link.arg1_tag.tag_name, link.arg2_tag.tag_name]))
