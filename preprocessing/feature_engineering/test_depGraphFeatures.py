from unittest import TestCase

from nltk.parse.stanford import StanfordDependencyParser
import features_config as cfg

from preprocessing.feature_engineering.features import DepGraphFeatures


class TestDepGraphFeatures(TestCase):
    def test_dependency_parse(self):
        sent = ['First', 'In', 'the', 'Beckman', 'Ti45', '1', 'hr', '35K', 'and', 'second', 'PEG', 'ppt', 'using',
                '1111', '11111111' 'PEG', '6K', '0.5', 'M', 'NaCl', 'Yamamoto', '1970', 'Virology', 'aswdf', 'asdf']
        dep_p = StanfordDependencyParser(path_to_jar=cfg.STANFORD_PARSER_JAR,
                                         path_to_models_jar=cfg.STANFORD_PARSER_MODEL_JAR)
        dep = DepGraphFeatures(dep_p)
        dep.dep_parser.raw_parse(" ".join(sent))

    def test_ex(self):
        def digits(str):
            n = ''.join(x for x in str if x.isdigit())
            return n

        print(digits('0.22-0.45'))