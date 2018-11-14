# -*- coding: utf-8 -*-
"""
Contains:
    1. Various classes (feature generators) to convert windows (of words/tokens) to feature values.
       Each feature value is a string, e.g. "starts_with_uppercase=1", "brown_cluster=123".
    2. A method to create all feature generators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals


# All capitalized constants come from this file
import features_config as cfg

from preprocessing.feature_engineering.rel_feature_groups import words
from preprocessing.feature_engineering.rel_feature_groups.chunk import ChunkFeatureGroup
from preprocessing.feature_engineering.rel_feature_groups.dep import DependencyFeatureGroup
from preprocessing.feature_engineering.rel_feature_groups.entity import EntityFeatureGroup
from preprocessing.feature_engineering.rel_feature_groups.overlap import OverlapFeatureGroup
from preprocessing.feature_engineering.rel_feature_groups.parse import ParseFeatureGroup
from preprocessing.feature_engineering.rel_feature_groups.words import WordFeatureGroup

from preprocessing.feature_engineering.unigrams import Unigrams


def create_features(verbose=True):
    """This method creates all feature generators.
    The feature generators will be used to convert windows of tokens to their string features.

    This function may run for a few minutes.

    Args:
        verbose: Whether to output messages.
    Returns:
        List of feature generators
        :param verbose: prints stuff if true
        :param articles: list of bit lengths that will be used as features
    """

    # create feature generators
    result = [
        WordFeatureGroup(),
        EntityFeatureGroup(),
        OverlapFeatureGroup(),
        ChunkFeatureGroup(),
        DependencyFeatureGroup(),
        ParseFeatureGroup(),
    ]

    return result



