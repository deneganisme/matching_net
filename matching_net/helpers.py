#! /usr/bin/env python
# -*- coding: utf-8 -*-
# trainer.py
#
######################################################################################
#
# Helper functions for MatchingNet
#
########################################################################################

"""
==========================================================================

Authors: Dennis Egan <d.james.egan@gmail.com>

==========================================================================

"""


############################################
# Modules
#############################################

import numpy as np

np.random.seed(7)  # Lucky number 7

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from typing import List, Dict, Tuple, Set


def vectorize_data(data: List[Tuple[str, str]],
                   inputs_idx: Dict[str, int],
                   outputs_index: Dict[str, int],
                   input_maxlen: int,
                   output_maxlen: int,
                   unk_value: str='UNK')->Tuple[np.array, np.array]:
    """
    Vectorize data for rem using mapping
    :param data: list of tuples containing (sentence, relation)
    :param inputs_idx: mapping of words in the input to a unique index e.g. "the" -> 101
    :param outputs_index: mapping of words in the output to a unique index e.g. "unknown" -> 2
    :param input_maxlen: max length for input to determine how to pad inputs
    :param output_maxlen: max length for output to determine how to pad outputs
    :param one_hot: determines of outputs are one hot vectors
    :return:
    """

    X = []
    Y = []
    for sentence, relation in data:
        words = sentence.lower().split()

        x = []
        for word in words:
            try:
                x.append(inputs_idx[word])

            # UNK value
            except KeyError:
                x.append(inputs_idx[unk_value])
                pass

        y = np.array(range(len(outputs_index.keys())))
        y[:] = 0
        y[outputs_index[relation.lower()]] = 1

        X.append(x)
        Y.append(y)

    X = pad_sequences(X, maxlen=input_maxlen, padding='pre')
    Y = np.array(Y)
    # Y = np.array(Y) if one_hot else pad_sequences(Y, maxlen=output_maxlen, padding='post')

    return X, Y


def create_vocabulary(sentences: List[str], split_words: bool=True)->Tuple[Set[str], int]:
    """
    Generate a vocabulary from a list of sentences
    :param sentences:
    :param split_words: if True, splits sentence into words else use whole sentence as index (used for relations)
    :return:
    """
    vocabulary = set()
    max_length = -1

    for sentence in sentences:

        if split_words:
            words = sentence.lower().split()
            vocabulary.update(words)

            if len(words) > max_length:
                max_length = len(words)

        else:
            vocabulary.add(sentence.lower())
            max_length = 1

    return vocabulary, max_length


def generate_vocabularies(data: List[Tuple[str, str]],
                          max_len_pct:float=0.9,
                          split_relation:bool = False)->Tuple[Set[str], Set[str], int, int]:
    """
    Generate input and output vocabulary from data
    :param data: list of tuples (input, output)
    :param max_len_pct: Percentile at which to set max length for input
    :return:
    """

    data_df = pd.DataFrame(data, columns=["input", "output"])
    inputs = data_df['input'].unique()
    input_vocabulary, max_input_length = create_vocabulary(inputs)

    if max_len_pct is not None:
        if max_len_pct < 1:
            max_len_pct = int(max_len_pct * 100)

        data_df['word_count'] = data_df['input'].apply(lambda x: len(x.split()))
        description = data_df.describe(percentiles=np.linspace(0, 1, num=101))

        max_input_length = description.loc["{}%".format(max_len_pct), "word_count"]

    outputs = data_df['output'].unique()
    output_vocabulary, max_output_length = create_vocabulary(outputs, split_words=split_relation)

    return input_vocabulary, output_vocabulary, int(max_input_length), int(max_output_length)


def split_data(data_df, split_pct=0.8):
    """
    Split a DF into two parts based on split_pct
    """
    # Shuffle and split
    split_idx = int(len(data_df) * split_pct)

    data_df  = data_df.sample(frac=1.).reset_index(drop=True)
    train_df = data_df.iloc[:split_idx]
    test_df  = data_df.iloc[split_idx:]

    return train_df, test_df


def get_vocabularies(data: List[Tuple[str, str]]):
    """
    Get input word2index (word2index) and output word2index (rel2index) of a data set as well as the length of inputs
    and outputs.
    Used to create inputs/outputs of the model
    NOTE: if classification, rel2index = class labels and outptu_length =1
    :param data:
    :return:
    """
    vocabulary, relation_vocabulary, input_length, output_length = generate_vocabularies(data=data)

    word2index = dict(zip(vocabulary, range(2, len(vocabulary) + 2)))
    word2index['PAD'] = 0
    word2index['UNK'] = 1

    rel2index = dict(zip(relation_vocabulary, range(len(vocabulary))))

    return word2index, rel2index, input_length, output_length
