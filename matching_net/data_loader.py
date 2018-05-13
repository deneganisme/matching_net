#! /usr/bin/env python
# -*- coding: utf-8 -*-
# trainer.py
#
######################################################################################
#
# Data loader/handler to run an experiment with Matching Net
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
from typing import Union, List, Tuple
from matching_net.helpers import vectorize_data, split_data, get_vocabularies


#############################################
# Logging for Module
#############################################


import logging
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger_name = "{}_logger".format(str(__file__).replace("\\", "/").split("/")[-1].replace(".py", ""))
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)


class DataLoader(object):
    """
    Data handler for Matching Net
    """
    def __init__(self,
                 data_df: pd.DataFrame,
                 split_pct: float=0.8,
                 input_col: str='sentence',
                 target_col: str='relation',
                 classes_per_batch: int=15,
                 n_shot: int=5):

        self.data_df = data_df
        self.input_col = input_col
        self.target_col = target_col
        self.n_shot = n_shot
        self.split_pct = split_pct
        self.classes_per_batch = classes_per_batch

        self.data = self._data_from_df(self.data_df)

        self.word2index, self.rel2index, self.input_size, self.output_size = get_vocabularies(self.data)

        self.vocabulary_size = len(self.word2index.keys())
        self.num_relations = len(self.rel2index.keys())

        self.__create_data()

    def __create_data(self):

        self.data_df[self.input_col] = self.data_df[self.input_col].str.lower()
        self.data_df[self.target_col] = self.data_df[self.target_col].str.lower()

        self.data_df = self.data_df.drop_duplicates(subset=[self.input_col])

        assert len(self.data_df) == len(self.data_df[self.input_col].unique()), \
            (len(self.data_df), len(self.data_df[self.input_col].unique()))

        train_df, test_df = split_data(data_df=self.data_df, split_pct=self.split_pct)

        test_inputs = set(test_df[self.input_col].values.tolist())
        train_inputs = set(train_df[self.input_col].values.tolist())

        intersect = train_inputs.intersection(test_inputs)
        assert len(intersect) == 0, f"Found {len(intersect)} overlapping examples in training and testing"

        assert all(input_ not in train_df[self.input_col].values for input_ in test_df[self.input_col].values)

        train_df = train_df.drop_duplicates([self.input_col, self.target_col])
        test_df  = test_df.drop_duplicates([self.input_col, self.target_col])

        train_df[self.target_col] = train_df[self.target_col].str.lower()
        test_df[self.target_col]  = test_df[self.target_col].str.lower()

        # Drop relations below n-shot count
        to_drop = []
        train_tgt_counts = train_df[self.target_col].value_counts().to_dict()
        test_tgt_counts  = test_df[self.target_col].value_counts().to_dict()

        # Are there any relations in test that are not in train? if so, drop them
        in_train = set(list(train_tgt_counts.keys()))
        in_test  = set(list(test_tgt_counts.keys()))
        diff = in_train.difference(in_test)
        diff.update(in_test.difference(in_train))

        to_drop.extend(list(diff))

        for relation, count in train_tgt_counts.items():
            if count < self.n_shot:
                to_drop.append(relation)

            elif relation in test_tgt_counts.keys():
                if test_tgt_counts[relation] < 2:
                    to_drop.append(relation)

        to_drop = list(set(to_drop))
        logger.warning(f"{len(to_drop)} targets being dropped for being < n_shot value ({self.n_shot})")

        self.train_df = train_df.loc[~train_df[self.target_col].isin(to_drop), :].reset_index(drop=True)
        self.test_df  = test_df.loc[~test_df[self.target_col].isin(to_drop), :].reset_index(drop=True)

        self.train_data = self._data_from_df(self.train_df)
        self.test_data = self._data_from_df(self.test_df)

        self.__mode = 'train'

        self.__train_index = 0
        self.train_batches = self.get_batches(data=self.train_data,
                                              for_testing=False,
                                              classes_per_batch=self.classes_per_batch)

        self.__test_index = 0
        self.test_batches = self.get_batches(data=self.test_data,
                                             for_testing=True,
                                             classes_per_batch=self.classes_per_batch)

    def to_dict(self):
        return {
            'train_batches': self.train_batches,
            'test_batches': self.test_batches,
            'word2index': self.word2index,
            'rel2index': self.rel2index,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'n_shot': self.n_shot,
            'input_col': self.input_col,
            'target_col': self.target_col,
            'train_df': self.train_df,
            'test_df': self.test_df,
            'classes_per_batch': self.classes_per_batch
        }

    @classmethod
    def from_dict(cls, value):

        for k, v in value.items():
            setattr(cls, k, v)

        return cls

    @classmethod
    def from_csv(cls, filepath):
        return cls(data_df=pd.read_csv(filepath))

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'test']
        if value == self.__mode:
            pass
        else:
            self.__mode = value
            self.__train_index = 0
            self.__test_index = 0

    def shuffle(self):
        np.random.shuffle(self.train_batches)

    def convert_input(self, input_: List[int])->List[str]:
        """
        Convert input from ints -> words
        e.g. convert_input([45, 68, 93]) = ["hey", "how", "nice"]
        :param input_:
        :return:
        """
        index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
        return [index2word[i] for i in input_]

    def convert_pred(self, prediction: List[int])->str:
        """
        Convert a One-Hot or scalar from int -> word
        :param prediction:
        :return:
        """

        index2rel = dict(zip(self.rel2index.values(), self.rel2index.keys()))

        # Scalar
        if isinstance(prediction, (float, int)):
            converted = index2rel[int(prediction)]

        # One-Hot
        else:
            idx = np.argmax(prediction)
            converted = index2rel[int(idx)]

        return converted

    def get_vocabularies(self):
        return {
            'word2index': self.word2index,
            'rel2index': self.rel2index
        }

    def get_sizes(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size
        }

    def _data_from_df(self, df: pd.DataFrame):
        return df[[self.input_col, self.target_col]].values

    def vectorize(self, data: List[Tuple[str, str]])->Tuple[np.array, np.array]:
        """
        Transform [(sentence_i, relation_i), ... ] into vectors
        """
        return vectorize_data(data=data,
                              inputs_idx=self.word2index,
                              outputs_index=self.rel2index,
                              input_maxlen=self.input_size,
                              output_maxlen=self.output_size)

    def get_batches(self,
                    data,
                    classes_per_batch: Union[int, None]=None,
                    examples_per_class: int=1,
                    example_in_set: bool=True,
                    for_testing: bool=False):
        """
        Generate n-shot learning batch. See section 2.2 of paper
        :param generator: if True, yields each batch
        :param classes_per_batch: Number of classes to use as adversarial examples
        :param examples_per_class: Number of examples for each class per batch
        :param example_in_set: if True, an example of the holdout set is in the class
        :param for_testing: if True, lax restraints on data
        :return:
        """

        # inputs, labels = zip(*data)
        if classes_per_batch is None:
            classes_per_batch = len(self.rel2index.keys()) - 1

        # Pre-processing -- turn data into pd.DataFrame for ease of handling
        data_df = pd.DataFrame(data, columns=['input', 'label'])
        data_df['input'] = data_df['input'].str.lower()
        data_df['label'] = data_df['label'].str.lower()

        # If no unknown model converges very quickly
        unique_labels = list(data_df['label'].unique())

        # Model converges quickly if no unknowns
        # unique_labels = list(set(unique_labels).difference({'pad unknown pad'})
        num_labels = len(unique_labels)

        label_counts = data_df['label'].value_counts().to_dict()
        min_label, min_count = min(label_counts.items(), key=lambda x: x[1])

        if classes_per_batch > num_labels:
            if for_testing:
                logger.warning(f"\nNumber of classes exceeds actual number of classes."
                               f"\nSetting number of classes to actual number={num_labels}")

                classes_per_batch = num_labels

            else:
                raise ValueError("Number of classes exceeds actual number of classes.")

        if examples_per_class + 1 > min_count:
            if for_testing:
                logger.warning(f"\nNumber of examples exceeds minimum count for labels."
                               f"\nSetting examples per class to min_count={min_count}")
                examples_per_class = min_count
                assert examples_per_class > 1, "Too few examples. Need at least 2."

            else:
                raise ValueError("Number of examples exceeds minimum count for labels.")

        # Calculate number of points per task and from that determine the number of batches
        points_per_task = (examples_per_class * (classes_per_batch+1)) + 1
        number_of_batches = int(len(data_df) / points_per_task)

        if self.n_shot > number_of_batches:
            raise ValueError("Number of batches less than n_shot.")

        # n_shot determines how much each class can be seen therefore iterate over its range
        batches = []
        for _ in range(self.n_shot):

            # Shuffle between n-shots
            np.random.shuffle(unique_labels)
            for label in unique_labels:

                # Number of examples for our predictive class == y_hat (+ 1 for x_hat and y_hat)
                example_sample_count = examples_per_class + 1

                # Pick a random example (where != unknown)
                given_example = data_df[data_df['label'] == label].sample(n=example_sample_count,
                                                                          replace=False).reset_index(drop=True)

                # The values we'll actually predict
                x_hat, y_hat = given_example[['input', 'label']].values[0]

                # Pick a random sample of adversarial examples
                adv_labels  = list(set(unique_labels).difference({label}))             # Get all labels != y_hat
                rand_labels = np.random.permutation(adv_labels)[:classes_per_batch]    # Shuffle the labels and grab the first few

                # Assemble a sample DF
                sample_df = pd.DataFrame()
                for rand_label in rand_labels:
                    label_sample = data_df[data_df['label'] == rand_label].sample(n=examples_per_class, replace=True)
                    sample_df = sample_df.append(label_sample, ignore_index=True)

                # Add `examples_per_class` examples of x_hat
                if example_in_set:
                    sample_df = sample_df.append(given_example.iloc[1:], ignore_index=True)

                # Fail safes to ensure data is correct
                assert x_hat not in sample_df['input'].values.tolist(), (x_hat, y_hat, sample_df)

                if example_in_set:
                    assert len(sample_df[sample_df['label'] == label]) == examples_per_class
                else:
                    assert len(sample_df[sample_df['label'] == label]) == 0

                # Vectorize data
                x_hat, y_hat = self.vectorize(data=[(x_hat, y_hat)])
                x_sample, y_sample = self.vectorize(data=sample_df.values)

                # Assemble batch
                batch = [(x_hat, y_hat), (x_sample, y_sample)]
                batches.append(batch)

        return batches

    def __next__(self):

        try:
            if self.__mode == 'train':
                item = self.train_batches[self.__train_index]
            else:
                item = self.test_batches[self.__test_index]

        except IndexError:
            raise StopIteration()

        if self.__mode == 'train':
            self.__train_index += 1
        else:
            self.__test_index += 1

        return item
