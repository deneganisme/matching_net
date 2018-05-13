#! /usr/bin/env python
# -*- coding: utf-8 -*-
# trainer.py
#
######################################################################################
#
# Module to train Matching Net
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

import os
from time import time
import torch
from torch import nn
from typing import Union, Tuple, List
from matching_net.data_loader import DataLoader
from matching_net.model import MatchingNetModel
import pandas as pd


#############################################
# Logging for Module
#############################################


import logging
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger_name = "{}_logger".format(str(__file__).replace("\\", "/").split("/")[-1].replace(".py", ""))
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)


class MatchingNetTrainer(object):
    """
    Class for defining and training a Matching Net
    """
    def __init__(self,
                 data_loader: DataLoader,
                 embedding_dim: int=100):
        """
        :param data_loader: DataLoader class that handles all data for model
        :param embedding_dim: length of word vectors in embedding;
                              embedding will be of size len(vocabulary) x embedding_dim
        """
        super(MatchingNetTrainer, self).__init__()
        self.data_loader = data_loader

        # Create model
        self.model = MatchingNetModel(input_size=self.data_loader.input_size,
                                      vocabulary_size=self.data_loader.vocabulary_size,
                                      relation_vocab_size=self.data_loader.num_relations,
                                      output_size=self.data_loader.output_size,
                                      embedding_dim=embedding_dim)

        logger.info(f"Model:\n{self.model}")

        # Define loss and optimizer
        # TODO wrap this into + model into a different class?
        self.loss_func = nn.CrossEntropyLoss()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters)

    def save(self, save_path: Union[str, None]=None):
        """
        Save model using PyTorch methods
        :param save_path:
        :return:
        """

        if save_path is None:
            model_dir = os.path.join(os.path.dirname(__file__), "saved_models")

            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            save_path = os.path.join(model_dir, "matching_net")

        # Objects to save
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_func': self.loss_func.state_dict(),
            'data_loader': self.data_loader.to_dict()
        }

        torch.save(state, save_path)

    def load(self, filepath: str):
        """
        Load model state from a filepath and update object variables
        :param filepath:
        :return:
        """
        state = torch.load(filepath)

        self.model = state['model']
        self.loss_func = state['loss']
        self.optimizer = state['optimizer']
        self.data_loader = state['data_loader']

    def predict(self, data, get_label: bool=False, verbose: bool=True)->Tuple[List[int], float]:
        """
        Get model's predictions and the accuracy of its predictions
        :param data:
        :param get_label:
        :return predictions, accuracy:
        """
        correct = 0
        total = 0

        # Flip rel2index to convert integer -> str
        reverse_rel = dict(zip(self.data_loader.rel2index.values(), self.data_loader.rel2index.keys()))

        predictions = []
        for item in data:

            # Transform data and get the reconstructed sentence
            inputs, target_y, sentence = self.transform_data(item, get_sentence=True)

            # mMake the prediction
            pred = self.model(*inputs)

            # Pred is a vector
            # Get max value to determine the model's confidence and to determine correctness
            confidence, pred_idx = pred.max(0)
            confidence = confidence.data.tolist()[0]

            # Get true index to compare to model's answer
            _, true_idx = target_y.max(0)

            pred_idx = pred_idx.data.tolist()[0]
            pred_rel = reverse_rel[pred_idx]

            if get_label:
                predictions.append(pred_rel)
            else:
                predictions.append(pred_idx)

            true_idx = true_idx.data.tolist()[0]
            true_rel = reverse_rel[true_idx]

            total += 1
            correct += 1 if pred_idx == true_idx else 0

            # Collect model's output vector representation to test t-SNE
            # TODO make parameter to collect this data
            # model_vec.append((pred.data.tolist(), true_rel))

            if verbose:
                logger.info(f"\nSentence:   {sentence}"
                            f"\nTruth:      {true_rel}"
                            f"\nPred:       {pred_rel}"
                            f"\nConfidence: {confidence}")

        accuracy = correct / float(total)
        return predictions, accuracy

    def transform_data(self, data, get_sentence: bool=False):
        """
        Transform data from data loader so it can be fed into the model
        :param data:
        :param get_sentence:
        :return:
        """
        target_x, target_y = data[0]
        sentence = " ".join([p for p in self.data_loader.convert_input(target_x[0]) if p != 'PAD'])

        target_y = self.model.to_long(target_y[0])

        train_set = data[1]

        if not get_sentence:
            return (target_x, train_set), target_y
        else:
            return (target_x, train_set), target_y, sentence

    def _run_epoch(self, batches)->dict:
        """
        Run a training epoch
        :param batches:
        :return:
        """

        correct = 0
        total = 0
        total_loss = 0
        avg_loss = 0
        total_conf = 0
        avg_conf = 0
        accuracy = 0

        for batch in batches:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Step 2. Get data
            inputs, target_y = self.transform_data(batch)
            # Step 3. Run our forward pass.
            pred = self.model.forward(*inputs)

            # Pred is a vector
            # Get max value to determine the model's confidence and to determine correctness
            confidence, pred_idx = pred.max(0)

            # Get true index to compare to model's answer
            _, true_idx = target_y.max(0)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            pred = pred.view(1, -1)

            loss = self.loss_func(pred, true_idx)
            loss.backward()
            self.optimizer.step()

            total += 1

            total_loss += loss.data.tolist()[0]
            avg_loss = total_loss / float(total)

            correct += 1 if pred_idx.data.equal(true_idx.data) else 0

            total_conf += confidence.data.tolist()[0]
            avg_conf = total_conf / float(total)

            accuracy = correct / float(total)

        return {
            'accuracy': accuracy,
            'average_loss': avg_loss,
            'average_confidence': avg_conf
        }

    def train(self, epochs: int, shuffle: bool = True)->Tuple[float, float]:
        """
        Train MatchingNet model
        :param epochs: number of epochs to train model for
        :param shuffle: if True, shuffle data between epochs
        :return acc, loss: accuracy and loss of the final epoch
        """
        assert epochs > 0

        for epoch in range(epochs):

            if shuffle:
                self.data_loader.shuffle()

            timer = time()
            epoch_results = self._run_epoch(batches=self.data_loader.train_batches)
            total_time = time() - timer

            logger.info(f"\n\n-----Epoch #{epoch + 1}-----\n"
                        f"Accuracy:       {epoch_results['accuracy']}\n"
                        f"Average conf:   {epoch_results['average_confidence']}\n"
                        f"Average loss:   {epoch_results['average_loss']}\n"
                        f"Time (seconds): {round(total_time, 2)}")

            acc = epoch_results['accuracy']
            loss = epoch_results['average_loss']

        # Return final accuracy and loss
        return acc, loss

    def test(self):
        """
        Test a model and get its accuracy
        :return:
        """

        # TODO add more metrics
        _, acc = self.predict(data=self.data_loader.test_batches)
        return acc

    def evaluate(self, epochs: int=1, shuffle: bool=True)->float:
        """
        Train and test model
        :param epochs: number of epochs to train model for
        :param shuffle: if True, shuffle data between epochs
        :return test_acc: model accuracy on test data
        """

        train_acc, train_loss = self.train(epochs=epochs, shuffle=shuffle)
        test_acc = self.test()

        logger.info(f"\nTrain accuracy: {train_acc}\nTest accuracy:  {test_acc}")
        return test_acc


def main():

    data_dir = os.path.join(os.path.dirname(__file__), "cache")
    data_path = os.path.join(data_dir, "data.csv")

    logger.info(f"Loading data from {data_path}")

    """
    CSV with at least the columns sentence and relation
    - Sentence is the input column of the model and contains the relation in it
    - Relation is a string representing the relation e.g. "X acquire Y"
        + If we keep relation as a string, then it is classification 
        + If we split relation up, we can think of it as a language model
        + Classification is easier so we'll stick with that
    """
    data_df = pd.read_csv(data_path)

    """
    Experiment params
    """
    input_col  = "sentence"  # Tell data_loader what our input is
    target_col = "relation"  # Tell data_loader what our output is
    classes_per_batch = 15   # Number of adversarial examples to embed in our model
    n_shot = 5               # Number of times each class is seen; note that count(class) >= n_shot

    """
    This class handles all data preparation. Our trainer will use it to train and test the model
    e.g. creating word lookup dict that maps words to integers (used for embedding),
         creates training and testing batches as specified in the paper,
         and determines the input and output length (if classification, output length == 1)
    """
    data_loader = DataLoader(data_df=data_df,
                             classes_per_batch=classes_per_batch,
                             n_shot=n_shot,
                             input_col=input_col,
                             target_col=target_col)

    logger.info(f"\n# of training points: {len(data_loader.train_data)}"
                f"\n# of test points:     {len(data_loader.test_data)}"
                f"\n# of relations:       {data_loader.num_relations}")

    # Create our trainer
    # This takes in our data_loader and uses it to create the MatchingNet model
    matching_net = MatchingNetTrainer(data_loader=data_loader)

    epochs = 5  # Could even be 3 if you're in a rush -- model performance peaks around this epoch
    matching_net.evaluate(epochs=epochs)

    # Save model
    matching_net.save()


if __name__ == '__main__':
    main()
