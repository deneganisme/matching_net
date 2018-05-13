#! /usr/bin/env python
# -*- coding: utf-8 -*-
# trainer.py
#
######################################################################################
#
# A PyTorch implementation of the Matching Net described in
# https://arxiv.org/pdf/1606.04080.pdf
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

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from typing import List, Tuple

#############################################
# Logging for Module
#############################################

import logging
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger_name = "{}_logger".format(str(__file__).replace("\\", "/").split("/")[-1].replace(".py", ""))
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)


class MatchingNetModel(nn.Module):
    """
    Implementation of Matching Net architecture
    """

    def __init__(self, input_size: int, vocabulary_size: int, relation_vocab_size: int, output_size: int=1,
                 embedding_dim: int=100, K: int=20, n_layers: int=1):
        """
        :param input_size: size of each input (Needed?)
        :param vocabulary_size: size of vocabulary (input vocabulary)
        :param relation_vocab_size: size of relation vocabulary (i.e. number of relations)
        :param output_size: specifies output size will be [batch_size, output_size]
        :param embedding_dim: size of embedding vectors
        :param K: number of attention steps -- TODO rename
        :param n_layers: number of layers for F and G (i.e. the embedding models)
        """
        super(MatchingNetModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.vocabulary_size = vocabulary_size
        self.rel_vocab_size  = relation_vocab_size

        # Let f' = g' and go simple with f' initially making it just an embedding
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Let's not train the embedding at first
        self.embedding.weight.requires_grad = False

        # g'(x) "Let g'(x_i) be a neural network"
        self.g_output_size      = self.embedding_dim // 2
        self.bidirectional_lstm = nn.LSTM(embedding_dim, self.g_output_size, num_layers=1, bidirectional=True)

        # f(ˆx, S) = attLSTM(f'(ˆx), g(S), K) where S = input set to embed input with
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True)
        self.out  = nn.Linear(self.embedding_dim, self.rel_vocab_size)

    @staticmethod
    def calculate_cosine_sim(x: torch.LongTensor, y: torch.LongTensor)->torch.LongTensor:
        set_cosine = F.cosine_similarity(x, y)
        set_cosine_sum = torch.sum(torch.exp(set_cosine))

        kernel = set_cosine / set_cosine_sum

        return kernel

    @staticmethod
    def to_long(x: List[int])->Variable:
        return Variable(torch.from_numpy(x).type(torch.LongTensor))

    def forward(self, input: List[int], input_set: Tuple[List[int], List[int]])->Variable:
        """
        Forward pass through Matching net
        :param input: list of indices representing words in a sentences
                      Size: batch_size x self.input_size

        :param input_set: Tuple containing (example inputs, example labels) used to embed input
                          Size: [(batch_size x self.input_size), (batch_size x self.output_size x len(self.rel2index))]
        :return:
        """

        input = self.to_long(input)

        set_inputs, set_labels = input_set

        set_inputs = self.to_long(set_inputs)
        set_labels = Variable(torch.from_numpy(set_labels).type(torch.FloatTensor))

        # Embed inputs in f' and g' (in our case f' = g')
        embedded_input   = self.embedding(input)
        embedded_example = self.embedding(set_inputs)

        # embedded_labels  = self.rel_embedding(set_labels)

        # Embed inputs using fully conditional embedding g
        g_hidden = self.init_hidden_g(set_size=embedded_example.size())
        g_output, g_hidden = self.bidirectional_lstm(embedded_example, g_hidden)

        # Compute f
        f_hidden = self.init_hidden_f()
        for _ in range(self.K):
            _, f_hidden = self.lstm(embedded_input.view(self.input_size, 1, -1), f_hidden)

            # eq. 4
            f_h, f_c = f_hidden
            f_h = f_h.add(embedded_input)

            #  Attention vector (eq. 6)
            a = F.softmax(F.linear(input=g_output, weight=f_h.squeeze(0)))

            # Readout vector (eq. 5)
            r        = torch.bmm(a, g_output)
            f_h_r    = r
            f_hidden = (f_h_r, f_c)

        # Calculate cosine similarity
        f_h, f_c = f_hidden
        kernel = self.calculate_cosine_sim(x=f_h, y=g_output)

        # Pass through a linear layer to slim to proper dimensions
        kernel = self.out(kernel)

        """
        Here, we "weight-blend" the outputs from the kernel with the target labels in our set S
        The intuition is that the kernel represents the distance of our x_hat example sentence to 
        each sentence in the set S
        By "weight-blending" using the one-hot y labels in S, we ask the model the question of the 
        n examples in S which are you the closest to?
        """
        output = torch.mm(kernel.t(), set_labels)
        output = F.softmax(output.sum(0))

        return output

    def init_hidden_g(self, set_size: Tuple[int, int])->Tuple[Variable, Variable]:
        hidden_dims = (set_size[0], set_size[1], self.g_output_size)
        return (Variable(torch.zeros(hidden_dims)),
                Variable(torch.zeros(hidden_dims)))

    def init_hidden_f(self)->Tuple[Variable, Variable]:
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (batch_size, num_layers, hidden_dim)
        return (Variable(torch.zeros(1, self.input_size, self.embedding_dim)),
                Variable(torch.zeros(1, self.input_size, self.embedding_dim)))
