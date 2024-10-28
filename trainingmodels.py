import numpy as np
import torch
from torch import nn

TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 50

# This is a test network for learning pytorch
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(TOKEN_VECTOR_LENGTH * MAX_TOKEN_COUNT, 2400),
            nn.ReLU(),
            nn.Linear(2400, 1200),
            nn.ReLU(),
            nn.Linear(1200, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# This is a test network for learning LSTM
class TestLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(TestLSTM, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = self.flatten(x)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = torch.nn.functional.log_softmax(lstm_out, dim=1)
        return lstm_out