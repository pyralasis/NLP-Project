import numpy as np
import spacy
import spacy.tokens
import torch
# from torch.ao import nn
from torch import nn


# Get cpu, gpu or mps device for training.


TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 50

# Define model
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
