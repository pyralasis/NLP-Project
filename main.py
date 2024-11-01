from datasets import *
from filereader import *
from trainingmodels import *
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda
from transformers import BertTokenizer, BertForTokenClassification
import pickle
from trainingmodels import * 

def main():
    # The semeval paper used r-bert but we are using bert for now
    # Not sure which bert needs to be used. using this one for now
    bertModel = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", # lower case model
        num_labels = NUMBER_OF_UNIQUE_LABELS, # we want 11 labels for the bio tagging
        output_attentions = False, 
        output_hidden_states = False,
    )
    # Use GPU
    bertModel.cuda()

    with open("pickled_train_dataset.pkl", 'rb') as f:
        train_ds = pickle.load(f)
    
    with open("pickled_test_dataset.pkl", 'rb') as f:
        test_ds = pickle.load(f)

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(bertModel.parameters(), lr = LEARNING_RATE)
    
    for currentEpoch in range(TRAINING_EPOCHS):
        print(f"-------------------------------Epoch {currentEpoch+1}-------------------------------")
        print(f"Training...")
        train(train_dataloader, bertModel, optimizer)

    test(test_dataloader, bertModel)
        
    print("Done!")

if __name__=="__main__":
    main()