from datasets import SentimentDataset
from filereader import *
from trainingmodels import *
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda
from transformers import BertTokenizer, BertForSequenceClassification

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def main():
    print(text_to_array("THIS IS A TEST"))
    # The semeval paper used r-bert but we are using bert for now
    # Not sure which bert needs to be used. using this one for now
    bertModel = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # lower case model
        num_labels = 11, # we want 11 labels for the bio tagging
        output_attentions = False, 
        output_hidden_states = False,
    )
    # Use GPU
    bertModel.cuda()

    transform = Compose([            
        Lambda(text_to_array),
        ToTensor()
    ])
    
    target_transform = Lambda(lambda opinion_list: torch.tensor([1 if len(opinion_list) > 0 else 0], dtype=torch.float32))

    train_ds = SentimentDataset(
        'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/train.json',
        transform,
        target_transform
    )

    test_ds = SentimentDataset(
        'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/test.json',
        transform,
        target_transform
    )

    # Bert paper recommeds batch of 16, or 32
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True)
    # bert paper recommends adamw optimizer with lr of 5e-5, 3e-5, or 2e-5
    optimizer = torch.optim.AdamW(bertModel.parameters(),
                  lr = 2e-5
                )
    loss = nn.MSELoss()

    # bert paper recommends epoch of 2, 3, or 4
    epochs = 4
    # for t in range(epochs):
    #     print(f"-------------------------------Epoch {t+1}-------------------------------")
    #     print(f"Training...")
    #     train(train_dataloader, bertModel, loss, optimizer)
    #     print(f"Testing...")
    #     test(test_dataloader, bertModel, loss)
        
    #     bertModel.eval()
    #     print(f"Custom...")
    #     with torch.no_grad():
    #         x1 = transform("I walked to the grocery store on Tuesday .").to(device)
    #         x2 = transform("Comments on my stay at Club Hotel Dolphin").to(device)
    #         x3 = transform("Room service needs to be improved and we experienced that some of the Linen provided are damaged .").to(device)
    #         x4 = transform("The staff at the grocery store were nice to me . I enjoyed my shopping trip at the grocery store .").to(device)
    #         print(bertModel.forward(x1).item(), bertModel.forward(x2).item(), bertModel.forward(x3).item(), bertModel.forward(x4).item())
    # print("Done!")

TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 50

def text_to_array(text: str):
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    encoded_dict = tokenizer.encode_plus(
                        text,                      
                        add_special_tokens = True, 
                        max_length = MAX_TOKEN_COUNT,
                        padding = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True
                   )

    token_ids = torch.cat(encoded_dict['input_ids'], dim=0)
    return token_ids

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.7).float() == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=="__main__":
    main()