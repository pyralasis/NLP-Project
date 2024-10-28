from datasets import SentimentDataset
from filereader import *
from trainingmodels import *
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch
import spacy
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda

# Check if there is a usable gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def main():
    # Transform the sentence into an array of embeddings
    transform = Compose([            
        Lambda(text_to_array),
        ToTensor()
    ])
    
    # Check if the datapoint has an opinion
    target_transform = Lambda(lambda opinion_list: torch.tensor([1 if len(opinion_list) > 0 else 0], dtype=torch.float32))

    # The training dataset
    train_ds = SentimentDataset(
        'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/train.json',
        transform,
        target_transform
    )

    # The test dataset
    test_ds = SentimentDataset(
        'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/test.json',
        transform,
        target_transform
    )

    # Dataloaders iteratoe over the dataset
    # The training dataloader
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # The test dataloader
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)
    
    model = NeuralNetwork().to(device)

    # The optimizer; Holds current state and updates the parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # The loss function
    loss = nn.MSELoss()

    # The training loop for my test network
    # epochs = 30
    # for t in range(epochs):
    #     print(f"-------------------------------Epoch {t+1}-------------------------------")
    #     print(f"Training...")
    #     train(train_dataloader, model, loss, optimizer)
    #     print(f"Testing...")
    #     test(test_dataloader, model, loss)
        
    #     model.eval()
    #     print(f"Custom...")
    #     with torch.no_grad():
    #         x1 = transform("I walked to the grocery store on Tuesday .").to(device)
    #         x2 = transform("Comments on my stay at Club Hotel Dolphin").to(device)
    #         x3 = transform("Room service needs to be improved and we experienced that some of the Linen provided are damaged .").to(device)
    #         x4 = transform("The staff at the grocery store were nice to me . I enjoyed my shopping trip at the grocery store .").to(device)
    #         print(model.forward(x1).item(), model.forward(x2).item(), model.forward(x3).item(), model.forward(x4).item())
    # print("Done!")

    LSTMmodel = TestLSTM(96, 32, train_ds.vocab_count, 0).to(device)
    LSTMoptimizer = torch.optim.SGD(LSTMmodel.parameters(), lr=1e-3)

    # The training loop for my test LSTM network
    epochs = 30
    for t in range(epochs):
        print(f"-------------------------------Epoch {t+1}-------------------------------")
        print(f"Training...")
        train(train_dataloader, LSTMmodel, loss, LSTMoptimizer)
        print(f"Testing...")
        test(test_dataloader, LSTMmodel, loss)
        
        LSTMmodel.eval()
        print(f"Custom...")
        with torch.no_grad():
            x1 = transform("I walked to the grocery store on Tuesday .").to(device)
            x2 = transform("Comments on my stay at Club Hotel Dolphin").to(device)
            x3 = transform("Room service needs to be improved and we experienced that some of the Linen provided are damaged .").to(device)
            x4 = transform("The staff at the grocery store were nice to me . I enjoyed my shopping trip at the grocery store .").to(device)
            print(LSTMmodel.forward(x1).item(), LSTMmodel.forward(x2).item(), LSTMmodel.forward(x3).item(), LSTMmodel.forward(x4).item())
    print("Done!")


# Spacy embedding model
nlp = spacy.load("en_core_web_sm")

TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 50

# Converts a string into tokens then into an array of embeddings
def text_to_array(text: str):
    tokens = nlp(text)
    x = np.zeros((MAX_TOKEN_COUNT, TOKEN_VECTOR_LENGTH), dtype=np.float32)

    i = 0
    for i in range(min(len(tokens), 50)):
        x[i] = tokens[i].vector
        i += 1

    return x

# The training function
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

# The testing function
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