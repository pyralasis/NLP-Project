from datasets import *
from filereader import *
from trainingmodels import *
from torch.utils.data import DataLoader
from torch import Tensor, nn
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
import pickle
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
    bertModel = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", # lower case model
        num_labels = 11, # we want 11 labels for the bio tagging
        output_attentions = False, 
        output_hidden_states = False,
    )
    # Use GPU
    bertModel.cuda()
    a, b = text_to_array("THIS IS A TEST")
    # print(bertModel.forward(a.to(device), b.to(device)))
    transform = Lambda(text_to_array)
    
    target_transform = Compose(Lambda(tag_sentiment_data))


    with open("pickled_train_dataset.pkl", 'rb') as f:
        train_ds = pickle.load(f)
    
    with open("pickled_test_dataset.pkl", 'rb') as f:
        test_ds = pickle.load(f)

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
    
    for t in range(epochs):
        print(f"-------------------------------Epoch {t+1}-------------------------------")
        print(f"Training...")
        train(train_dataloader, bertModel, optimizer)
        test(test_dataloader, bertModel)
        bertModel.eval()
        print(f"Custom...")
        
    print("Done!")


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

    token_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return token_ids, attention_masks

def train(dataloader, model: BertForTokenClassification,optimizer):
    size = len(dataloader.dataset) 
    model.train()
    for batch, (token_ids, attention_masks, labels) in enumerate(dataloader):
        total_train_loss = 0
        token_ids = token_ids.squeeze()
        attention_masks = attention_masks.squeeze()
        token_ids, attention_masks, labels = token_ids.to(device), attention_masks.to(device), labels.to(device)
        model.zero_grad()  

        loss = model(input_ids=token_ids, 
                    attention_mask=attention_masks, 
                    labels=labels).loss
        
        total_train_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(token_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test(dataloader, model):
    model.eval()
    test_loss, correct, total_eval_accuracy = 0, 0, 0
    with torch.no_grad():
        for batch, (token_ids, attention_masks, labels) in enumerate(dataloader):
            token_ids = token_ids.squeeze()
            attention_masks = attention_masks.squeeze()
            token_ids, attention_masks, labels = token_ids.to(device), attention_masks.to(device), labels.to(device)
            output  = model(input_ids=token_ids, 
                            attention_mask=attention_masks, 
                            labels=labels)
            test_loss += output.loss

            logits = output.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = test_loss / len(dataloader)
        
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print(f"Test Error: \n Accuracy: {total_eval_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=="__main__":
    main()