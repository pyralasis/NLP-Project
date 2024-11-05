import torch
import numpy as np
from transformers import BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 50
NUMBER_OF_UNIQUE_LABELS = 12

BATCH_SIZE = 32                # Bert paper recommends batch of 16, or 32
LEARNING_RATE = 3e-5           # Bert paper recommends adamw optimizer with lr of 5e-5, 3e-5, or 2e-5
TRAINING_EPOCHS = 3            # Bert paper recommends epoch of 2, 3, or 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

def train(dataloader, model: BertForTokenClassification,optimizer):
    size = len(dataloader.dataset) 
    model.train()
    for batch, (token_ids, attention_masks, labels) in enumerate(dataloader):

        token_ids = token_ids.squeeze()
        attention_masks = attention_masks.squeeze()
        token_ids, attention_masks, labels = token_ids.to(DEVICE), attention_masks.to(DEVICE), labels.to(DEVICE)

        model.zero_grad()  

        # Compute prediction error
        loss = model(input_ids=token_ids, 
                    attention_mask=attention_masks, 
                    labels=labels).loss
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip to prevent explosive gradient
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
            token_ids, attention_masks, labels = token_ids.to(DEVICE), attention_masks.to(DEVICE), labels.to(DEVICE)
            output  = model(input_ids=token_ids, 
                            attention_mask=attention_masks)
            # test_loss += output.loss

            logits = output.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    # avg_val_loss = test_loss / len(dataloader)
        
    # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print(f"Test Error: \n Accuracy: {total_eval_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

def manual_validation(dataset, model: BertForTokenClassification):
    num_full_correct = 0
    num_tokens = 0
    num_labels_correct = 0

    for i in range(len(dataset.token_ids)):
        output: TokenClassifierOutput = model.forward(dataset.token_ids[i].to(DEVICE), dataset.attention_masks[i].to(DEVICE))
        has_label_been_wrong = False
        for j in range(len(dataset.labels[i])):
            if dataset.labels[i][j].item() != torch.argmax(output[0][0][j]).item():
                has_label_been_wrong = True
            elif dataset.labels[i][j].item() != 11:
                num_labels_correct += 1
            if dataset.labels[i][j].item() != 11:
                num_tokens += 1
        if not has_label_been_wrong:
            num_full_correct += 1
    print(f"Datapoints Fully Correct: {num_full_correct} out of {len(dataset.token_ids)}, Percent datapoints Fully Correct: {num_full_correct/len(dataset.token_ids)}, Tokens Labeled Correctly: {num_labels_correct}, Tokens Labeled Incorrectly: {num_tokens - num_labels_correct}, Percent Tokens Correctly Labeled: {num_labels_correct / num_tokens}")