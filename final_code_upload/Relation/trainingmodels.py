import sklearn
import torch
import numpy as np
from transformers import BertForTokenClassification, BertForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import f1_score
import json

TOKEN_VECTOR_LENGTH = 96
MAX_TOKEN_COUNT = 64
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
    pred_flat = np.argmax(preds, axis=1).flatten()
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

    
    # f1_score = f1_score(labels.cpu().data, labels.data.cpu())
    # print(f1_score)

    # Calculate the average loss over all of the batches.
    # avg_val_loss = test_loss / len(dataloader)
        
    # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print(f"Test Error: \n Accuracy: {total_eval_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

def manual_validation(dataset, model: BertForTokenClassification):
    num_full_correct = 0
    num_tokens = 0
    num_labels_correct = 0

    for i in range(len(dataset.token_ids)):
        output: TokenClassifierOutput = model(dataset.token_ids[i].to(DEVICE), dataset.attention_masks[i].to(DEVICE))
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


def trainRelations(dataloader, model: BertForSequenceClassification,optimizer):
    size = len(dataloader.dataset.token_ids) 
    model.train()
    cumulative_loss = 0
    for batch, (token_ids, attention_masks, labels) in enumerate(dataloader):

        token_ids = token_ids.squeeze(1)
        attention_masks = attention_masks.squeeze(1)
        labels = labels.squeeze()
        token_ids, attention_masks, labels = token_ids.to(DEVICE), attention_masks.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()  

        # Compute prediction error
        loss = model(token_ids, 
                    token_type_ids=None, 
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

def testrel(dataloader, model):
    model.eval()
    test_loss, correct, total_eval_accuracy = 0, 0, 0
    all_labels = []
    all_preds = []
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

            for logit in output.logits.detach().cpu():
                all_preds.append(torch.argmax(logit).reshape(1))

            for label in labels.to('cpu'):
                all_labels.append(label)

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    from torcheval.metrics.functional import binary_f1_score
    f1_scorea = binary_f1_score(torch.cat(all_preds), torch.cat(all_labels))
    print(f1_scorea)


def first_model_run(dataset, model: BertForTokenClassification):
    ground_truths = {}
    preds = {}
    for i in range(len(dataset.token_ids)):
        output: TokenClassifierOutput = model(dataset.token_ids[i].to(DEVICE), dataset.attention_masks[i].to(DEVICE))
        current_sentence_tags = []
        ground_truth_tags = []
        for j in range(len(dataset.labels[i])):
            tag_value = torch.argmax(output[0][0][j]).item()
            if tag_value != 11:
                current_sentence_tags.append(tag_value)
            if dataset.labels[i][j].item() != 11:
                ground_truth_tags.append(dataset.labels[i][j].item())
        preds[dataset.my_data_points[i].text] = current_sentence_tags
        ground_truths[dataset.my_data_points[i].text] = ground_truth_tags
    
    with open('./outputs/model_one_preds.json', 'w') as fp:
        json.dump(preds, fp)
    
    with open('./outputs/model_one_ground_truths.json', 'w') as fp:
        json.dump(ground_truths, fp)

    flat_truths = np.concatenate(list(ground_truths.values())).flat
    flat_preds = np.concatenate(list(preds.values())).flat
    cutoff = flat_truths.base.size
    if flat_preds.base.size < cutoff:
        cutoff = flat_preds.base.size
    f1score = sklearn.metrics.f1_score(flat_truths[:cutoff], flat_preds[:cutoff], labels=[0,1,2,3,4,5,6,7,8,9,10], average='weighted')
    print(f1score)
    return preds

def get_sentence_pairs_from_preds():
    with open('./outputs/model_one_preds.json', 'r') as fp:
        data = json.load(fp)
    for sentence, preds in list(data.items()):
        expressions = []
        holders = []
        targets = []
        split_sentence = sentence.split()
        for i, word in enumerate(split_sentence):
            phrase_inner_length = 0
            # Negative Exp
            if preds[i] == 5:
                for j, pred in enumerate(preds[i + 1:]):
                    if pred != 6:
                        phrase_inner_length = j
                        break
                expressions.append(split_sentence[i:i+1+phrase_inner_length])
            # Neutral Exp
            if preds[i] == 7:
                for j, pred in enumerate(preds[i + 1:]):
                    if pred != 8:
                        phrase_inner_length = j
                        break
                expressions.append(split_sentence[i:i+1+phrase_inner_length])
            # Positive Exp
            if preds[i] == 9:
                for j, pred in enumerate(preds[i + 1:]):
                    if pred != 10:
                        phrase_inner_length = j
                        break
                expressions.append(split_sentence[i:i+1+phrase_inner_length])
            # Holder
            if preds[i] == 1:
                for j, pred in enumerate(preds[i + 1:]):
                    if pred != 2:
                        phrase_inner_length = j
                        break
                holders.append(split_sentence[i:i+1+phrase_inner_length])
            # Target
            if preds[i] == 3:
                for j, pred in enumerate(preds[i + 1:]):
                    if pred != 4:
                        phrase_inner_length = j
                        break
                targets.append(split_sentence[i:i+1+phrase_inner_length])
            
        print(expressions)
        print(targets)
        print(holders)

        def createPair(sentence, expression, other, placetext, pairslist):
            if other != [[], []]:
                otherindex = other[1][0].split(':')
                otherindex = list(map(int, otherindex))
                newstring = sentence[0:otherindex[0]] + placetext + " " + other[0] + " " + placetext + sentence[otherindex[1]:]
                difference = len(newstring) - len(sentence)

                expressionindex = expression[1][0].split(':')
                expressionindex = list(map(int, expressionindex))
                if otherindex[0] < expressionindex[0]:
                    first = expressionindex[0] + difference
                    second = expressionindex[1] + difference
                    expressionindex = [first, second]
                
                newstringtwo = newstring[0:expressionindex[0]] + "[expression] " + expression[0] + " [expression]" + newstring[expressionindex[1]:]
                pairslist.append(newstringtwo)
        
        created_pairs = []
        for expression in expressions:
            expression_idx = [str(sentence.find(" ".join(expression))) + ":" + str(sentence.find(" ".join(expression)) + len(" ".join(expression)))]
            for target in targets:
                target_idx = [str(sentence.find(" ".join(target))) + ":" + str(sentence.find(" ".join(target)) + len(" ".join(target)))]
                createPair(sentence, [" ".join(expression), expression_idx], [" ".join(target), target_idx], "[target]", created_pairs)

            for holder in holders:
                holder_idx = [str(sentence.find(" ".join(holder))) + ":" + str(sentence.find(" ".join(holder)) + len(" ".join(holder)))]
                createPair(sentence, [" ".join(expression), expression_idx], [" ".join(holder), holder_idx], "[holder]", created_pairs)
        print(created_pairs)


def second_model_run(dataset, model: BertForTokenClassification):
    ...