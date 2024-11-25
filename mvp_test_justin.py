import os
import pickle

import torch
from transformers import BertTokenizerFast,BertForTokenClassification,AutoTokenizer,Trainer, TrainingArguments,DataCollatorForTokenClassification
from datasets import Dataset

from datasets_local import load_json_data, tag_sentiment_data


###FOR EVAL:
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
import numpy as np

# custom eval function
#pip install transformers datasets evaluate seqeval
import evaluate
seqeval = evaluate.load("seqeval")
import pandas as pd
import matplotlib.pyplot as plt
#custom dataset inspection
from visualize_dataset  import plot_class_distribution

##CHECK ALL OFTHIS 
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


tags = ['B-holder', 'I-holder','B-targ','I-targ','B-exp-Neg','I-exp-Neg','B-exp-Neu', 'I-exp-Neu', 'B-exp-Pos', 'I-exp-Pos' ,'O' ]
def compute_metrics(p): 
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print("\n\nSeqeval Results:", results)
    token_level_f1 = {}
    for tag in tags:
        if tag in results:
            token_level_f1[tag] = results[tag]["f1"]

    metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]    
        }
    metrics.update({f"f1_{tag}": token_level_f1.get(tag, 0) for tag in tags})

    return metrics 

#CHECK THIS :
#this function is used after data is seperated in to tokens and tags in datasets_local 
# goal is to ensure data is in proper format for trainer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")#move wihtin function 
def preprocess(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        return_offsets_mapping=True
    )

    labels = []
    attention_masks = []

    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map subwords to original words
        previous_word_idx = None
        aligned_labels = []
        aligned_attention_mask = tokenized_inputs["attention_mask"][i].copy()  # Copy initial attention mask

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Handle special tokens ([CLS], [SEP], etc.)
                aligned_labels.append(-100)
            elif label[word_idx] == -101:  # Handle label -101: set attention mask to 0
                aligned_labels.append(-100)
                aligned_attention_mask[idx] = 0
            elif word_idx != previous_word_idx:
                aligned_labels.append(label[word_idx])  # First subword gets the label
            else:
                aligned_labels.append(-100)  # Ignore subsequent subwords
            previous_word_idx = word_idx

        labels.append(aligned_labels)
        attention_masks.append(aligned_attention_mask)

    # Replace the original attention mask with the updated one
    tokenized_inputs["labels"] = labels
    tokenized_inputs["attention_mask"] = attention_masks

    return tokenized_inputs

id2label={1:'B-holder', 2:'I-holder', 3:'B-targ', 4:'I-targ', 5:'B-exp-Neg', 6:'I-exp-Neg', 7:'B-exp-Neu', 8:'I-exp-Neu', 9:'B-exp-Pos', 10:'I-exp-Pos', 0:'O'}
label2id={'B-holder':1, 'I-holder':2,'B-targ':3, 'I-targ':4, 'B-exp-Neg':5, 'I-exp-Neg':6, 'B-exp-Neu':7, 'I-exp-Neu':8, 'B-exp-Pos':9, 'I-exp-Pos':10, 'O':0}


###############################################################################
#CHOSE DATASET BY NAME; should be in one of the follwing:
#                       -/DATA/test_data/{DATASET}.json or 
#                       -/DATA/pickle_data/{DATASET}_processed_dataset.pkl
DATASET = "WIP_merged_english_data_WIP"#WIP_merged_english_data_WIP #smalltoken #medium_test
#CHOSE MODEL SAVE NAME
MODEL_NAME = "bert-token-classification"
###############################################################################
# folder for saving results
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = f"./saved_models/{DATASET}_x_{MODEL_NAME}_x{TIMESTAMP}"
os.makedirs(MODEL_DIR, exist_ok=True)

#load  model and tokenizer (TO-DO:test with diffrent models here[mbert,roberta,distilbert...]
use_pretrained_model = False
if use_pretrained_model:
    model = BertForTokenClassification.from_pretrained("./results")## set this up FOR EVAL ON GIVEN MODEL and DATASET? maybe do in diffrent script 
else:
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=11,id2label=id2label, label2id=label2id)
 

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


# move to GPU or CPU
# Check GPU availability
#  pip install tensorflow transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

pickle_file = f"./DATA/pickle_data/{DATASET}_processed_dataset.pkl"
#if pickled dataset exists #else: process data 
if os.path.exists(pickle_file):
    # Load the preprocessed dataset
    with open(pickle_file, "rb") as file:
        processed_dataset = pickle.load(file) 
        print(f"Loaded processed dataset {DATASET} from pickle.")
else:
    # Get, convert to Dataset, allign, and validate data 
    json_file = f"./DATA/test_data/{DATASET}.json"
    data = load_json_data(json_file) # get json data, If you change this check for pickle 
    print(f"Loaded {len(data)} data samples from {json_file}.")#print("Sample data:", data[0])  # validate

    tagged_data_list = tag_sentiment_data(data) # ground truth tags given proper format 
    print(f"Generated tags for {len(tagged_data_list)} samples.")#print("Sample tagged data:", tagged_data_list[0]) # validate

    tagged_dataset = Dataset.from_list(tagged_data_list)

    # Apply preprocessing
    processed_dataset = tagged_dataset.map(preprocess, batched=True)

  
    ### START: TEMP INSPECTION CODE --------------------------------------
    def filter_padded_output(sample):
        tokens = sample["tokens"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]

        # Filter tokens, labels, and attention mask where attention_mask == 1
        non_padded_tokens = [token for token, mask in zip(tokens, attention_mask) if mask == 1]
        non_padded_labels = [label for label, mask in zip(labels, attention_mask) if mask == 1]
        non_padded_attention_mask = [mask for mask in attention_mask if mask == 1]

        return non_padded_tokens, non_padded_labels, non_padded_attention_mask

    # Inspect the first sample in the dataset
    if(False):
        for sample in processed_dataset.select(range(1)):  # Adjust range for more examples
            non_padded_tokens, non_padded_labels, non_padded_attention_mask = filter_padded_output(sample)
            print("Non-padded Tokens:", non_padded_tokens)
            print("Non-padded Labels:", non_padded_labels)
            print("Non-padded Attention Mask:", non_padded_attention_mask)

    ### END: TEMP INSPECTION CODE --------------------------------------


    # PICKLE HERE FOR NEW DATA(note: if model or tokenizer is changed must re-pickle):
    with open(pickle_file, "wb") as file:
       pickle.dump(processed_dataset, file)
       print(f"Saved processed dataset {DATASET} to {pickle_file}.")

### TEST ---------------------------------------------

# training testing split 
split = processed_dataset.train_test_split(test_size=0.2)  # 80-20 #stratify_by_column=""

training_dataset = split["train"]
evaluation_dataset = split["test"] 

#check for imbalnce 
plot_class_distribution(processed_dataset, id2label)
# ALSO SPLIT FOR FINAL EVAL DATA? 


#  training args 
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=10,#ADJUST for GPU UTILIZTION 
    per_device_eval_batch_size=10,
    num_train_epochs=3,
    weight_decay=0.01,
    #warmup_steps? 
    logging_dir="./logs",
    #armu
    #remove_unused_columns=False #??
    #logging_steps=10,  # Log every 10 steps # default is 500?
    #gradient_accumulation_steps
    fp16=torch.cuda.is_available(),  # Enable mixed precision for faster training (optional, NVIDIA GPUs only)

)
# Save hyperparameters
hyperparams = {
    "learning_rate": training_args.learning_rate,
    "num_train_epochs": training_args.num_train_epochs,
    "per_device_train_batch_size": training_args.per_device_train_batch_size,
    "weight_decay": training_args.weight_decay,
    "fp16": training_args.fp16,
}
with open(os.path.join(MODEL_DIR, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# trainer
collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=evaluation_dataset,  # Use a separate eval set in production
    data_collator=collator, # nsure uniform sequence length i
    processing_class=tokenizer,#?
    compute_metrics=compute_metrics

)

#  training
trainer.train()

# evaluation 
results = trainer.evaluate()
print("Evaluation Results:", results)

# Confusion matrix for error analysis[need to create seperate data set?]
predictions = trainer.predict(evaluation_dataset)
labels = predictions.label_ids.flatten()
preds = predictions.predictions.argmax(-1).flatten()

cm = confusion_matrix(labels, preds)
ConfusionMatrixDisplay(cm).plot()

# save evaluation metrics
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(results , f, indent=4)

# save the model and tokenizer
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)


#CHECK ME x
#plot here
# Run evaluation
print("Evaluation Results:", results)


