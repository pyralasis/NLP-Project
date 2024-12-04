# ssytem and standard
import os
import pickle
import json
from datetime import datetime

# data handling 
import numpy as np
import pandas as pd

# ML and NLP
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from transformers import BertTokenizerFast,BertForTokenClassification,AutoTokenizer,Trainer, TrainingArguments,DataCollatorForTokenClassification
from torch.utils.data import DataLoader
        # pip install tensorflow transformers

# eval 
import evaluate
import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
#from seqeval.metrics import classification_report as seqeval_report
#from seqeval.metrics import precision_score, recall_score, f1_score
# pip install transformers datasets evaluate seqeval

#local
from datasets_local import load_json_data, tag_sentiment_data
from visualize_dataset import plot_class_distribution

###############################################################################
#CHOSE DATASET BY NAME; should be in one of the follwing:
#                       -/DATA/test_data/{DATASET}.json or 
#                       -/DATA/pickle_data/{DATASET}_processed_dataset.pkl
DATASET = "WIP_merged_english_data_WIP" #WIP_merged_english_data_WIP #smalltoken #medium_test #med_large_test #test
#CHOSE MODEL
MODEL_NAME = "bert-token-classification"    #bert-token-classification"
###############################################################################
# folder for saving results
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = f"./saved_models/{DATASET}_x_{MODEL_NAME}_x{TIMESTAMP}"
os.makedirs(MODEL_DIR, exist_ok=True)
# Define the log file path
log_file = os.path.join(MODEL_DIR, "metrics_log.json")

#
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# move to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



tags = ['B-holder', 'I-holder','B-targ','I-targ','B-exp-Neg','I-exp-Neg','B-exp-Neu', 'I-exp-Neu', 'B-exp-Pos', 'I-exp-Pos' ,'O' ]
id2label={1:'B-holder', 2:'I-holder', 3:'B-targ', 4:'I-targ', 5:'B-exp-Neg', 6:'I-exp-Neg', 7:'B-exp-Neu', 8:'I-exp-Neu', 9:'B-exp-Pos', 10:'I-exp-Pos', 0:'O'}
label2id={'B-holder':1, 'I-holder':2,'B-targ':3, 'I-targ':4, 'B-exp-Neg':5, 'I-exp-Neg':6, 'B-exp-Neu':7, 'I-exp-Neu':8, 'B-exp-Pos':9, 'I-exp-Pos':10, 'O':0}

seqeval = evaluate.load("seqeval")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")#?

#ompute accuracy, precision, recall, and F1 scores for token-level and aggregated metrics.
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)  # Convert logits to predicted labels

    # Flatten predictions and labels, and filter out ignored tokens (-100)
    true_labels = []
    true_predictions = []
    for label, prediction in zip(labels, predictions):
        for l, p in zip(label, prediction):
            if l != -100:  # Ignore special tokens
                true_labels.append(l)
                true_predictions.append(p)

    # Calculate overall metrics
    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted", zero_division=0
    )

    # Classification report per class
    label_names = [v for k, v in sorted(id2label.items())]
    report = classification_report(
        true_labels, true_predictions, target_names=label_names, zero_division=0, output_dict=True
    )

    # Extract F1 scores for each class and aggregated metrics
    metrics = {
        "accuracy": accuracy,  # Include accuracy
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "B-holder F1": report.get("B-holder", {}).get("f1-score", 0.0),
        "I-holder F1": report.get("I-holder", {}).get("f1-score", 0.0),
        "B-targ F1": report.get("B-targ", {}).get("f1-score", 0.0),
        "I-targ F1": report.get("I-targ", {}).get("f1-score", 0.0),
        "B-exp-Neg F1": report.get("B-exp-Neg", {}).get("f1-score", 0.0),
        "I-exp-Neg F1": report.get("I-exp-Neg", {}).get("f1-score", 0.0),
        "B-exp-Neu F1": report.get("B-exp-Neu", {}).get("f1-score", 0.0),
        "I-exp-Neu F1": report.get("I-exp-Neu", {}).get("f1-score", 0.0),
        "B-exp-Pos F1": report.get("B-exp-Pos", {}).get("f1-score", 0.0),
        "I-exp-Pos F1": report.get("I-exp-Pos", {}).get("f1-score", 0.0),
        "Holder F1 Total": (report.get("B-holder", {}).get("f1-score", 0.0) + report.get("I-holder", {}).get("f1-score", 0.0)) / 2,
        "Target F1 Total": (report.get("B-targ", {}).get("f1-score", 0.0) + report.get("I-targ", {}).get("f1-score", 0.0)) / 2,
        "Exp. F1 Total": (
            report.get("B-exp-Pos", {}).get("f1-score", 0.0) +
            report.get("I-exp-Pos", {}).get("f1-score", 0.0) +
            report.get("B-exp-Neg", {}).get("f1-score", 0.0) +
            report.get("I-exp-Neg", {}).get("f1-score", 0.0) +
            report.get("B-exp-Neu", {}).get("f1-score", 0.0) +
            report.get("I-exp-Neu", {}).get("f1-score", 0.0)
        ) / 6,
    }

    return metrics



# Trainer callback for logging
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    
    # logging callback to log metrics at the end of each evaluation.
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = int(state.epoch) if state.epoch is not None else "unknown"
            print(f"Metrics at epoch {epoch}: {metrics}")  # Debugging output
            log_metrics(epoch, metrics, log_file)
            print(f"Epoch {epoch} metrics logged.")

    def on_init_end(self, args, state, control, **kwargs):
        print("Trainer initialization complete.")

class LossTrackingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_loss = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # Save training and evaluation losses at the end of each epoch
        if state.log_history:
            for log in state.log_history:
                if "loss" in log or "eval_loss" in log:
                    self.epoch_loss.append(log)
                    print(f"Epoch {state.epoch}: {log}")


#Logs metrics to a file for each epoch.
def log_metrics(epoch, metrics, log_file):
    print(f"Logging metrics for epoch {epoch}: {metrics}")  # Debugging
    log_entry = {
        "epoch": epoch,
        "overall_metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        },
        "aggregated_metrics": {
            "Holder F1 Total": metrics.get("Holder F1 Total"),
            "Target F1 Total": metrics.get("Target F1 Total"),
            "Exp. F1 Total": metrics.get("Exp. F1 Total"),
        },
        "Class F1 Scores": {
            "B-holder F1": metrics.get("B-holder F1"),
            "I-holder F1": metrics.get("I-holder F1"),
            "B-targ F1": metrics.get("B-targ F1"),
            "I-targ F1": metrics.get("I-targ F1"),
            "B-exp-Pos F1": metrics.get("B-exp-Pos F1"),
            "I-exp-Pos F1": metrics.get("I-exp-Pos F1"),
            "B-exp-Neg F1": metrics.get("B-exp-Neg F1"),
            "I-exp-Neg F1": metrics.get("I-exp-Neg F1"),
            "B-exp-Neu F1": metrics.get("B-exp-Neu F1"),
            "I-exp-Neu F1": metrics.get("I-exp-Neu F1"),
        }
    }
    with open(log_file, "a") as f:
        json.dump(log_entry, f, indent=4)
        f.write("\n")


#CHECK THIS :
#this function is used after data is seperated in to tokens and tags in datasets_local 
# goal is to ensure data is in proper format for trainer
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


# if available pickle file load dataset else, preprocess it.
def load_or_process_data():
    pickle_file = f"./DATA/pickle_data/{DATASET}_processed_dataset.pkl"
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

        # PICKLE HERE FOR NEW DATA(note: if model or tokenizer is changed must re-pickle by deleting old pickle file):
        with open(pickle_file, "wb") as file:
           pickle.dump(processed_dataset, file)
           print(f"Saved processed dataset {DATASET} to {pickle_file}.")
           
    return processed_dataset


def main():
    """
    Main function for training and evaluation.
    """

    #load  model and tokenizer TO-DO: evalaute existing model? 
    use_pretrained_model = False
    if use_pretrained_model:
        model = BertForTokenClassification.from_pretrained("./results")## set this up FOR EVAL ON GIVEN MODEL and DATASET? maybe do in diffrent script 
    else:
        model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=11,id2label=id2label, label2id=label2id)
 
        
    print(f"Using device: {device}")
    model.to(device)
    

    #if pickled dataset exists #else: process data 
    processed_dataset = load_or_process_data()
    
    # training testing split 
    split = processed_dataset.train_test_split(test_size=0.2)  # 80-20 #stratify_by_column=""

    training_dataset = split["train"]
    evaluation_dataset = split["test"] 

    #check for imbalnce 
    if(False):
        plot_class_distribution(training_dataset, id2label, "training")
        plot_class_distribution(evaluation_dataset, id2label, "testing")
        plot_class_distribution(processed_dataset, id2label, "total")

     
    collator = DataCollatorForTokenClassification(tokenizer)

    # ALSO SPLIT FOR FINAL EVAL DATA? 

   
       
    #  training args 
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,#
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        #weight_decay=0.01,
        #warmup_steps? 
        logging_dir="./logs",
        #armu
        #remove_unused_columns=False #??
        #logging_steps=10, # default is 500?
        #gradient_accumulation_steps
        fp16=torch.cuda.is_available(),  # Enable mixed precision for faster training (optional, NVIDIA GPUs only)

    )
    # Save hyperparameters
    hyperparams = {
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "weight_decay": training_args.weight_decay,
        "fp16": training_args.fp16,
    }
    with open(os.path.join(MODEL_DIR, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # trainer
    loss_tracker = LossTrackingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=evaluation_dataset,  # Use a separate eval set in production
        data_collator=collator, # nsure uniform sequence length i
        processing_class=tokenizer,#?
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback(), loss_tracker]  # Add custom logging callback

    )

    #  training
    trainer.train()

    # evaluation 
    results = trainer.evaluate()
    #print("Evaluation Results:", results)
    print("Epoch Losses:", loss_tracker.epoch_loss)

    # Confusion matrix for error analysis[need to create seperate data set?]
    predictions = trainer.predict(evaluation_dataset)
    labels = predictions.label_ids.flatten()
    preds = predictions.predictions.argmax(-1).flatten()

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot()

    # save evaluation metrics
    save_results = input("Would you like to save the metrics? (y/n): ").strip().lower()
    if save_results:
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
            json.dump(results , f, indent=4)
        with open(os.path.join(MODEL_DIR, "epoch_losses.json"), "w") as f:
            json.dump(loss_tracker.epoch_loss, f, indent=4)

        
    save_model = input("Would you like to save the model and tokenizer? (y/n): ").strip().lower()

    if save_model == "y":
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"Model and tokenizer have been saved to {MODEL_DIR}.")
    else:
        print("Model and tokenizer were not saved.")



if __name__ == "__main__":
    main()