import os
import pickle

import torch
from transformers import BertTokenizerFast,BertForTokenClassification,AutoTokenizer,Trainer, TrainingArguments,DataCollatorForTokenClassification
from datasets import Dataset

from datasets_local import load_json_data, tag_sentiment_data

###FOR EVAL:
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# custom eval function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    labels_flat = labels.flatten()
    preds_flat = preds.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted',zero_division=0)
    accuracy = accuracy_score(labels_flat, preds_flat)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



###############################################################################
#CHOSE DATASET BY NAME; should be in one of the follwing:
#                       -/DATA/test_data/{DATASET}.json or 
#                       -/DATA/pickle_data/{DATASET}_processed_dataset.pkl
DATASET = "WIP_merged_english_data_WIP"
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
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=11)
 

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

    # Convert list to Dataset
    tagged_data = Dataset.from_list(tagged_data_list)


    # preprocess dataset
    def preprocess(examples):
        # Tokenize the text
        encodings = tokenizer(examples["tokens"], is_split_into_words=True, padding=True, truncation=True)
        encodings["labels"] = examples["tags"]  # Map tags to labels
        return encodings

    # Apply the preprocessing dataset
    processed_dataset = tagged_data.map(preprocess, batched=True)

    # PICKLE HERE FOR NEW DATA(note: if model or tokenizer is changed must re-pickle):
    with open(pickle_file, "wb") as file:
        pickle.dump(processed_dataset, file)
        print(f"Saved processed dataset {DATASET} to {pickle_file}.")

### TEST ---------------------------------------------

# training testing split 
split = processed_dataset.train_test_split(test_size=0.2)  # 80-20 #stratify_by_column=""
training_dataset = split["train"]
evaluation_dataset = split["test"] 
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

print(f"Model, metrics, and hyperparameters saved to {MODEL_DIR}")



