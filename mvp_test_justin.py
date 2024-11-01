from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments,DataCollatorForTokenClassification
from datasets import Dataset
from datasets_local import  load_json_data, tag_sentiment_data

from transformers import AutoTokenizer
import torch

#load pretrained model and tokenizer 
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=11)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# Get, convert to Dataset, allign, and validate data 
data = load_json_data('medium_test.json') # get json data
print(f"Loaded {len(data)} data samples.")
#print("Sample data:", data[0])  # validate

tagged_data_list = tag_sentiment_data(data) # ground truth tags given proper format 
print(f"Generated tags for {len(tagged_data_list)} samples.")
#print("Sample tagged data:", tagged_data_list[0]) # validate


# Convert list to Dataset
tagged_data = Dataset.from_list(tagged_data_list)


### TEST ---------------------------------------------
# Function to preprocess your dataset
def preprocess(examples):
    # Tokenize the text
    encodings = tokenizer(examples["tokens"], is_split_into_words=True, padding=True, truncation=True)
    encodings["labels"] = examples["tags"]  # Map tags to labels
    return encodings

# Apply the preprocessing dataset
processed_dataset = tagged_data.map(preprocess, batched=True)
### TEST ---------------------------------------------

# training testing split 
split = processed_dataset.train_test_split(test_size=0.2)  # 80-20
training_dataset = split["train"]
evaluation_dataset = split["test"]



#  training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    #remove_unused_columns=False #??
    #logging_steps=10,  # Log every 10 steps # default is 500?
)

# trainer
collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=evaluation_dataset,  # Use a separate eval set in production
    data_collator=collator # nsure uniform sequence length i

)

# start training
trainer.train()

# evaluation 
results = trainer.evaluate()
print("Evaluation Results:", results)

#SAVE MODEL 
#model.save_pretrained("./saved_model")
#tokenizer.save_pretrained("./saved_model")

