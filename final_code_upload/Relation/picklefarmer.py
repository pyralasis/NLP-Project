# Run this script to prepare the data for training and testing
# It creates the SentimentDataset object and saves it as a .pkl file
# This saves a few minutes and prevents you from having to create the datasets every time you run the model

import pickle
from datasets import *
from filereader import *

# train_ds = SentimentDataset(
#     'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/train.json'
# )
# with open("pickled_train_dataset.pkl", "wb") as f:
#     pickle.dump(train_ds, f)

# test_ds = SentimentDataset(
#     'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/test.json'
# )
# with open("pickled_test_dataset.pkl", "wb") as f:
#     pickle.dump(test_ds, f)


train_ds = RelationsDataset(
    'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/train.json'
)
with open("pickled_train_dataset_rel.pkl", "wb") as f:
    pickle.dump(train_ds, f)

test_ds = RelationsDataset(
    'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/test.json'
)
with open("pickled_test_dataset_rel.pkl", "wb") as f:
    pickle.dump(test_ds, f)
