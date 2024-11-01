import pickle
from datasets import *
from filereader import *

train_ds = SentimentDataset(
    'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/train.json'
)

test_ds = SentimentDataset(
    'https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/refs/heads/master/data/opener_en/test.json'
)


with open("pickled_test_dataset.pkl", "wb") as f:
    pickle.dump(test_ds, f)

with open("pickled_train_dataset.pkl", "wb") as f:
    pickle.dump(train_ds, f)