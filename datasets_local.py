import os
# import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from filereader import *
from enum import Enum
from transformers import BertTokenizerFast


class Tag(Enum):
    B_holder = 1
    I_holder =  2
    B_targ = 3
    I_targ =  4
    B_exp_Neg = 5
    I_exp_Neg = 6
    B_exp_Neu = 7
    I_exp_Neu = 8
    B_exp_Pos = 9
    I_exp_Pos =  10
    O = 0


class SentimentDataset(Dataset):
    def __init__(self, url, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
    
        data = getFileFromUrl(url)
        self.my_data_points: list[DataPoint] = []
        for item in data:
            self.my_data_points.append(DataPoint(item))    

    def __len__(self):
        return len(self.my_data_points)

    def __getitem__(self, idx):
        data_point = self.my_data_points[idx]
        text = data_point.text
        opinions = data_point.opinions
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            opinions = self.target_transform(opinions)        
        return text, opinions


# Load pretrained tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

#old tokenizer 
def tokenizer_TBD(text):

    return text.split()

 # huggung face tokinizer 
def tokenizer_custom(text):
    return  tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True, # check 
            padding="max_length",  #True:dynamic padding? adjust?
            max_length=128,        # adjust?
            )



def tag_sentiment_data(data): # PROCESS DATA IN TO BIO FORMAT ### TO-DO ? 
    tagged_data = []
    for entry in data:# every data entry 
        text = entry['text']
        opinions = entry.get('opinions', [])
  
        #tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True)
        tokenized = tokenizer_custom( text )
        tokens = tokenized.tokens()  # List of tokens
        token_char_spans = tokenized['offset_mapping']  # Character span for each token  # gets odffests for you  :/ 

        # Default Case and Special Values
        tags = [Tag.O.value] * len(tokens)  

        cls_token_index = tokens.index("[CLS]") if "[CLS]" in tokens else None
        sep_token_index = tokens.index("[SEP]") if "[SEP]" in tokens else None

        if cls_token_index is not None:
            tags[cls_token_index] = -100  # Special value for [CLS]
        if sep_token_index is not None:
            tags[sep_token_index] = -100  # Special value for [SEP]
        # Handle Padding Tokens
        for index, token in enumerate(tokens):
            if token == "[PAD]":
                tags[index] = -100  # Assign -100 to [PAD] tokens


        # TAG labeling #
        for opinion in opinions:  

            #expression tagging 
            for expression_loc in opinion["Polar_expression"][1]:
                start, end = map(int, expression_loc.split(":"))

                for index, (start_index, end_index) in enumerate(token_char_spans):
                    if start_index >= start and end_index <= end:
                        if tags[index] == Tag.O.value:  # if not already tagged
                            if start_index == start:  # B-tag
                                if opinion["Polarity"] == 'Positive':
                                    tags[index] = Tag.B_exp_Pos.value
                                elif opinion["Polarity"] == 'Negative':
                                    tags[index] = Tag.B_exp_Neg.value
                                else:
                                    tags[index] = Tag.B_exp_Neu.value
                            else:  # I-tag
                                if opinion["Polarity"] == 'Positive':
                                    tags[index] = Tag.I_exp_Pos.value
                                elif opinion["Polarity"] == 'Negative':
                                    tags[index] = Tag.I_exp_Neg.value
                                else:
                                    tags[index] = Tag.I_exp_Neu.value
             # Source Tagging
            for source_loc in opinion["Source"][1]:
                start, end = map(int, source_loc.split(":"))

                for index, (start_index, end_index) in enumerate(token_char_spans):
                    if start_index >= start and end_index <= end:
                        if tags[index] == Tag.O.value:  # if not already tagged
                            if start_index == start:  # B-tag
                                tags[index] = Tag.B_holder.value
                            else:  # I-tag
                                tags[index] = Tag.I_holder.value

            # Target Tagging
            for target_loc in opinion["Target"][1]:
                start, end = map(int, target_loc.split(":"))

                for index, (start_index, end_index) in enumerate(token_char_spans):
                    if start_index >= start and end_index <= end:
                        if tags[index] == Tag.O.value:  # if not already tagged
                            if start_index == start:  # B-tag
                                tags[index] = Tag.B_targ.value
                            else:  # I-tag
                                tags[index] = Tag.I_targ.value
        


        # Append the results
        tagged_data.append({"tokens": tokens, "tags": tags})


      
 

   # print (tagged_data)
    return tagged_data


#test code
if False:
    from filereader import load_json_data

    # Load and process the data
    data = load_json_data('mytest.json')  # or 'test2.json'
    tagged_data = tag_sentiment_data(data)

    # Print a few examples for verification
    for entry in tagged_data[:7]:  # Adjust index to see more samples
        tokens = entry['tokens']
        labels = entry['tags']
        print("Tokens:", tokens)
        print("Labels:", labels)
    print()