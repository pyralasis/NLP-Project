import os
# import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from filereader import *
from enum import Enum
from transformers import BertTokenizer
from tqdm import tqdm
import torch 

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

        self.token_ids = []
        self.attention_masks = []
        self.labels = []
        unformatted_labels = tag_sentiment_data(data)
        for item in unformatted_labels:
            currentTags = []
            for i in range(50):
                if i < len(item["tags"]):
                    currentTags.append(item["tags"][i].value)
                else:
                    currentTags.append(0)
            
            self.labels.append(torch.tensor(currentTags))

        for item in tqdm(self.my_data_points):
            token_id, attention_mask = text_to_array(item.text)
            self.token_ids.append(token_id)
            self.attention_masks.append(attention_mask)


    def __len__(self):
        return len(self.my_data_points)

    def __getitem__(self, idx):
        # data_point = self.my_data_points[idx]
        # text = data_point.text
        # opinions = data_point.opinions
        # if self.transform:
        #     token_ids, attention_masks = self.transform(text)
        # if self.target_transform:
        #     labels = self.target_transform(opinions)        
        return self.token_ids[idx], self.attention_masks[idx], self.labels[idx]



def tokenizer_TBD(text):
    return text.split()

def tag_sentiment_data(data): # PROCESS DATA IN TO BIO FORMAT 
    tagged_data = []
    for entry in data:# every data entry 
        text = entry['text']
        opinions = entry.get('opinions', [])

        # split in to tokens: TO-DO: FIND BEST TOKENIZER, should ensure all entrys have matching labels? This is discussed in paper
        tokens = tokenizer_TBD(text)# FOR NOW JUST USEING SPLIT. CHANGE THIS
        
        # (start, end) character spans for each token XXX
        token_char_spans = []
        char_index = 0
        for token in tokens:
            start_index = text.find(token, char_index)
            end_index = start_index + len(token) - 1
            token_char_spans.append((start_index, end_index))
            char_index = end_index + 1

        # Default Case
        tags = [Tag.O] * len(tokens)  

        # BIO labeling 
        #print(tokens)
        
        # TO-DO: account for overlapping components, repeated words  \??
        for opinion in opinions:  
            #expression tagging 
            for i,expression_loc in enumerate(opinion["Polar_expression"][1]):# for all expression instances:
                expression_tokens = tokenizer_TBD(opinion["Polar_expression"][0][i]) # get tokens 
                start, end = expression_loc.split(":") # get token phrase location
                for j, token in enumerate(expression_tokens):  # for each expression token
                    index_of_word_to_tag = None 
                    for index, (start_index, end_index) in enumerate(token_char_spans):
                        if start_index >= int(start) and end_index <= int(end) and tokens[index] == token:
                            index_of_word_to_tag = index
                            break  # Exit after match  

                    if index_of_word_to_tag is not None:
                        
                        #if opinion["Polarity"] == 'neutral'?:
                        if j == 0:  # B-tag 
                            if opinion["Polarity"] == 'Positive':
                                 tags[index_of_word_to_tag] =  Tag.B_exp_Pos #  
                            elif opinion["Polarity"] == 'Negative':
                                 tags[index_of_word_to_tag] = Tag.B_exp_Neg #   
                            else:
                                tags[index_of_word_to_tag] = Tag.B_exp_Neu#?????????????/
                        else:  # I-tag 
                            if opinion["Polarity"] == 'Positive':
                                 tags[index_of_word_to_tag] = Tag.I_exp_Pos#  
                            elif opinion["Polarity"] == 'Negative':
                                 tags[index_of_word_to_tag] = Tag.I_exp_Neg #   
                            else:
                                tags[index_of_word_to_tag] = Tag.I_exp_Neg#?????????????/
            #source tagging 
            for i,source_loc in enumerate(opinion["Source"][1]):# for all source instances:
                source_tokens = tokenizer_TBD(opinion["Source"][0][i]) # get tokens 
                start, end = source_loc.split(":") # get token phrase location
                for j, token in enumerate(source_tokens):  # for each source token
                    index_of_word_to_tag = None 
                    for index, (start_index, end_index) in enumerate(token_char_spans):
                        if start_index >= int(start) and end_index <= int(end) and tokens[index] == token:
                            index_of_word_to_tag = index
                            break  # Exit after match

                    if index_of_word_to_tag is not None:
                        if j == 0:  # B-tag 
                            tags[index_of_word_to_tag] = Tag.B_holder #"B-source"
                        else:  # I-tag 
                            tags[index_of_word_to_tag] = Tag.I_holder #"I-source"
            #target tagging 
            for i,target_loc in enumerate(opinion["Target"][1]):# for all target instances:
                target_tokens = tokenizer_TBD(opinion["Target"][0][i]) # get tokens 
                start, end = target_loc.split(":") # get token phrase location
                for j, token in enumerate(target_tokens):  # for each target token
                    index_of_word_to_tag = None 
                    for index, (start_index, end_index) in enumerate(token_char_spans):
                        if start_index >= int(start) and end_index <= int(end) and tokens[index] == token:
                            index_of_word_to_tag = index
                            break  # Exit after match

                    if index_of_word_to_tag is not None:
                        if j == 0:  # B-tag 
                            tags[index_of_word_to_tag] = Tag.B_targ #"B-target"
                        else:  # I-tag 
                            tags[index_of_word_to_tag] = Tag.I_targ #"I-target"
        
        # Append the results
        tagged_data.append({"tokens": tokens, "tags": ([Tag.O] + tags + [Tag.O])})
   # print (tagged_data)
    return tagged_data

from filereader import load_json_data

# Load and process the data
# data = load_json_data('mytest.json')  # or 'test2.json'
# tagged_data = tag_sentiment_data(data)

# Print a few examples for verification
# for entry in tagged_data[:7]:  # Adjust index to see more samples
#     tokens = entry['tokens']
#     labels = entry['tags']
#     print("Tokens:", tokens)
#     print("Labels:", labels)
#     print()


def text_to_array(text: str):
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    encoded_dict = tokenizer.encode_plus(
                        text,                      
                        add_special_tokens = True, 
                        max_length = 50,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True
                   )

    token_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return token_ids, attention_masks


class My_Dataset():
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels