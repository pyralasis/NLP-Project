from itertools import product
import os
from typing import Any
# import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from filereader import *
from enum import Enum
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm
import torch 
from trainingmodels import MAX_TOKEN_COUNT
import re


class Tag(Enum):
    O = 0
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
    Special = 11
    
    # Special = -100
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

        for item in tqdm(self.my_data_points):
            token_id, attention_mask = self.encode_text(item.text)
            self.token_ids.append(token_id)
            self.attention_masks.append(attention_mask)
            raw_labels = tag_sentiment_data(item)
            self.labels.append(torch.tensor(list(map(lambda x: x.value, raw_labels))))

    def __len__(self):
        return len(self.my_data_points)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.attention_masks[idx], self.labels[idx]
    
    # Returns the token ids and attention masks for the dataset
    def encode_text(self, text: str):
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = MAX_TOKEN_COUNT,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation = True
        )

        token_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return token_ids, attention_masks

def tokenizer_TBD(text: str):
    return text.split()

def tag_sentiment_data(datapoint: DataPoint): # PROCESS DATA IN TO BIO FORMAT 
    text = datapoint.text
    opinions = datapoint.rawOpinions

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
    tags = [Tag.Special] + tags + [Tag.Special]
    outputList = [Tag.Special] * MAX_TOKEN_COUNT
    if(len(tags) <= MAX_TOKEN_COUNT):
        outputList[0:len(tags)] = tags
    else:
        outputList = tags[0:MAX_TOKEN_COUNT]
    return outputList


class RelationsDataset(Dataset):
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

        self.truepairs = []
        self.falsepairs = []

        def createPair(expression, other, placetext, pairslist):
            if other != [[], []]:
                otherindex = other[1][0].split(':')
                otherindex = list(map(int, otherindex))
                newstring = text[0:otherindex[0]] + placetext + " " + other[0][0] + " " + placetext + text[otherindex[1]:]
                difference = len(newstring) - len(text)

                expressionindex = expression[1][0].split(':')
                expressionindex = list(map(int, expressionindex))
                if otherindex[0] < expressionindex[0]:
                    first = expressionindex[0] + difference
                    second = expressionindex[1] + difference
                    expressionindex = [first, second]
                
                newstringtwo = newstring[0:expressionindex[0]] + "[expression] " + expression[0][0] + " [expression]" + newstring[expressionindex[1]:]
                pairslist.append(newstringtwo)

        for item in tqdm(self.my_data_points):

            def get_update_opinion(d: dict, val: Any):
                if val != [[], []]:
                    txt = val[0][0]
                    idx = list(map(int, val[1][0].split(":")))
                    d[txt] = idx
                    return txt

                return None


            holders = {}
            targets = {}
            expressions = {}
            
            valid = set[tuple[str, str]]()

            opinion: Opinion
            for opinion in item.opinions:
                hldr = get_update_opinion(holders, opinion.source)
                tgt = get_update_opinion(targets, opinion.target)
                exp = get_update_opinion(expressions, opinion.polar_expression)

                if exp is not None:
                    if hldr is not None:
                        valid.add((exp, hldr))
                    if tgt is not None:
                        valid.add((exp, tgt))

            for expression, exp_idx in expressions.items():
                exp_str = item.text[0:exp_idx[0]] + "[expression] " + expression + " [expression]" + item.text[exp_idx[1]:]
                str_len_incr = len(exp_str) - len(item.text)

                def create_pair(text: str, idx: list[int], holder: bool):
                    start = idx[0]
                    end = idx[1]

                    if idx[0] >= exp_idx[1]:
                        start += str_len_incr
                        end += str_len_incr

                    if holder:
                        new_str = exp_str[0:start] + "[holder] " + text + " [holder]" + exp_str[end:]
                    else:
                        new_str = exp_str[0:start] + "[target] " + text + " [target]" + exp_str[end:]

                    if (expression, text) in valid:
                        self.truepairs.append(new_str)
                    else:
                        self.falsepairs.append(new_str)


                for holder, hold_idx in holders.items():
                    create_pair(holder, hold_idx, True)

                for target, tgt_idx in targets.items():
                    create_pair(target, tgt_idx, False)

        # for item in tqdm(self.my_data_points):
        #     currentOpinion: Opinion
        #     for i, currentOpinion in enumerate(item.opinions):
        #         text = item.text
        #         expression = currentOpinion.polar_expression
        #         holder = currentOpinion.source
        #         target = currentOpinion.target

        #         createPair(expression, holder, "[holder]", self.truepairs)
        #         createPair(expression, target, "[target]", self.truepairs)
        #         otherOpinions = item.opinions[:i] + item.opinions[i+1:]
        #         for otherOpinion in otherOpinions:
        #             otherholder = otherOpinion.source
        #             othertarget = otherOpinion.target

        #             createPair(expression, otherholder, "[holder]", self.falsepairs)
        #             createPair(expression, othertarget, "[target]", self.falsepairs)

        for pair in tqdm(self.truepairs):
            token_id, attention_mask = self.encode_text(pair)
            self.token_ids.append(token_id)
            self.attention_masks.append(attention_mask)
            self.labels.append(torch.tensor([1]))
        
        for pair in tqdm(self.falsepairs):
            token_id, attention_mask = self.encode_text(pair)
            self.token_ids.append(token_id)
            self.attention_masks.append(attention_mask)
            self.labels.append(torch.tensor([0]))
        print(len(self.token_ids))


    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.attention_masks[idx], self.labels[idx]
    
    # Returns the token ids and attention masks for the dataset
    def encode_text(self, text: str):
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = MAX_TOKEN_COUNT,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation = True
        )

        token_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return token_ids, attention_masks