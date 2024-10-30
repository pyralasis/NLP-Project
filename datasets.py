import os
# import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from filereader import *



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
        tags = ['O'] * len(tokens)  

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
                        if j == 0:  # B-tag 
                            tags[index_of_word_to_tag] = "B-expression" # TO-DO add polarity 
                        else:  # I-tag 
                            tags[index_of_word_to_tag] = "I-expression" #  TO-DO add polarity 
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
                            tags[index_of_word_to_tag] = "B-source"
                        else:  # I-tag 
                            tags[index_of_word_to_tag] = "I-source"
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
                            tags[index_of_word_to_tag] = "B-target"
                        else:  # I-tag 
                            tags[index_of_word_to_tag] = "I-target"
        
        # Append the results
        tagged_data.append({"tokens": tokens, "tags": tags})


      
 

   # print (tagged_data)
    return tagged_data

from filereader import load_json_data

# Load and process the data
data = load_json_data('test.json')  # or 'test2.json'
tagged_data = tag_sentiment_data(data)

# Print a few examples for verification
for entry in tagged_data[:7]:  # Adjust index to see more samples
    tokens = entry['tokens']
    labels = entry['tags']
    print("Tokens:", tokens)
    print("Labels:", labels)
    print()