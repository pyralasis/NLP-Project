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