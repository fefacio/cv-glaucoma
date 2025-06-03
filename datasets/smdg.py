import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

class SMDGDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_path, transform = None):
        self.df = df
        self.images_path = images_path
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_path, row['fundus'].strip('/'))

        image = Image.open(image_path).convert('RGB')
        label = torch.tensor([row['types']], dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        
        return image, label
    

