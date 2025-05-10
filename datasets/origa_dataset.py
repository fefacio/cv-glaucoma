import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

class OrigaDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, images_path, transform = None):
        self.df = pd.read_csv(df_path)
        self.images_path = images_path
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_path, row['Filename'])

        image = Image.open(image_path).convert('RGB')
        label = torch.tensor([row['Glaucoma']], dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def save_image_transform(self, idx):
        image, _ = self[idx]
        print(f'Normalized size: {image.size()}')
        plt.imshow(image.permute(1,2,0).numpy())
        plt.savefig("image_normalized.png")
        plt.close()

    def save_image(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_path, row['Filename'])
        image = Image.open(image_path).convert('RGB')

        print(f'Original size: {image.size}')
        plt.imshow(image)
        plt.savefig("image_normal.png")
        plt.close()

