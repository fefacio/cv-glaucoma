# Data-science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from sklearn.model_selection import train_test_split

# Utils
import time
import os
from tqdm import tqdm

# Custom
from datasets.smdg import SMDGDataset
from models import vgg, resnet


# Global variables
IMAGES_PATH='./data/'
DATASET_PATH='./data/metadata.csv'
RESULTS_PATH='./results/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET = 'types'



#####################################
# Training and validation functions #
#####################################
def train(model, train_loader, loss_fn, optimizer):
  model.train()
  epoch_losses = []
  epoch_accuracies = []

  for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Forward pass
        prediction = model(x)

        # Calculate loss
        batch_loss = loss_fn(prediction, y)
        epoch_losses.append(batch_loss.item())

        # Calculate accuracy
        is_correct = (prediction > 0.5).int() == y.int()
        epoch_accuracies.extend(is_correct.cpu().numpy())

        # Backward pass and optimization
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

  # Calculate mean loss and accuracy for this epoch
  epoch_loss = np.mean(epoch_losses)
  epoch_accuracy = np.mean(epoch_accuracies)

  return epoch_loss, epoch_accuracy


def evaluate(model, validation_loader, loss_fn):
    model.eval()
    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for x, y in validation_loader:
            # Forward pass
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x)

            # Calculate loss
            val_loss = loss_fn(prediction, y)
            epoch_losses.append(val_loss.item())

            # Calculate accuracy
            is_correct = (prediction > 0.5).int() == y.int()
            
            epoch_accuracies.extend(is_correct.cpu().numpy())

    # Calculate mean loss and accuracy for validation
    epoch_loss = np.mean(epoch_losses)
    epoch_accuracy = np.mean(epoch_accuracies)

    return epoch_loss, epoch_accuracy


def train_val_loop(model, loss_fn, optimizer, train_loader, test_loader, epochs, output='0'):
    print("Starting train-validation loop...")
    result_csv_path = os.path.join(RESULTS_PATH, f"{output}.csv")
    columns = [
        'epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    ]
    pd.DataFrame(columns=columns).to_csv(result_csv_path, index=False)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")

        start = time.time()

        # Training
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer)

        # Evaluating
        val_loss, val_acc = evaluate(model, test_loader, loss_fn)

        end = time.time()
        elapsed_time = end- start

        # Store metrics for plotting or logging
        result = [epoch + 1, 
                  elapsed_time, 
                  train_loss, 
                  train_acc, 
                  val_loss, 
                  val_acc]
        pd.DataFrame([result], columns=columns).to_csv(
            result_csv_path, mode='a', index=False, header=False
        )




def main():
    df = pd.read_csv(DATASET_PATH)
    
    # Balance dataset
    min_class = len(df[df[TARGET]==1])
    df = pd.concat([
        df[df[TARGET] == 1],
        df[df[TARGET] == 0].iloc[:min_class]
    ])


    train_df, test_df = train_test_split(df, 
                                         test_size=0.2, 
                                         stratify=df[TARGET], 
                                         random_state=42)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Normalize from 0-255 to 0-1 and rearrange to (C, H, W)
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])


    # Train and test datasets
    train_dataset = SMDGDataset(train_df, images_path=IMAGES_PATH,
                             transform=transform)
    test_dataset = SMDGDataset(test_df, images_path=IMAGES_PATH, 
                            transform=transform)


    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    print(train_dataset.df[TARGET].value_counts())
    print(test_dataset.df[TARGET].value_counts())

    models_pipeline = {
        "vgg16-fe": vgg.get_vgg16(),
        "vgg16-ft3": vgg.get_vgg16_ft(3),
        "vgg16-ft6": vgg.get_vgg16_ft(6),
        "resnet50-fe": resnet.get_resnet50(),
        "resnet50-ft1": resnet.get_resnet50_ft(1)
    }

    ##################################
    #   Loop using models pipeline   #
    ##################################

    MODELS_PATH = os.path.join(RESULTS_PATH, "models_trained")
    os.makedirs(MODELS_PATH, exist_ok=True)

    for model_name, (model, loss_fn, optimizer) in models_pipeline.items():
        print(f"\n===== Training model: {model_name} =====")
        
        # Train and save metrics
        model = model.to(DEVICE)
        train_val_loop(model, loss_fn, optimizer,
                       train_loader, test_loader, epochs=10, output=model_name)

        # Save the trained model in the MODELS_PATH
        model_path = os.path.join(MODELS_PATH, f"{model_name}.pt")
        torch.save(model, model_path)
        
        print(f"Model '{model_name}' saved in: {model_path}")
    


if __name__ == "__main__":
    main()
