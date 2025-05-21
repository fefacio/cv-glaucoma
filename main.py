# Data-science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import VGG16_Weights
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

# Utils
import time
import os

# Custom
from datasets.origa_dataset import OrigaDataset
from models import vgg, resnet


# Global variables
IMAGES_PATH='./data/Images/'
DATA_PATH='./data/'
RESULTS_PATH='./results/'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


#####################################
# Training and validation functions #
#####################################
def train(model, train_loader, loss_fn, optimizer):
  model.train()
  epoch_losses = []
  epoch_accuracies = []

  for x, y in train_loader:
        # Forward pass
        prediction = model(x)

        # Calculate loss
        #y = y.squeeze().long()
        # print(f'pred: {prediction}')
        # print(f'y: {y}')
        batch_loss = loss_fn(prediction, y)
        epoch_losses.append(batch_loss.item())

        # Calculate accuracy
        is_correct = (prediction > 0.5).int() == y.int()
        #preds = prediction.argmax(dim=1)     # predicted class
        #is_correct = (preds == y)
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
            prediction = model(x)

            # y = y.squeeze().long()
            # Calculate loss
            val_loss = loss_fn(prediction, y)
            epoch_losses.append(val_loss.item())

            # Calculate accuracy
            is_correct = (prediction > 0.5).int() == y.int()
            #preds = prediction.argmax(dim=1)     # predicted class
            #is_correct = (preds == y)
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

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        start = time.time()
        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer)

        # Evaluate on the validation set
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
    df = pd.read_csv(DATA_PATH+'OrigaList.csv')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converte de 0-255 para 0-1 e rearranja para (C, H, W)
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    origa = OrigaDataset(df,
                         IMAGES_PATH,
                         transform)
    
    train_df, test_df = train_test_split(origa.df, 
                                         test_size=0.2, 
                                         stratify=origa.df['Glaucoma'], 
                                         random_state=42)
    
    # Train and test datasets
    train_df = OrigaDataset(train_df, images_path=IMAGES_PATH,
                             transform=transform)
    test_df = OrigaDataset(test_df, images_path=IMAGES_PATH, 
                            transform=transform)


    # DataLoaders
    train_loader = DataLoader(train_df, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=32, shuffle=False)


    print(train_df.df['Glaucoma'].value_counts())
    print(test_df.df['Glaucoma'].value_counts())
    
    models_pipeline = {
        "vgg16-fe": vgg.get_vgg16(),
        "vgg16-ft3": vgg.get_vgg16_ft(3),
        "vgg16-ft6": vgg.get_vgg16_ft(6),
        "resnet50-fe": resnet.get_resnet50(),
        "resnet50-ft4": resnet.get_resnet50_ft()
    }


    # Loop sobre os modelos do dicionário
    MODELS_PATH = os.path.join(RESULTS_PATH, "models_trained")
    os.makedirs(MODELS_PATH, exist_ok=True)

    for model_name, (model, loss_fn, optimizer) in models_pipeline.items():
        print(f"\n===== Treinando modelo: {model_name} =====")
        
        # Treina e salva métricas
        train_val_loop(model, loss_fn, optimizer,
                       train_loader, test_loader, epochs=10, output=model_name)

        # Caminho para salvar o modelo
        model_path = os.path.join(MODELS_PATH, f"{model_name}.pt")
        
        # Salva o modelo inteiro (estrutura + pesos)
        torch.save(model, model_path)
        
        print(f"Modelo '{model_name}' salvo em: {model_path}")
    


if __name__ == "__main__":
    main()
