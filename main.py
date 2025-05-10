import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torchvision import transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import VGG16_Weights
from torch.utils.data.sampler import SubsetRandomSampler

# Datasets
from datasets.origa_dataset import OrigaDataset
from models import vgg, resnet



DATA_PATH='./data/ORIGA/'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'




def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    
    optimizer.zero_grad()          
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()


def train_val_loop(model, loader, optimizer, loss_fn, epochs):
    print(f'Starting training and validating')
    train_losses, train_accuracies = [], []
    val_accuracies = []
    for epoch in range(epochs):
        print(f" epoch {epoch + 1}/5")
        train_epoch_losses, train_epoch_accuracies = [], []
        # val_epoch_accuracies = []

        # Treinamento
        for ix, batch in enumerate(iter(loader)):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # Validação da acurácia
        for ix, batch in enumerate(iter(loader)):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # for ix, batch in enumerate(iter(val_dl)):
        #     x, y = batch
        #     val_is_correct = accuracy(x, y, model)
        #     val_epoch_accuracies.extend(val_is_correct)
        # val_epoch_accuracy = np.mean(val_epoch_accuracies)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        # val_accuracies.append(val_epoch_accuracy)

        print(f'loss: {train_losses}')
        print(f'accuracy: {train_accuracies}')

def save_image_mat():
    mat = scipy.io.loadmat(DATA_PATH+"Semi-automatic-annotations/001.mat")
    image = mat['mask']
    print(f'Ground-truth size: {image.shape}')
    plt.imshow(image)
    plt.savefig("mat_mask.png")
    plt.close()

def debug_cuda():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

def main():
    df = pd.read_csv(DATA_PATH+'OrigaList.csv')
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converte de 0-255 para 0-1 e rearranja para (C, H, W)
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    origa = OrigaDataset(DATA_PATH+'OrigaList.csv',
                         DATA_PATH+'Images',
                         transform_test)
    
    # origa.save_image_transform(0)

    # train_prop = 0.8
    # test_prop = 0.2
    # train_set, val_set = torch.utils.data.random_split(
    #     origa, [train_prop * len(origa), test_prop * len(origa)]
    # )
    # loader = DataLoader(origa, batch_size=32, shuffle=True)

    #print(f'Models available: {torchvision.models.list_models()}')
    #model, loss_fn, optimizer = get_model()

    #train_val_loop(model, loader, optimizer, loss_fn, 2)

    #model, _, _ = resnet.get_resnet18()
    labels = origa.df["Glaucoma"]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(weights, 32)

    print(sampler)
    print(class_weights)
    print(class_counts)
    print(sample_weights)


if __name__ == "__main__":
    main()
