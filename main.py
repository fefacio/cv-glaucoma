import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights

# Datasets
from Datasets.origa_dataset import OrigaDataset


DATA_PATH='./data/ORIGA/'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


def get_model():
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    """
    Os parâmetros são os pesos e os bias de todas as camadas do modelo.
    param.requires_grad = False -> "congela" os parâmetros do modelo, impedindo
    que eles sejam atualizados durante o treinamento pelo otimizador.
    """
    for param in model.parameters():
        param.requires_grad = False
    """
    A saída é ajustada para ter um tamanho de (1, 1), o que significa que o
    pooling médio global será aplicado em um tamanho de saída fixo de 1x1
    independentemente do tamanho da entrada.
    """
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) # Ajustar a saída original (7x7), por (1x1)
    # Substituindo o Classificador original
    model.classifier = nn.Sequential(nn.Flatten(),
                              nn.Linear(512, 128),
                              nn.ReLU(),
                              # Desativa aleatoriamente 20% dos neurônios durante o treinamento
                              nn.Dropout(0.2),
                              # Função de ativação sigmoide aplicada à saída, transformando-a em um valor entre 0 e 1
                              nn.Linear(128, 1),
                              nn.Sigmoid())
    """
    Observe que a função de perda é uma perda BINÁRIA DE ENTROPIA CRUZADA
    (nn.BCELoss()), pois a saída fornecida é de uma classe binária
    """
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(DEVICE), loss_fn, optimizer

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

def main():
    print("Hello from proj-final!")
    
    df = pd.read_csv(DATA_PATH+'OrigaList.csv')
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converte de 0-255 para 0-1 e rearranja para (C, H, W)
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    origa = OrigaDataset(DATA_PATH+'OrigaList.csv',
                         DATA_PATH+'Images_Cropped',
                         transform_test)
    
    origa.save_image_transform(0)

    # train_prop = 0.8
    # test_prop = 0.2
    # train_set, val_set = torch.utils.data.random_split(
    #     origa, [train_prop * len(origa), test_prop * len(origa)]
    # )
    loader = DataLoader(origa, batch_size=32, shuffle=True)

    #print(f'Models available: {torchvision.models.list_models()}')
    model, loss_fn, optimizer = get_model()

    train_val_loop(model, loader, optimizer, loss_fn, 2)

    # print(type(model))
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
   
    

    
    


if __name__ == "__main__":
    main()
