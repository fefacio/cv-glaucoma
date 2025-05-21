import torchvision.models as models
import torch.nn as nn
from torch import optim

def get_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    # Substituindo a camada totalmente conectada
    model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(model.fc.in_features, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, 1),
                            nn.Sigmoid())
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 1e-3)
    return model, loss_fn, optimizer


def get_resnet50_ft():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Congelar todos os parâmetros inicialmente
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar as camadas da layer4 (último bloco residual)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Substituir o classificador (fc) por um para saída binária
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()  # sem sigmoid, usa BCEWithLogitsLoss
    )

    # Garantir que o novo classificador seja treinável
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss e otimizador
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    return model, loss_fn, optimizer