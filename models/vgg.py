import torchvision.models as models
import torch.nn as nn
from torch import optim
from torchvision.models import VGG16_Weights
from torchvision.models import VGG19_Weights

def get_vgg16():
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    
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
    
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 1e-3)

    return model, loss_fn, optimizer

def get_vgg16_ft(unfreeze_last_n_conv=0):
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    
    # Initial freezing everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Getting conv layers for later unfreezing
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    
    if unfreeze_last_n_conv > 0:
        for layer in conv_layers[-unfreeze_last_n_conv:]:
            for param in layer.parameters():
                param.requires_grad = True

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) 
    
    model.classifier = nn.Sequential(nn.Flatten(),
                              nn.Linear(512, 128),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(128, 1),
                              nn.Sigmoid())

    # Só os parâmetros com requires_grad = True vão ser otimizados
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    

    return model, loss_fn, optimizer