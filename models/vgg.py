import torchvision.models as models
import torch.nn as nn
from torch import optim
from torchvision.models import VGG16_Weights
from torchvision.models import VGG19_Weights

def get_vgg16():
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
    optimizer = optim.Adam(model.parameters(), lr= 1e-3)

    return model, loss_fn, optimizer

def get_vgg19():
    model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    
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