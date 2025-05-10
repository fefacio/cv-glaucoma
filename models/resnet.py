import torchvision.models as models
import torch.nn as nn
from torch import optim

def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    """
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    Observe que a não faz sentido ajustar a saída original uma vez que ela já
    está no formato desejado (1x1). Veja a saída do comando:
    summary(model, torch.zeros(1,3,224,224));
    """
    # Substituindo a camada totalmente conectada
    model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, 1),
                            nn.Sigmoid())
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 1e-3)
    return model, loss_fn, optimizer