import torchvision.models as models
import torch.nn as nn
from torch import optim

def get_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Change final classifier
    model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(model.fc.in_features, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, 1),
                            nn.Sigmoid())
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 1e-3)
    return model, loss_fn, optimizer


def get_resnet50_ft(unfreeze_layers=0):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    layers_to_unfreeze = ['layer4', 'layer3', 'layer2', 'layer1'][:unfreeze_layers]

    # Unfreeze last N layers where N is specified by 'unfreeze_layers'
    for layer_name in layers_to_unfreeze:
        layer = getattr(model, layer_name)
        print(f'Unfreezing layer {layer}')
        for param in layer.parameters():
            param.requires_grad = True

    # Change final classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()  
    )

    # Unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    return model, loss_fn, optimizer