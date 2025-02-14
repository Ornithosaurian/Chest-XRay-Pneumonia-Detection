import torch
import torchvision.models as models

def resnet18(weights):
    """ Load the ResNet-18 model. """
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model
