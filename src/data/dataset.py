import os
from torchvision import datasets
from .transforms import train_transform, test_transform
from src.config import Config

def load_data():
    """ Load the training, test and validation datasets. """
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=test_transform)
    val_dataset = datasets.ImageFolder(Config.VAL_DIR, transform=test_transform)

    return train_dataset, test_dataset, val_dataset