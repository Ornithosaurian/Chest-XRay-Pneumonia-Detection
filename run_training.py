import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import load_data
from src.models.resnet import resnet18
from src.training.train import train_model
from src.config import Config
from torchvision.models import ResNet18_Weights

def main():
    train_dataset, test_dataset, val_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS, 
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True, 
                            persistent_workers=True)
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    if Config.WEIGHTED_LOSS:
        class_counts = [len(train_dataset.targets) for i in range(len(set(train_dataset.targets)))] 
        class_weights = torch.tensor([1.0 / x for x in class_counts]).to(Config.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=Config.EPOCHS)

if __name__ == "__main__":
    main()
