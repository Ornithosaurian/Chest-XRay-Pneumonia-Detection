import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import load_data
from src.models.resnet import resnet18
from src.training.evaluation import evaluate_model, plot_confusion_matrix
from src.config import Config
from torchvision.models import ResNet18_Weights

def main():
    train_dataset, test_dataset, val_dataset = load_data()
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS, 
                              pin_memory=True, persistent_workers=True)
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.load_state_dict(torch.load("models/model_epoch_10.pth", weights_only=False))
    model = model.to(Config.DEVICE)

    cm = evaluate_model(model, test_loader)
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
