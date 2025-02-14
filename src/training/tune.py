import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from src.training.train import train_model 
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

def objective(trial):
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='chest_xray/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4, 
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            num_workers=4, pin_memory=True, 
                            persistent_workers=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  
    model.to('cuda' if torch.cuda.is_available() else 'cpu')


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    val_accuracy = train_model(model, train_loader, val_loader, optimizer, criterion, is_save=False)

    return val_accuracy  

def main():
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=10) 

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best validation accuracy: {study.best_value}")

if __name__ == "__main__":
    main()
