import torch
from src.config import Config
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=Config.EPOCHS, is_save=True):
    writer = SummaryWriter(Config.LOGS_DIR)
    model.to(Config.DEVICE)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        print(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total * 100

        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)

        print(f"[VALID] Epoch {epoch+1}/{num_epochs}, Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.2f}%")

        if is_save:
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth"))

    writer.close()

    return val_epoch_acc


