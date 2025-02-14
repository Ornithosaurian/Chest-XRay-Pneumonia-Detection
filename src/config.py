import os
import torch

class Config:
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

    DATA_DIR = "chest_xray"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    VAL_DIR = os.path.join(DATA_DIR, "val")

    BATCH_SIZE = 48
    LEARNING_RATE = 0.00020483073752118586
    EPOCHS = 10
    
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_SAVE_PATH = "models"
    LOGS_DIR = "logs"
    
    WEIGHTED_LOSS = True  
