from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from src.data.transforms import test_transform
from src.models.resnet import resnet18
from src.config import Config
from torchvision.models import ResNet18_Weights
import base64

app = FastAPI()

# Завантажуємо модель
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.load_state_dict(torch.load("models/model_epoch_10_86.pth", weights_only=False))
model.eval()
model = model.to(Config.DEVICE)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    buffered_orig = BytesIO()
    img.save(buffered_orig, format="PNG")
    img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode()

    img_tensor = test_transform(img).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    prediction = Config.CLASS_NAMES[predicted.item()]

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    prob = probabilities[0][predicted.item()].item()

    to_pil = ToPILImage()
    img_pil = to_pil(img_tensor.squeeze(0).cpu())

    buffered_processed = BytesIO()
    img_pil.save(buffered_processed, format="PNG")
    img_str_processed = base64.b64encode(buffered_processed.getvalue()).decode()

    return JSONResponse(content={
        "prediction": prediction,
        "probability": prob,
        "original_image": img_str_orig,
        "processed_image": img_str_processed
    })
