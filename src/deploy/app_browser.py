from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
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

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.load_state_dict(torch.load("models/model_epoch_10_86.pth", weights_only=False))
model.eval()
model = model.to(Config.DEVICE)

@app.get("/")
async def main():
    content = """
    <html>
        <body>
            <h2>Upload an X-ray image for prediction:</h2>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_tensor = test_transform(img).unsqueeze(0).to(Config.DEVICE)
    img_resized = img.resize((224, 224))

    buffered_orig = BytesIO()
    img_resized.save(buffered_orig, format="PNG")
    img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode()

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

    content = f"""
    <html>
        <head>
            <style>
                img {{
                    max-width: 40%; /* Reduce maximum width to 40% of the screen */
                    height: auto;
                }}
                .image-container {{
                    display: flex;
                    justify-content: space-around;
                }}
            </style>
        </head>
        <body>
            <div>
                <h2>Prediction: {prediction}</h2>
                <h3>Probability: {prob * 100:.2f}%</h3>
                <div class="image-container">
                    <div>
                        <h3>Original Image (resized to 224x224):</h3>
                        <img src="data:image/png;base64,{img_str_orig}" alt="Original Image">
                    </div>
                    <div>
                        <h3>Processed Image (after preprocessing):</h3>
                        <img src="data:image/png;base64,{img_str_processed}" alt="Processed Image">
                    </div>
                </div>
            </div>
            <br><br>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """
    return HTMLResponse(content=content)