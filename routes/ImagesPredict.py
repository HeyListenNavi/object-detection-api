from fastapi import APIRouter, WebSocket
from io import BytesIO
from PIL import Image
import json
import base64
from ultralytics import YOLO
from models import Prediction, Object
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLO("gun_detection_model.pt").to(device)

router = APIRouter()


@router.websocket("/predict")
async def predict(websocket: WebSocket):
    await websocket.accept();

    while True:
        base64_image = await websocket.receive_text()
        # Cut the data url metadata and encode it in ascii
        base64_image = base64_image.split(",")[1]
        base64_bytes = base64_image.encode(encoding="ascii")

        # Decode the base64 image
        decoded_image = BytesIO(base64.b64decode(base64_bytes))

        # Make an inference
        source = Image.open(decoded_image)
        results = model(source=source)
        
        if results[0].__len__() > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Create a response based on the prediction response model
                    coordinates = box.xyxy[0].tolist()
                    inferenced_object = box.cls[0].tolist()
                    probability = box.conf[0].tolist()
                    inferenced_object = model.names[inferenced_object]
                    prediction = Prediction(x_coordinate=coordinates[0], y_coordinate=coordinates[1], x2_coordinate=coordinates[2], y2_coordinate=coordinates[3], detected_object=inferenced_object, probs=probability)
                    await websocket.send_json(prediction.model_dump())                    
# Example
# ip:port/objects-list -> returns
# [
#   {
#       "object_name": Objeto 1"
#   },
#   {
#       "object_name": Objeto 2"
#   },
#   {
#       "object_name": Objeto 3"
#   },
# ]
@router.get("/objects-list")
def objects():
    results = model.names
    return results