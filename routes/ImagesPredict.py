from fastapi import APIRouter, Body, HTTPException
from io import BytesIO
from PIL import Image
import base64
from ultralytics import YOLO
from models import Prediction, Object

model = YOLO("gun_detection_model.pt")

router = APIRouter()

# Example
# ip:port/predict?base64_image=examplestring -> returns
# {
#   "x_coordinate": 10,
#   "y_coordinate": 10,
#   "width": 100,
#   "height": 100,
#   "cls": "objeto detectado",
#   "probs": 90.99
# }
@router.post("/predict")
def predict(base64_image: str = Body(...)):
    # Cut the data url metadata and encode it in ascii
    base64_image = base64_image.split(",")[1]
    base64_bytes = base64_image.encode(encoding="ascii")

    # Decode the base64 image
    decoded_image = BytesIO(base64.b64decode(base64_bytes))

    # Make an inference
    source = Image.open(decoded_image)
    width, height = source.size

    print("width: " + str(width))
    print("height: " + str(height))

    results = model(source=source, imgsz=[640, 480], rect=False)
    
    # Response is a list of predictions 
    response = []

    print(results)

    if results[0].__len__() > 0:
        try:
            for result in results:
                
                boxes = result.boxes.cpu().numpy()

                for box in boxes:
                    # Create a response based on the prediction response model
                    prediction = Prediction()
                    coordinates = box.xyxy[0].tolist()
                    inferenced_object = box.cls[0].tolist()
                    probability = box.conf[0].tolist()
                    inferenced_object = model.names[inferenced_object]

                    prediction.x_coordinate = coordinates[0]
                    prediction.y_coordinate = coordinates[1]
                    prediction.x2_coordinate = coordinates[2]
                    prediction.y2_coordinate = coordinates[3]
                    prediction.detected_object = inferenced_object
                    prediction.probs = probability
                    
                    
                    response.append(prediction)
        except Exception as e:
            raise HTTPException(500, str(e))
    else:
        raise HTTPException(404, "No inference")

    return response

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