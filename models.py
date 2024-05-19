from pydantic import BaseModel

# Models
class Prediction(BaseModel):
    x_coordinate: float
    y_coordinate: float
    x2_coordinate: float
    y2_coordinate: float
    detected_object: str
    probs: float

class Object:
    object_name: str

    def __init__(self, object_name):
        self.object_name = object_name