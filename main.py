from fastapi import FastAPI
import routes.ImagesPredict as ImagesPredict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

origins = [
    "http://127.0.0.1:8000/predict"
    "http://127.0.0.1:8080"
    "127.0.0.1:8000"
    "127.0.0.1:8080"
    "localhost:8000"
    "localhost:8080"
    "http://localhost:8000"
    "http://localhost:8080"
]


app.include_router(ImagesPredict.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

app.mount("/", StaticFiles(directory="static",html = True), name="static")
