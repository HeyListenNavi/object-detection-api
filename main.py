from fastapi import FastAPI
import routes.ImagesPredict as ImagesPredict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(ImagesPredict.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

app.mount("/", StaticFiles(directory="static",html = True), name="static")