from fastapi import FastAPI
from fastapi import UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import os
import cv2
import torch_utils as load_model
global model, graph
model = load_model.init()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origin =["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origin,
    allow_credentials = True,
    allow_methods =["*"],
    allow_headers = ["*"],
)

def generate_input_image(img):

    im_resized = cv2.resize(img, (64, 64))
    X = im_resized.reshape(1, 64, 64, 3)
    X = X/255.0
    return X

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    I = np.asarray(image)
    return I

@app.get('/index')
async def hello_world():
    return "hello world"

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    file_read = await file.read()
    X = generate_input_image(read_imagefile(file_read))
            
    prediction = model.predict(X)
    pred_value = prediction.flatten().tolist()[0]
    return { "prediction": pred_value }

