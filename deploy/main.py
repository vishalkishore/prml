from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import cv2
import base64
import joblib

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from google.cloud import storage

from app.models.image_retriever import ImageRetriever, extract_cnn_features, inference_pipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def load_cifar_from_cloud():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("cifar_10_dataset")
        blob = bucket.blob("cifar10_dataset.npz")
        blob.download_to_filename("/tmp/cifar10_dataset.npz")
        with np.load("/tmp/cifar10_dataset.npz") as data:
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
    except:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return (x_train, y_train), (x_test, y_test)

(x_train, _), (x_test, y_test) = load_cifar_from_cloud()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.flatten()
print("CIFAR-10 dataset loaded successfully.")

cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

image_retriever = None

def get_image_retriever():
    global image_retriever
    if image_retriever is None:
        storage_client = storage.Client()
        bucket = storage_client.bucket("cifar_10_dataset")
        blob = bucket.blob("image_retriever.joblib")
        blob.download_to_filename("/tmp/image_retriever.joblib")
        image_retriever = ImageRetriever.load('/tmp/image_retriever.joblib')
    return image_retriever

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Image Retrieval System"}
    )

@app.post("/api/initialize")
async def initialize_retriever():
    retriever = get_image_retriever()
    return {"status": "success", "message": "Image retriever initialized"}

@app.get("/api/random-image")
async def get_random_image():
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx]
    img_class = cifar10_classes[y_test[idx]]
    
    # Convert to Base64
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": "success",
        "image": img_base64,
        "id": int(idx),
        "class": img_class
    }

@app.post("/api/retrieve-similar")
async def retrieve_similar_images(request: Request):
    body = await request.json()
    image_id = body.get("imageId")
    
    if image_id is None:
        raise HTTPException(status_code=400, detail="Image ID is required")

    query_image = x_test[int(image_id)]
    retriever = get_image_retriever()
    result_ids, distances = inference_pipeline(
        query_image, retriever=retriever, top_k=5
    )
    
    # Convert similar images to base64
    similar_images = []
    for idx in result_ids[0]:
        img = x_test[idx]
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        similar_images.append(img_base64)
    
    return {
        "status": "success",
        "similar_images": similar_images,
        "distances": distances[0].tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
