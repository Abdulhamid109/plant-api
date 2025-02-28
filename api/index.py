# # api/index.py
# from fastapi import FastAPI, File, UploadFile, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os

# app = FastAPI()

# # CORS middleware configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define global variables for model and class labels
# model = None
# class_labels = [
#     'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
#     'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger',
#     'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy',
#     'papaya', 'peperchili', 'pineapple', 'pomelo', 'shallot', 'soybeans',
#     'spinach', 'sweetpotatoes', 'tobacco', 'waterapple', 'watermelon'
# ]

# # Model loading function - lazy load when needed
# def load_model():
#     global model
#     if model is None:
#         # Path to model in Vercel deployment
#         model_path = os.path.join(os.path.dirname(__file__), 'plant_classification_AM_model.h5')
#         model = tf.keras.models.load_model(model_path)
#     return model

# @app.get("/")
# async def root():
#     return {"message": "Plant Classification API is running"}

# @app.get("/test/")
# async def test():
#     """
#     Test endpoint
#     """
#     return {"status": "working"}

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     """
#     Main prediction endpoint
#     """
#     if not file:
#         raise HTTPException(status_code=400, detail="No file uploaded")

#     try:
#         # Read file
#         contents = await file.read()
#         if not contents:
#             raise HTTPException(status_code=400, detail="Empty file")

#         # Load model if needed
#         model = load_model()

#         # Process image
#         img = Image.open(io.BytesIO(contents)).convert('RGB')
#         img = img.resize((224, 224))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make prediction
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         confidence = float(np.max(predictions))

#         return JSONResponse({
#             "success": True,
#             "predicted_class": class_labels[predicted_class],
#             "confidence": confidence
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import imghdr

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FORMATS = {"jpeg", "png", "bmp"}

class_labels = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger',
    'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy',
    'papaya', 'peperchili', 'pineapple', 'pomelo', 'shallot', 'soybeans',
    'spinach', 'sweetpotatoes', 'tobacco', 'waterapple', 'watermelon'
]

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'plant_classification_AM_model.h5')
    model = tf.keras.models.load_model(model_path)

@app.get("/")
async def root():
    return {"message": "Plant Classification API is running"}

@app.get("/test/")
async def test():
    return {"status": "working"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Image classification endpoint.
    """
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    # Check file format
    if imghdr.what(None, h=contents) not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid image format (JPEG, PNG, BMP only)")

    try:
        # Convert image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return JSONResponse({
            "success": True,
            "predicted_class": class_labels[predicted_class],
            "confidence": confidence
        })

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
