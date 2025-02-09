import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import requests
import cv2
import numpy as np
import sympy
from google.cloud import vision

from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str = ""

# Initialize FastAPI app
app = FastAPI()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Explain Text Using LLM
@app.post("/explain")
async def explain_text(request: TextRequest):
    text = request.text
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": f"Explain this: {text}"}]
        }
    )
    explanation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No explanation found.")
    return {"explanation": explanation}# OCR Endpoint - Extract text from image
# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()

    # Convert image to a PIL object
    image = Image.open(BytesIO(image_data))

    # Send the image data to Google Vision API for text detection
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)

    # Extract recognized text from the response
    extracted_text = response.text_annotations[0].description if response.text_annotations else "No text found"

    return {"recognized_text": extracted_text}

def preprocess_image(image):
    """Apply preprocessing techniques to enhance OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Adaptive Thresholding (better for varying lighting)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Denoise using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed

# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

