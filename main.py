import base64
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision

#class TextRequest(BaseModel):
    #text: str = ""


# Initialize FastAPI app
app = FastAPI()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")
MATHPIX_API_APP_ID = os.getenv("MATHPIX_API_APP_ID")

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Explain Text Using LLM
@app.post("/explain")
async def explain_text(file: UploadFile = File(...)):
    text = await process_image(file)

    explanation = request_explanation(text)

    return {"explanation": explanation}  # OCR Endpoint - Extract text from image


def request_explanation(text):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": f"Explain this: {text}"}]
        }
    )
    explanation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No explanation found.")
    return explanation  # OCR Endpoint - Extract text from image

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    text = await process_image(file)

    return {"recognized_text": text}


async def process_image(file):
    # Read the uploaded image file
    image_data = await file.read()

    # Send the image data to Google Vision API for text detection
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)

    # Extract recognized text from the response
    extracted_text = response.text_annotations[0].description if response.text_annotations else "No text found"

    return extracted_text


# Mathpix API URL for v3/text endpoint
MATHPIX_API_URL = "https://api.mathpix.com/v3/text"

@app.post("/mathocr")
async def recognize_math(file: UploadFile = File(...)):
    # Read the uploaded file
    image_content = await file.read()

    # Convert the image content to base64 encoding
    image_base64 = base64.b64encode(image_content).decode('utf-8')

    # Prepare the headers for the Mathpix API request
    headers = {
        "app_id": MATHPIX_API_APP_ID,
        "app_key": MATHPIX_API_KEY,
        "Content-Type": "application/json"
    }

    # Prepare the data for the Mathpix API
    data = {
        "src": f"data:image/jpeg;base64,{image_base64}",
        "formats": ["latex", "text"]
    }

    # Make the request to the Mathpix API
    response = requests.post(MATHPIX_API_URL, headers=headers, json=data)

    # Parse the response from Mathpix
    result = response.json()

    # Return the extracted LaTeX and plain text formulas
    return JSONResponse(content=result)

# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
