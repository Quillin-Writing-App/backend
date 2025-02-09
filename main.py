import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from google.cloud import vision

#class TextRequest(BaseModel):
    #text: str = ""


# Initialize FastAPI app
app = FastAPI()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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


# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
