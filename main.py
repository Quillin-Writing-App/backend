import base64
import json
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision

# class TextRequest(BaseModel):
#   text: str = ""


# Initialize FastAPI app
app = FastAPI()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")
MATHPIX_API_APP_ID = os.getenv("MATHPIX_API_APP_ID")
encoded_credentials = os.getenv("GOOGLE_CREDENTIALS")

# Initialize the Google Cloud Vision client
credentials_json = base64.b64decode(encoded_credentials).decode("utf-8")
credentials_dict = json.loads(credentials_json)
client = vision.ImageAnnotatorClient.from_service_account_info(credentials_dict)

conversation_history = []


# Explain Text Using LLM
@app.post("/explain")
async def explain_text(file: UploadFile = File(...)):
    text = await process_image(file)
    global conversation_history

    conversation_history = [{"role": "user", "content": f"Explain this: {text}"}]

    explanation = request_groq(text, "Explain this", True)

    clarifying_prompts = request_groq("", "Suggest three clarifying prompts a user might ask in plaint text, "
                                          "separated only by a semicolon. Do not include any extra "
                                          "information").split("; ")

    return {"explanation": explanation, "clarifying_prompts": clarifying_prompts}


def request_groq(text, prompt, append_history=False):
    global conversation_history
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": f"{prompt}: {text}"}]
        }
    )
    response = response.json()
    if append_history:
        print(response.get("choices", [{}])[0])
        conversation_history.append(response.get("choices", [{}])[0])
    explanation = response.get("choices", [{}])[0].get("message", {}).get("content", "Can't process request.")
    return explanation


# OCR Endpoint - Extract text from image
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
        "formats": ["latex_styled", "text"]
    }

    # Make the request to the Mathpix API
    response = requests.post(MATHPIX_API_URL, headers=headers, json=data)

    # Check for errors in the response
    if response.status_code != 200:
        return JSONResponse(content="Error recognizing math")

    # Parse the response from Mathpix
    result = response.json()

    # Return the extracted LaTeX and plain text formulas
    return JSONResponse(content=result)


# Endpoint to process image containing both plain text and math
@app.post("/process-page/")
async def process_page(file: UploadFile = File(...)):
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

    # Prepare the data for the Mathpix API (we'll request both text and LaTeX)
    data = {
        "src": f"data:image/jpeg;base64,{image_base64}",
        "formats": ["latex_styled", "text", "html"]
    }

    # Make the request to the Mathpix API
    response = requests.post(MATHPIX_API_URL, headers=headers, json=data)

    # Check for errors in the response
    if response.status_code != 200:
        return JSONResponse(content="Error recognizing math")

    # Parse the response from Mathpix
    result = response.json()

    # print(result)

    # Extract the plain text and LaTeX (if available)
    plain_text = result.get("text", "")
    latex = result.get("latex_styled", "")

    # Convert plain text and LaTeX into a Markdown-compatible format
    markdown_content = convert_to_markdown(plain_text or latex)

    # Return the generated markdown
    return JSONResponse(content={"markdown": markdown_content})


def convert_to_markdown(markdown: str) -> str:
    """
    Convert the OCR results into a Markdown-formatted LaTeX page.
    - plain_text: The extracted plain text content.
    - latex: The extracted LaTeX formulas.
    Returns a Markdown string.
    """

    markdown = request_groq(markdown,
                            "Convert MathPix markdown into a regular markdown, fix spelling mistakes, and replace "
                            "arrays with inline equations. Do not include any additional notes or other information "
                            "apart from the markdown itself.")

    return markdown


# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
