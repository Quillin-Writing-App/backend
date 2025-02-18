import base64
import json
import os
import random
import redis
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from google.cloud import vision
from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


# Initialize FastAPI app
app = FastAPI()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")
MATHPIX_API_APP_ID = os.getenv("MATHPIX_API_APP_ID")
encoded_credentials = os.getenv("GOOGLE_CREDENTIALS")
TEST_KEY = os.getenv("TEST_KEY")
PROD_KEY = os.getenv("PROD_KEY")

API_KEYS = {
    TEST_KEY: {"role": "tester", "rate_limit": 10},  # Limited key
    PROD_KEY: {"role": "admin", "rate_limit": None}  # Full access
}

# Initialize the Google Cloud Vision client
credentials_json = base64.b64decode(encoded_credentials).decode("utf-8")
credentials_dict = json.loads(credentials_json)
client = vision.ImageAnnotatorClient.from_service_account_info(credentials_dict)

conversation_history = []

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return API_KEYS[x_api_key]


# Rate-limiting function
def rate_limiter(api_key: dict = Depends(verify_api_key)):
    rate_limit = api_key["rate_limit"]

    if rate_limit is not None:
        key = f"rate_limit:{api_key}"

        # Check how many requests have been made in the last minute
        request_count = redis_client.get(key)

        if request_count is None:
            # If the key doesn't exist, this is the first request. Set the count to 1
            redis_client.set(key, 1, ex=60)  # Set expiration to 60 seconds
        else:
            # If the count exceeds the rate limit, reject the request
            if int(request_count) >= rate_limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Otherwise, increment the request count
            redis_client.incr(key)

    return api_key


# Explain Text Using LLM
@app.post("/explain")
async def explain_text(file: UploadFile = File(...), api_key: dict = Depends(rate_limiter)):
    text = await process_image(file)

    global conversation_history

    conversation_history = [{"role": "user", "content": f"Explain this: {text}"}]

    explanation = request_groq(conversation_history, True)
    prompts_message = [{"role": "user", "content": "Suggest three clarifying prompts a user might ask in plaint text, "
                                                   "separated only by a semicolon. Do not include any extra "
                                                   "information"}]
    clarifying_prompts = request_groq(conversation_history + prompts_message).split("; ")

    return {"explanation": explanation, "clarifying_prompts": clarifying_prompts}


def request_groq(messages, append_history=False):
    global conversation_history
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": messages
        }
    )
    response = response.json()
    if append_history:
        conversation_history.append(response.get("choices", [{}])[0].get('message', {}))
    explanation = response.get("choices", [{}])[0].get("message", {}).get("content", "Can't process request.")
    return explanation


@app.post("/clarify")
async def clarify(text: TextRequest, api_key: dict = Depends(rate_limiter)):
    global conversation_history

    conversation_history.append({"role": "user", "content": text.text})
    clarification = request_groq(conversation_history, True)

    prompts_message = [{"role": "user", "content": "Suggest three clarifying prompts a user might ask in plaint text, "
                                                   "separated only by a semicolon. Do not include any extra "
                                                   "information"}]
    clarifying_prompts = request_groq(conversation_history + prompts_message).split("; ")

    return {"explanation": clarification, "clarifying_prompts": clarifying_prompts}


# OCR Endpoint - Extract text from image
@app.post("/ocr")
async def extract_text(file: UploadFile = File(...), api_key: dict = Depends(rate_limiter)):
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
async def recognize_math(file: UploadFile = File(...), api_key: dict = Depends(rate_limiter)):
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
@app.post("/process-page")
async def process_page(file: UploadFile = File(...), api_key: dict = Depends(rate_limiter)):
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

    # Extract the plain text and LaTeX (if available)
    plain_text = result.get("text", "")
    latex = result.get("latex_styled", "")

    # Convert plain text and LaTeX into a Markdown-compatible format
    markdown_content = convert_to_markdown(plain_text or latex)

    # Return the generated markdown
    return JSONResponse(content={"markdown": markdown_content})


def convert_to_markdown(markdown: str) -> str:
    global conversation_history
    """
    Convert the OCR results into a Markdown-formatted LaTeX page.
    - plain_text: The extracted plain text content.
    - latex: The extracted LaTeX formulas.
    Returns a Markdown string.
    """

    conversation_history = [{"role": "user", "content": f"Convert MathPix markdown into a regular markdown, "
                                                        f"fix spelling mistakes, and replace arrays with inline "
                                                        f"equations. Do not include any additional notes or other "
                                                        f"information apart from the markdown itself: {markdown}"}]

    markdown = request_groq(conversation_history)

    return markdown


# NEW CODE START: Fetty Wap API Endpoint
# file: UploadFile = File(...)
@app.post("/fetty_wap")
async def send_to_memenome(file: UploadFile = File(...), api_key: dict = Depends(rate_limiter)):
    """
    Extracts text from an uploaded image, replaces the mitochondria text 
    in the Memenome API payload, and sends the request.
    """
    # Step 1: Extract text from the image using Google Vision API
    extracted_text = await process_image(file)
    # extracted_text = "Lebron James is the greatest of all time"

    headers = {
        "x-api-key": "037893a8-363e-4328-a0e3-207eaf065dea",  # If Memenome API uses X-API-Key
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    templates_response = requests.get("https://api.memenome.ai/templates", headers=headers).json().get('templates')
    random_template = random.choice(templates_response).get('url')

    # Step 2: Create the request payload with the extracted text
    payload = {
        "message": {
            "type": "text",
            "text": extracted_text  # Replace mitochondria text with extracted text
        },
        "template": {
            "url": random_template
        },
        "sound": {
            "url": "https://memenome-prod.sfo3.cdn.digitaloceanspaces.com/sounds/again.mp3"
        }
    }

    # Step 3: Send the request to the Memenome API
    MEMENOME_API_URL = "https://api.memenome.ai/fetty_wap"

    response = requests.post(MEMENOME_API_URL, headers=headers, json=payload)

    # Step 4: Return the API response
    if response.status_code != 200:
        return JSONResponse(content={"error": "Failed to send request"}, status_code=response.status_code)

    return JSONResponse(content=response.json())


# NEW CODE END

# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)