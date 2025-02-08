import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import requests
import sympy

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
    return {"explanation": explanation}

# Run locally (if not using a cloud service)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

