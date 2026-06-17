import ollama
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

with open("llm-config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration for Ollama Server connection
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", config.get("llm_endpoint"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", config.get("llm_model"))

def get_ollama_client():
    """Initialize and return an Ollama client"""
    client = ollama.Client(
        host=OLLAMA_HOST,
        headers={"ngrok-skip-browser-warning": "true"} if "ngrok" in OLLAMA_HOST else {},
    )
    return client

def generate_with_ollama(prompt):
    """Send prompt to Ollama and return the response"""
    client = get_ollama_client()
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return None
