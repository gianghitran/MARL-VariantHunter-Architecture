import ollama
import yaml

with open("llm-config.yaml", "r") as f:
    config = yaml.safe_load(f)

OLLAMA_HOST = config.get("llm_endpoint")
OLLAMA_MODEL = config.get("llm_model")

# Khởi tạo client trỏ đến server từ xa của bạn
client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"} if "ngrok" in OLLAMA_HOST else {},
)

# Tải trước model về server (nếu chưa có)
print("Đang tải model về server Kaggle...")
#client.pull(model="gpt-oss:20b")
client.pull(model=OLLAMA_MODEL)
print("Tải model thành công!")

# Gửi một câu hỏi
response = client.chat(
    model=OLLAMA_MODEL,
    messages=[
        {
            "role": "user",
            "content": "say Hello, I'm + {your model's name}",
        },
    ],
)

print(response["message"]["content"])