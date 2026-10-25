import chromadb
from openai import OpenAI
from chromadb.config import Settings
import os
import json
from dotenv import load_dotenv
load_dotenv()
print(os.environ.get('DATA_BASE_URL'))
client=OpenAI(
    base_url=os.getenv("DATA_BASE_URL"),
    api_key=os.environ.get('SILICONFLOW_API_KEY')
)
res = client.embeddings.create(
    input = ["测试文本"],
    model="Qwen/Qwen3.6-35B-A3B"
)
print(res.data[0].embedding)
