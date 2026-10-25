import chromadb
from openai import OpenAI
from chromadb.config import Settings
import os
import json
from dotenv import load_dotenv
load_dotenv()

client=OpenAI(
    base_url=os.getenv("DATA_BASE_URL"),
    api_key=os.environ.get('SILICONFLOW_API_KEY')
)

def get_embeddings(texts,model="Qwen/Qwen3.6-35B-A3B"):
    '''封装Embedding 模型接口'''
    data = client.embeddings.create(input=texts,model=model).data
    return [x.embedding for x in data]

def get_embeddings_batch(texts,model="Qwen/Qwen3.6-35B-A3B",batch_size=10):
    all_embeddings = []
    for i in range(0,len(texts),batch_size):
        #  texts[0:10]
        batch_texts = texts[i:i+batch_size]
        data = client.embeddings.create(input=batch_texts,model=model).data
        all_embeddings.extend(x.embedding for x in data)
        return all_embeddings

with open('../file/01.json','r', encoding='utf-8') as f:
    data = json.load(f)

instruction = [entry['instruction'] for entry in data[0:100]]
output = [entry['output'] for entry in data[100:]]

class MyVectorDBConnector:
    def __init__(self,collection_name,embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client.reset()
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self,instruction,output):
        """向collection中添加文档与向量"""
        embeddings = self.embedding_fn(instruction)
        self.collection.add(
            embeddings=embeddings, #每个文档的向量
            documents=output, #文档原文
            ids=[f"is{i}" for i in range(len(output))] #每个文档的id
        )

    def search(self,query,top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn(query),
            n_results=top_n,
        )
        return results


#创建向量数据库对象
vector_db = MyVectorDBConnector("demo",get_embeddings)
vector_db.add_documents(instruction,output)

#测试检索
user_query = "王奶奶"
results = vector_db.search(user_query,2)
for para in results['documents'][0]:
    print(para+'\n')