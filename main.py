import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

# ✅ Configuração da OpenAI (substitua sua chave aqui)
os.environ["OPENAI_API_KEY"] = "sua-chave-da-openai-aqui"

# ✅ Inicializa o LlamaIndex com o JSON
docs = SimpleDirectoryReader(input_files=["LicitAI.json"]).load_data()
index = VectorStoreIndex.from_documents(docs, llm=OpenAI(model="gpt-3.5-turbo"))
engine = index.as_query_engine()

# ✅ Inicializa o app FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def perguntar(pergunta: Pergunta):
    resposta = engine.query(pergunta.pergunta)
    return {"resposta": resposta.response}
