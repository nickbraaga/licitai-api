import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding  # NOVO

# ✅ Configuração da chave da OpenAI
api_key = os.environ["OPENAI_API_KEY"]

# ✅ Carrega os documentos
docs = SimpleDirectoryReader(input_files=["LicitAI.json"]).load_data()

# ✅ Define o modelo de embeddings explicitamente
embed_model = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-small")

# ✅ Cria o índice com embedding e LLM
index = VectorStoreIndex.from_documents(
    docs,
    llm=OpenAI(api_key=api_key, model="gpt-3.5-turbo"),
    embed_model=embed_model
)
engine = index.as_query_engine()

# ✅ Inicializa o app FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar")
def perguntar(pergunta: Pergunta):
    resposta = engine.query(pergunta.pergunta)
    return {"resposta": resposta.response}
