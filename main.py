from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Obtém chave da API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY não definida")

# Carrega diretamente o arquivo JSON da raiz
docs = SimpleDirectoryReader(input_files=["LicitAI.json"]).load_data()

# Embedding com chave
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=api_key
)

# Criação do índice
index = VectorStoreIndex.from_documents(
    docs,
    llm=OpenAI(api_key=api_key, model="gpt-3.5-turbo"),
    embed_model=embed_model
)

# FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Query engine
query_engine = index.as_query_engine()

# Rota básica
@app.get("/")
def root():
    return {"message": "API LicitAI rodando com sucesso!"}

# Rota de pergunta
@app.get("/pergunta/")
async def pergunta_api(pergunta: str):
    resposta = query_engine.query(pergunta)
    return {"resposta": str(resposta)}
