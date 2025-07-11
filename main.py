from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

# Obter chave da API da OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY não definida no arquivo .env")

# Carregar documentos
docs = SimpleDirectoryReader("data").load_data()

# Inicializar modelo de embeddings da OpenAI
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=api_key,
)

# Criar índice com modelo de linguagem e embeddings
index = VectorStoreIndex.from_documents(
    docs,
    llm=OpenAI(api_key=api_key, model="gpt-3.5-turbo"),
    embed_model=embed_model
)

# Criar aplicação FastAPI
app = FastAPI()

# Permitir requisições CORS (opcional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criar engine de consulta
query_engine = index.as_query_engine()

# Rota principal de consulta
@app.get("/")
async def root():
    return {"message": "API da Licitai rodando com sucesso!"}

@app.get("/pergunta/")
async def pergunta_api(pergunta: str):
    resposta = query_engine.query(pergunta)
    return {"resposta": str(resposta)}
