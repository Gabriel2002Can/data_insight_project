from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import (
    load_data,
    embed_corpus,
    embedder,
    intent_embeddings,
    intent_keys,
    handle_query
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = load_data()
corpus_emb = embed_corpus(df)

class Query(BaseModel):
    text: str

@app.post("/query")
def query_api(body: Query):
    result = handle_query(body.text, df)

    if "result" in result and hasattr(result["result"], "to_dict"):
        result["result"] = result["result"].to_dict(orient="records")

    return result
