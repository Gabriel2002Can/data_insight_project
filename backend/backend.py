import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os

# 1. Load data
df = pd.read_csv("food_price_inflation.csv")

# 2. Create much more informative texts (this is the magic)
def create_rich_text(row):
    country = row['REF_AREA_LABEL']
    year = row['TIME_PERIOD'][:4]  # only the year
    month = row['TIME_PERIOD'][5:7] if len(row['TIME_PERIOD']) > 7 else ""
    value = float(row['OBS_VALUE'])
    
    # Natural sentences the model understands better
    return f"In {country}, food price inflation was {value:.2f}% in {year}{'-'+month if month else ''}"

texts = df.apply(create_rich_text, axis=1).tolist()

# 3. Load model (PyTorch only, not TensorFlow)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # or "cuda" if you have a GPU

# 4. Encode ONCE and save (never recalculate)
cache_file = "corpus_embeddings.pt"

if os.path.exists(cache_file):
    print("Loading embeddings from cache...")
    corpus_embeddings = torch.load(cache_file)
else:
    print("Generating embeddings (only once)...")
    corpus_embeddings = embedder.encode(
        texts,
        batch_size=64,           # larger = faster
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True  # helps a lot with cosine similarity
    )
    torch.save(corpus_embeddings, cache_file)
    print(f"Embeddings saved to {cache_file}")

# 5. Enhanced queries
queries = [
    "Which country had the highest food price inflation?",
    "Which country had the lowest food price inflation since 2001?",
    "What was the biggest increase in food prices?",
    "Show me the country with the worst food inflation"
]

# 6. Search
for query in queries:
    query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    topk = torch.topk(scores, k=5)
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    for score, idx in zip(topk.values, topk.indices):
        print(f"{score:.4f} → {texts[idx]}")
