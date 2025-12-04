import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. Load model ONCE
# =========================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

INTENTS = {
    "outliers": "detect outliers, anomalies, isolation forest",
    "top": "top values, highest values, biggest inflation",
    "trend": "trend, regression, line going up or down",
    "cluster": "cluster countries, similar inflation behavior",
}

intent_texts = list(INTENTS.values())
intent_keys = list(INTENTS.keys())

intent_embeddings = embedder.encode(
    intent_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# =========================================================
# 2. Data loading
# =========================================================

def load_data(path="food_price_inflation.csv"):
    return pd.read_csv(path)

# =========================================================
# 3. Corpus embeddings
# =========================================================

def embed_corpus(df, cache_file="corpus_embeddings.pt"):
    texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

    if os.path.exists(cache_file):
        return torch.load(cache_file)

    emb = embedder.encode(
        texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    torch.save(emb, cache_file)
    return emb

# =========================================================
# 4. Intent detection
# =========================================================

def detect_intent(query):
    q_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, intent_embeddings)[0]
    idx = int(torch.argmax(scores).item())
    return intent_keys[idx], float(scores[idx])

# =========================================================
# 5. Analysis functions
# =========================================================

def find_outliers(df):
    values = df["OBS_VALUE"].astype(float).values.reshape(-1, 1)
    iso = IsolationForest(contamination=0.01, random_state=42)
    pred = iso.fit_predict(values)
    return df[pred == -1]

def get_top_values(df, k=10):
    return df.nlargest(k, "OBS_VALUE")[["REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]]

def cluster_countries(df, k=5):
    means = df.groupby("REF_AREA_LABEL")["OBS_VALUE"].mean()
    scaled = StandardScaler().fit_transform(means.values.reshape(-1, 1))

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled)

    return pd.DataFrame({
        "Country": means.index,
        "MeanInflation": means.values,
        "Cluster": labels
    })

def trend_regression(df):
    df2 = df.copy()
    df2["t"] = pd.factorize(df2["TIME_PERIOD"])[0]

    model = LinearRegression()
    model.fit(df2[["t"]], df2["OBS_VALUE"])

    pred = model.predict(df2[["t"]])

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "predicted": pred.tolist(),
    }

# =========================================================
# 6. Query handler
# =========================================================

def handle_query(query, df):
    intent, score = detect_intent(query)

    if intent == "outliers":
        return {"intent": intent, "score": score, "result": find_outliers(df)}

    if intent == "top":
        return {"intent": intent, "score": score, "result": get_top_values(df)}

    if intent == "cluster":
        return {"intent": intent, "score": score, "result": cluster_countries(df)}

    if intent == "trend":
        return {"intent": intent, "score": score, **trend_regression(df)}

    return {"error": "unknown intent"}
