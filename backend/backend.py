import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path="food_price_inflation.csv"):
    df = pd.read_csv(path)
    return df

def embed_corpus(df, cache_file="corpus_embeddings.pt"):
    texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        return torch.load(cache_file)

    print("Generating embeddings...")
    emb = embedder.encode(
        texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    torch.save(emb, cache_file)
    return emb

def detect_intent(query):
    q_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, intent_embeddings)[0]
    idx = int(torch.argmax(scores).item())
    return intent_keys[idx], float(scores[idx])

def find_outliers(df):
    values = df["OBS_VALUE"].astype(float).values.reshape(-1, 1)
    iso = IsolationForest(contamination=0.01, random_state=42)
    pred = iso.fit_predict(values)
    return df[pred == -1]

def get_top_values(df, k=10):
    return df.nlargest(k, "OBS_VALUE")[["REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]]

def cluster_countries(df, k=5):
    country_means = df.groupby("REF_AREA_LABEL")["OBS_VALUE"].mean()
    scaled = StandardScaler().fit_transform(country_means.values.reshape(-1, 1))

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled)

    return pd.DataFrame({
        "Country": country_means.index,
        "MeanInflation": country_means.values,
        "Cluster": labels
    })

def plot_top10(df, out_file="top10_inflation.png"):
    data = df.dropna(subset=["OBS_VALUE"]).nlargest(10, "OBS_VALUE")

    if data.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="OBS_VALUE", y="REF_AREA_LABEL", palette="Reds")
    plt.title("Top 10 Countries with Highest Food Inflation")
    plt.xlabel("Inflation (%)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Plot saved as {out_file}")

def handle_query(query, df):
    intent, score = detect_intent(query)
    print(f"Detected intent: {intent} ({score:.2f})")

    if intent == "outliers":
        out = find_outliers(df)
        return {"intent": intent, "result": out}

    if intent == "top":
        top = get_top_values(df)
        return {"intent": intent, "result": top}

    if intent == "cluster":
        cl = cluster_countries(df)
        return {"intent": intent, "result": cl}

    if intent == "trend":
        # simplest trend possible
        df2 = df.copy()
        df2["t"] = pd.factorize(df2["TIME_PERIOD"])[0]

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(df2[["t"]], df2["OBS_VALUE"])
        pred = model.predict(df2[["t"]])

        return {
            "intent": intent,
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "predicted": pred.tolist(),
        }

    return {"error": "Unknown intent"}

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

INTENTS = {
    "outliers": "detect outliers, anomalies, isolation forest",
    "top": "top values, highest values, biggest inflation",
    "trend": "trend, regression, line going up or down",
    "cluster": "cluster countries, similar inflation behavior",
}

intent_embeddings = embedder.encode(
    list(INTENTS.values()),
    convert_to_tensor=True,
    normalize_embeddings=True
)
intent_keys = list(INTENTS.keys())

df = load_data()
corpus_emb = embed_corpus(df)

query = "Detect anomalies."
result = handle_query(query, df)

print(result["result"])

# Example: generate plot
if result["intent"] == "top":
    plot_top10(result["result"])
