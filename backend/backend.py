import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("food_price_inflation.csv")

# 2. Converter DF pra lista de textos (correção do erro!)
texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()  # ← linha corrigida

# 3. Load model (PyTorch only)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# 4. Encode ONCE and save (never recalculate)
cache_file = "corpus_embeddings.pt"
if os.path.exists(cache_file):
    print("Loading embeddings from cache...")
    corpus_embeddings = torch.load(cache_file)
else:
    print("Generating embeddings (only once)...")
    corpus_embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    torch.save(corpus_embeddings, cache_file)
    print(f"Embeddings saved to {cache_file}")

# ==================== 2. Outliers ====================
values = df['OBS_VALUE'].values.reshape(-1, 1)
iso = IsolationForest(contamination=0.01, random_state=42)
outlier_pred = iso.fit_predict(values)
outliers = df[outlier_pred == -1]
print("\nPaíses com inflação EXTREMA (outliers):")
print(outliers[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']].head(10))

# ==================== 3. Top 10 ====================
print("\nTop 10 maior inflação registrada:")
print(df.nlargest(10, 'OBS_VALUE')[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']])

print("\nTop 10 menor inflação (ou deflação):")
print(df.nsmallest(10, 'OBS_VALUE')[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']])

# ==================== 4. Clustering ====================
country_means = df.groupby('REF_AREA_LABEL')['OBS_VALUE'].mean().values.reshape(-1, 1)
scaler = StandardScaler()
country_scaled = scaler.fit_transform(country_means)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(country_scaled)
cluster_df = pd.DataFrame({
    'Country': df['REF_AREA_LABEL'].unique(),
    'Mean_Inflation': country_means.flatten(),
    'Cluster': clusters
}).sort_values('Mean_Inflation', ascending=False)
print("\nClusters de países por comportamento de inflação:")
for cluster in cluster_df['Cluster'].unique():
    countries_in_cluster = cluster_df[cluster_df['Cluster'] == cluster]['Country'].tolist()[:10]
    mean_inf = cluster_df[cluster_df['Cluster'] == cluster]['Mean_Inflation'].mean()
    print(f"Cluster {cluster} (média {mean_inf:.2f}%): {', '.join(countries_in_cluster)}")


# ==================== 5. Gráfico bonitão ====================
# Filtro pra remover outliers extremos (hiperinflação >1000%)
top10 = df.nlargest(10, 'OBS_VALUE')
top10 = top10[top10['OBS_VALUE'] < 1000]  # ← filtro aqui

plt.figure(figsize=(12, 6))
sns.barplot(data=top10, x='OBS_VALUE', y='REF_AREA_LABEL', hue='REF_AREA_LABEL', palette='Reds', legend=False)
plt.title("Top 10 Países com Maior Inflação de Alimentos (2001–2024)")
plt.xlabel("Inflação (%)")
plt.xscale('log')  # ← log scale pro eixo x, pra ver diferenças claras
plt.tight_layout()
plt.savefig("top10_inflation.png", dpi=200)
plt.close()

print("\nGráfico salvo em 'top10_inflation.png'!")