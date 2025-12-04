# -------------------------
# 1) Column type detection
# -------------------------
import pandas as pd
import numpy as np
import math
from datetime import datetime
import difflib

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") 

def detect_column_types(df, sample_frac=0.1):
    """
    Detects types for each column in df.
    Returns a dict: {col_name: {'type': one_of('numeric','datetime','categorical','text'), 'details': {...}}}
    Uses heuristics and sampling to be fast for large dataframes.
    """
    types = {}
    n = len(df)
    # build sample indices
    if n == 0:
        return {}
    sample_n = max(1, int(n * sample_frac))
    sample = df.sample(sample_n, random_state=0) if n > sample_n else df

    for col in df.columns:
        col_series = sample[col].dropna()
        col_info = {'type': None, 'details': {}}

        # 1) Try numeric
        # if a high proportion can be coerced to numeric, treat as numeric
        coerced = pd.to_numeric(col_series, errors='coerce')
        num_valid = coerced.notna().sum()
        num_frac = num_valid / max(1, len(col_series))
        if num_frac >= 0.8 and len(col_series) > 0:
            col_info['type'] = 'numeric'
            col_info['details']['num_frac'] = num_frac
            # include global stats
            try:
                full_coerced = pd.to_numeric(df[col], errors='coerce')
                col_info['details']['mean'] = float(full_coerced.mean(skipna=True))
                col_info['details']['std'] = float(full_coerced.std(skipna=True))
            except Exception:
                pass
            types[col] = col_info
            continue

        # 2) Try datetime
        # attempt to parse dates
        parsed = pd.to_datetime(col_series, errors='coerce', infer_datetime_format=True)
        dt_valid = parsed.notna().sum()
        dt_frac = dt_valid / max(1, len(col_series))
        if dt_frac >= 0.6 and len(col_series) > 0:  # lower threshold, dates are messy
            col_info['type'] = 'datetime'
            col_info['details']['dt_frac'] = dt_frac
            types[col] = col_info
            continue

        # 3) Categorical vs text
        unique_frac = col_series.nunique() / max(1, len(col_series))
        if unique_frac <= 0.05 or col_series.nunique() < 50:
            col_info['type'] = 'categorical'
            col_info['details']['unique_count'] = int(col_series.nunique())
            types[col] = col_info
            continue

        # default to text
        col_info['type'] = 'text'
        col_info['details']['unique_count'] = int(col_series.nunique())
        types[col] = col_info

    return types

# quick helper for fuzzy column matching
def find_best_column_matches(query_tokens, columns, n_matches=3, cutoff=0.6):
    """
    Attempts to match tokens (list of strings) or a single term to column names.
    Returns a list of matched column names ordered by score.
    Uses substring matching and difflib fuzzy matching.
    """
    if isinstance(query_tokens, str):
        tokens = [query_tokens.lower()]
    else:
        tokens = [t.lower() for t in query_tokens if t]

    scores = {}
    cols_lower = {c.lower(): c for c in columns}
    for token in tokens:
        # direct substring matches
        for c in columns:
            c_low = c.lower()
            if token in c_low or c_low in token:
                scores[c] = scores.get(c, 0) + 1.0
        # fuzzy matches (difflib)
        close = difflib.get_close_matches(token, cols_lower.keys(), n=n_matches, cutoff=cutoff)
        for k in close:
            scores[cols_lower[k]] = scores.get(cols_lower[k], 0) + 0.8
    # return sorted by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked]

# -------------------------
# 2) Intent detection using embeddings
# -------------------------
import torch
from sentence_transformers import SentenceTransformer, util

# load embedder once (reusar o embedder que já tens)
# embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # assume já carregado

_INTENT_EXAMPLES = {
    "trend": [
        "show the trend over time",
        "is it increasing or decreasing over years",
        "trend of sales"
    ],
    "correlation": [
        "correlation between two variables",
        "is there a relationship between X and Y",
        "show correlation"
    ],
    "topk": [
        "top 5",
        "highest values",
        "which are the most"
    ],
    "outlier": [
        "detect anomalies",
        "which are spikes",
        "any outliers"
    ],
    "cluster": [
        "group similar items",
        "cluster countries by behavior",
        "grouping"
    ],
    "describe": [
        "summary statistics",
        "describe the dataset",
        "show mean median std"
    ],
    "filter": [
        "filter by country",
        "only for 2020",
        "where year is 2019"
    ]
}

# Precompute intent embeddings (call once)
def build_intent_embeddings(embedder):
    intents = list(_INTENT_EXAMPLES.keys())
    texts = ["; ".join(_INTENT_EXAMPLES[k]) for k in intents]
    emb = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return intents, emb

# Detect intent
def detect_intent(query, embedder, intents_list=None, intent_emb=None, threshold=0.45):
    """
    Returns (intent_key, score) where score is cosine similarity to best intent.
    If below threshold, returns ('unknown', score).
    """
    if intent_emb is None or intents_list is None:
        intents_list, intent_emb = build_intent_embeddings(embedder)

    q_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, intent_emb)[0]
    top_idx = int(torch.argmax(scores).item())
    top_score = float(scores[top_idx].item())
    top_intent = intents_list[top_idx]
    if top_score < threshold:
        return 'unknown', top_score
    return top_intent, top_score

# -------------------------
# 3) Generic scikit-learn functions
# -------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats

def run_trend_regression(df, time_col, value_col, min_samples=3, normalize_time=True):
    """
    Fits LinearRegression on (time -> value).
    Returns dict with slope, intercept, r2, n_samples, years, observed, predicted.
    If not enough samples returns status insufficient_data.
    """
    # prepare
    sub = df[[time_col, value_col]].dropna()
    # try parse time to datetime if not already
    if not np.issubdtype(sub[time_col].dtype, np.number):
        try:
            sub[time_col] = pd.to_datetime(sub[time_col], errors='coerce')
            sub = sub.dropna(subset=[time_col])
            # convert to numeric (year or ordinal)
            sub['time_numeric'] = sub[time_col].dt.year
        except Exception:
            # if cannot parse, try numeric coercion
            sub['time_numeric'] = pd.to_numeric(sub[time_col], errors='coerce')
    else:
        sub['time_numeric'] = sub[time_col].astype(float)

    sub = sub.dropna(subset=['time_numeric', value_col])
    if len(sub) < min_samples:
        return {'status': 'insufficient_data', 'n_samples': int(len(sub))}

    # optionally normalize time to zero-based index
    if normalize_time:
        sub = sub.sort_values('time_numeric')
        sub['t_idx'] = (sub['time_numeric'] - sub['time_numeric'].min()).astype(float)
        X = sub[['t_idx']].values
    else:
        X = sub[['time_numeric']].values

    y = pd.to_numeric(sub[value_col], errors='coerce').values
    # fit
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(stats.pearsonr(y, y_pred)[0]**2) if len(y) > 1 else 0.0
    # return arrays for plotting
    years = sub[time_col].astype(str).tolist()
    observed = [float(v) for v in y]
    predicted = [float(v) for v in y_pred]
    return {
        'status': 'ok',
        'n_samples': int(len(y)),
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'years': years,
        'observed': observed,
        'predicted': predicted
    }

def run_correlation(df, col_x, col_y, method='pearson'):
    """
    Computes correlation between two numeric columns.
    Returns correlation value and p-value (if available).
    """
    sx = pd.to_numeric(df[col_x], errors='coerce')
    sy = pd.to_numeric(df[col_y], errors='coerce')
    mask = sx.notna() & sy.notna()
    if mask.sum() < 3:
        return {'status': 'insufficient_data', 'n_samples': int(mask.sum())}
    sx = sx[mask]
    sy = sy[mask]
    if method == 'pearson':
        corr, p = stats.pearsonr(sx, sy)
    elif method == 'spearman':
        corr, p = stats.spearmanr(sx, sy)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return {'status': 'ok', 'n_samples': int(mask.sum()), 'correlation': float(corr), 'p_value': float(p)}

def run_clustering(df, numeric_cols, k=3):
    """
    Runs KMeans clustering on the provided numeric columns.
    Returns labels per-row and cluster centroids (in original scale).
    """
    sub = df[numeric_cols].copy()
    sub = sub.apply(pd.to_numeric, errors='coerce')
    sub = sub.dropna()
    if len(sub) < k:
        return {'status': 'insufficient_data', 'n_samples': len(sub)}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub.values)
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(Xs)
    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    return {
        'status': 'ok',
        'n_samples': int(len(sub)),
        'labels': labels.tolist(),
        'centroids': centroids.tolist(),
        'rows_index': sub.index.tolist()
    }

def run_outlier_detection(df, col, method='zscore', z_thresh=3.0, contamination=0.05):
    """
    Detects outliers in a single numeric column.
    method: 'zscore' or 'isolation'
    Returns list of row indices considered anomalies and their values.
    """
    s = pd.to_numeric(df[col], errors='coerce')
    mask = s.notna()
    if mask.sum() < 5:
        return {'status': 'insufficient_data', 'n_samples': int(mask.sum())}
    s_valid = s[mask]
    if method == 'zscore':
        zs = np.abs(stats.zscore(s_valid))
        anomalies = s_valid[zs > z_thresh]
        return {'status': 'ok', 'n_samples': int(len(s_valid)), 'anomalies': anomalies.to_dict()}
    elif method == 'isolation':
        iso = IsolationForest(contamination=contamination, random_state=0)
        arr = s_valid.values.reshape(-1,1)
        labels = iso.fit_predict(arr)  # -1 anomaly
        anomalies = s_valid[[i for i,lab in enumerate(labels) if lab == -1]]
        idxs = anomalies.index.tolist()
        return {'status': 'ok', 'n_samples': int(len(s_valid)), 'anomalies': dict(zip(idxs, anomalies.tolist()))}
    else:
        raise ValueError("method must be 'zscore' or 'isolation'")

# -------------------------
# 4) Integrator: handle_query
# -------------------------
import re

# build intent embeddings once
intents_list, intent_emb = build_intent_embeddings(embedder)

def extract_entities_from_query(query, df_columns):
    """
    Very simple entity extraction: split query into tokens and attempt to match
    to column names using find_best_column_matches.
    Returns ordered list of matched columns.
    """
    # tokenization simple: words of length >=3
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    tokens = [t for t in tokens if len(t) >= 3]
    matches = find_best_column_matches(tokens, df_columns, n_matches=5, cutoff=0.6)
    return matches

def handle_query(query, df, embedder, intents_list=intents_list, intent_emb=intent_emb):
    """
    Full pipeline:
    - detect intent
    - detect column matches
    - choose appropriate generic function
    - return structured result
    """
    # 1) intent
    intent, intent_score = detect_intent(query, embedder, intents_list, intent_emb)
    col_types = detect_column_types(df)
    columns = list(df.columns)

    # 2) candidate columns from query
    matched = extract_entities_from_query(query, columns)

    # fallback heuristics:
    numerics = [c for c,t in col_types.items() if t['type']=='numeric']
    datetimes = [c for c,t in col_types.items() if t['type']=='datetime']
    categoricals = [c for c,t in col_types.items() if t['type']=='categorical']

    result = {'query': query, 'intent': intent, 'intent_score': intent_score, 'matched_columns': matched}

    # Routing based on intent
    if intent == 'trend':
        # pick time column
        time_col = None
        value_col = None
        # if matched includes something that is datetime, pick it
        for c in matched:
            if col_types.get(c, {}).get('type') == 'datetime':
                time_col = c
                break
        # else use any datetime
        if time_col is None and datetimes:
            time_col = datetimes[0]
        # pick value column: first matched numeric, else any numeric
        for c in matched:
            if col_types.get(c, {}).get('type') == 'numeric':
                value_col = c
                break
        if value_col is None and numerics:
            # if column names contain 'price' or 'value' prefer them
            prefer = [c for c in numerics if any(k in c.lower() for k in ['price','value','amount','obs','inflation','rate','count'])]
            value_col = prefer[0] if prefer else numerics[0]

        if time_col is None:
            # try to infer 'year' or 'date' columns by name
            candidates = [c for c in columns if any(k in c.lower() for k in ['year','date','time','period'])]
            time_col = candidates[0] if candidates else None

        if value_col is None:
            return {**result, 'error': 'no numeric column found to run trend'}

        res = run_trend_regression(df, time_col, value_col)
        return {**result, 'analysis': res, 'time_col': time_col, 'value_col': value_col}

    elif intent == 'correlation':
        # need two numeric cols
        candidates = matched + numerics
        # dedupe preserving order
        seen = []
        for c in candidates:
            if c not in seen and c in df.columns:
                seen.append(c)
        if len(seen) < 2:
            return {**result, 'error': 'need at least two numeric columns for correlation', 'numeric_columns': numerics}
        col_x, col_y = seen[0], seen[1]
        res = run_correlation(df, col_x, col_y)
        return {**result, 'analysis': res, 'col_x': col_x, 'col_y': col_y}

    elif intent == 'topk':
        # pick numeric + categorical
        if not numerics:
            return {**result, 'error': 'no numeric columns found for top-k'}
        num = numerics[0]
        # try to find categorical in matched
        cat = None
        for c in matched:
            if col_types.get(c, {}).get('type') == 'categorical':
                cat = c
                break
        if cat is None and categoricals:
            cat = categoricals[0]
        # compute topk means per category
        dfc = df[[cat, num]].dropna() if cat else df[[num]].dropna()
        if cat:
            agg = dfc.groupby(cat)[num].mean().reset_index().sort_values(num, ascending=False).head(10)
            return {**result, 'analysis': {'top_by_category': agg.to_dict(orient='records'), 'num_col': num, 'cat_col': cat}}
        else:
            top = dfc[num].nlargest(10).tolist()
            return {**result, 'analysis': {'top_values': top, 'num_col': num}}

    elif intent == 'cluster':
        # pick up to 3 numeric cols
        if not numerics:
            return {**result, 'error': 'no numeric columns found for clustering'}
        chosen = numerics[:3]
        res = run_clustering(df, chosen, k=3)
        return {**result, 'analysis': res, 'used_numeric_cols': chosen}

    elif intent == 'outlier':
        # pick numeric column
        chosen = None
        for c in matched:
            if col_types.get(c, {}).get('type') == 'numeric':
                chosen = c
                break
        if chosen is None and numerics:
            chosen = numerics[0]
        if chosen is None:
            return {**result, 'error': 'no numeric column found for outlier detection'}
        res = run_outlier_detection(df, chosen, method='isolation', contamination=0.05)
        return {**result, 'analysis': res, 'used_col': chosen}

    elif intent == 'describe' or intent == 'unknown':
        # generic describe / fallback
        summary = {}
        for c, info in col_types.items():
            if info['type'] == 'numeric':
                summary[c] = df[c].describe().to_dict()
            elif info['type'] == 'datetime':
                summary[c] = {'type': 'datetime'}
            else:
                summary[c] = {'type': info['type'], 'unique': int(df[c].nunique())}
        return {**result, 'analysis': {'summary': summary}}

    else:
        return {**result, 'error': 'intent not handled'}
