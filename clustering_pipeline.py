
from __future__ import annotations
import os, time, argparse, hashlib, unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

# =========================
# CONFIG
# =========================
load_dotenv()
API_KEY = os.getenv("AIzaSyDJQ5kHdC0a3yqtIJS0DvYm1GgHNd4Y8vg")

# Tipos de lugar por nicho (ajuste conforme o produto)
PLACES_TYPES: Dict[str, List[str]] = {
    # Fitness / Whey (exemplo)
    "gym": ["gym"],
    "office": ["bank", "real_estate_agency", "insurance_agency", "lawyer"],  # proxy de zona comercial
    "university": ["university"],
    "supermarket": ["supermarket", "grocery_or_supermarket"],
    # Outros nichos possíveis (ex.: infantil, premium)
    # "school": ["school", "primary_school", "secondary_school"],
    # "mall": ["shopping_mall"],
}

# Raios em metros para Nearby Search
RADII = [400, 800, 1200]

# Classe socioeconômica -> ordinal
CLASSE_ORD = {"A":5, "B":4, "C":3, "D":2, "E":1}

# Cache local das consultas à API (para economizar cota)
CACHE_PATH = Path("cache_places.parquet")

# =========================
# UTIL
# =========================
def hkey(lat: float, lon: float, place_type: str, radius: int) -> str:
    s = f"{round(float(lat),6)}|{round(float(lon),6)}|{place_type}|{radius}"
    return hashlib.md5(s.encode()).hexdigest()

def load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=["h","lat","lon","type","radius","count","ts"])

def save_cache(df: pd.DataFrame) -> None:
    df.to_parquet(CACHE_PATH, index=False)

def normalize_header(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.strip().upper().replace("  ", " ")
    s = s.replace("-", " ").replace("/", " ").replace("\\", " ")
    return s

# =========================
# GOOGLE PLACES
# =========================
def nearby_count(lat: float, lon: float, place_type: str, radius: int, session: requests.Session) -> int:
    """Conta resultados via Nearby Search com até 3 páginas."""
    base = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location": f"{lat},{lon}", "radius": radius, "type": place_type, "key": API_KEY}
    total, page = 0, 0
    while True:
        r = session.get(base, params=params, timeout=30)
        if r.status_code in (429, 500, 503):
            time.sleep(min(2**page, 10))
            page += 1
            continue
        r.raise_for_status()
        data = r.json()
        total += len(data.get("results", []))
        next_token = data.get("next_page_token")
        if not next_token or page >= 2:
            break
        time.sleep(2)  # exigência da API para liberar a próxima página
        params = {"pagetoken": next_token, "key": API_KEY}
        page += 1
    return total

def enrich_row_with_pois(row: pd.Series, cache_df: pd.DataFrame, session: requests.Session) -> Tuple[Dict[str,int], List[dict]]:
    lat, lon = float(row["lat"]), float(row["lon"])
    new_records = []
    results: Dict[str,int] = {}
    for label, types in PLACES_TYPES.items():
        for radius in RADII:
            subtotal = 0
            for tp in types:
                hk = hkey(lat, lon, tp, radius)
                cached = cache_df.loc[cache_df["h"] == hk]
                if not cached.empty:
                    cnt = int(cached.iloc[0]["count"])
                else:
                    cnt = nearby_count(lat, lon, tp, radius, session)
                    new_records.append({"h":hk,"lat":lat,"lon":lon,"type":tp,"radius":radius,"count":cnt,"ts":int(time.time())})
                subtotal += cnt
            results[f"poi_{label}_{radius}m"] = subtotal
    return results, new_records

# =========================
# FEATURE ENGINEERING
# =========================
def ensure_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Cria features para clustering:
       - classe_ord (A=5..E=1)
       - tipo_comercial (one-hot)
       - poi_* normalizados 0–1
    """
    df = df.copy()
    df["classe_ord"] = df["classe"].astype(str).str.upper().map(CLASSE_ORD).fillna(0).astype(int)

    poi_cols = [c for c in df.columns if c.startswith("poi_") and c.endswith("m")]
    for c in poi_cols:
        vals = df[c].fillna(0)
        mn, mx = vals.min(), vals.max()
        df[c + "_norm"] = 0.0 if mx == mn else (vals - mn) / (mx - mn)

    poi_norm_cols = [c for c in df.columns if c.startswith("poi_") and c.endswith("_norm")]

    df["tipo_comercial"] = df["tipo_comercial"].astype(str).str.upper().fillna("OUTROS")

    num_cols = ["classe_ord"] + poi_norm_cols
    cat_cols = ["tipo_comercial"]
    return df, num_cols, cat_cols

# =========================
# CLUSTERING
# =========================
def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    km.fit(X)
    return km

def rank_clusters(df_feat: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Ranking simples por potencial = 0.6*POIs_norm_med + 0.4*(classe_media/5)."""
    tmp = df_feat.copy()
    tmp["cluster"] = labels
    poi_norm_cols = [c for c in tmp.columns if c.startswith("poi_") and c.endswith("_norm")]
    agg = tmp.groupby("cluster").agg(
        classe_med=("classe_ord", "mean"),
        poi_med=(poi_norm_cols, "mean") if poi_norm_cols else ("classe_ord", "mean"),
    )
    if isinstance(agg["poi_med"], pd.DataFrame):
        agg["poi_med"] = agg["poi_med"].mean(axis=1)
    agg["score_potencial"] = 0.6*agg["poi_med"] + 0.4*(agg["classe_med"]/5.0)
    agg = agg.sort_values("score_potencial", ascending=False).reset_index()
    agg["ordem"] = np.arange(1, len(agg)+1)
    return agg[["cluster","classe_med","poi_med","score_potencial","ordem"]]

# =========================
# MAPA
# =========================
def cluster_colors(n: int) -> List[str]:
    base = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#1a9850", "#66bd63", "#f46d43"]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]

def make_map(df_full: pd.DataFrame, cluster_rank: pd.DataFrame, out_html: str) -> str:
    center = (df_full["lat"].median(), df_full["lon"].median())
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    dfm = df_full.merge(cluster_rank[["cluster","score_potencial","ordem"]], on="cluster", how="left")
    heat_vals = dfm[["lat","lon","score_potencial"]].dropna().values.tolist()
    if heat_vals:
        HeatMap(heat_vals, name="Heatmap (potencial por cluster)", min_opacity=0.3, radius=16, blur=14).add_to(m)

    colors = cluster_colors(dfm["cluster"].nunique())
    color_map = {c_idx: colors[i] for i, c_idx in enumerate(cluster_rank.sort_values("ordem")["cluster"])}

    for _, row in cluster_rank.sort_values("ordem").iterrows():
        c = int(row["cluster"])
        fg = folium.FeatureGroup(name=f"Cluster {c} (rank {int(row['ordem'])})", show=True)
        m.add_child(fg)
        sub = dfm[dfm["cluster"] == c]
        for _, r in sub.iterrows():
            popup = folium.Popup(
                f"""<b>{r['nome']}</b><br>
                Classe: {r['classe']} (ord={int(r['classe_ord'])})<br>
                Tipo: {r['tipo_comercial']}<br>
                Cluster: {c} (rank {int(row['ordem'])})<br>
                Score cluster: {row['score_potencial']:.2f}""",
                max_width=360
            )
            folium.CircleMarker(
                location=(r["lat"], r["lon"]),
                radius=6,
                fill=True,
                fill_opacity=0.9,
                color=color_map[c],
                fill_color=color_map[c],
                popup=popup,
                tooltip=f"{r['nome']} — C{c} (rank {int(row['ordem'])})"
            ).add_to(fg)

    cmap = LinearColormap(["#4575b4","#fee090","#d73027"],
                          vmin=float(cluster_rank["score_potencial"].min()),
                          vmax=float(cluster_rank["score_potencial"].max()))
    cmap.caption = "Potencial do cluster"
    cmap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de entrada (aceita ; ou ,)")
    parser.add_argument("--usar_api", default="false", choices=["true","false"], help="Usar Google Places? (default=false)")
    parser.add_argument("--n_clusters", type=int, default=3, help="Número de clusters KMeans")
    parser.add_argument("--out_prefix", default="clientes", help="Prefixo de arquivos de saída")
    args = parser.parse_args()

    usar_api = args.usar_api.lower() == "true"

    # 1) Leitura robusta (auto-detecção de separador + normalização de cabeçalhos)
    df = pd.read_csv(args.input, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [normalize_header(c) for c in df.columns]

    # Alias map de cabeçalhos reais -> nomes do pipeline
    alias_map = {
        "CLIENTE": "nome",
        "NOME": "nome",
        "REDE": "rede",
        "LAT": "lat",
        "LATITUDE": "lat",
        "LON": "lon",
        "LONGITUDE": "lon",
        "CLASSE SOCIAL": "classe",
        "CLASSE_SOCIAL": "classe",
        "CLASSE": "classe",
        "TIPO COMERCIAL": "tipo_comercial",
        "TIPO_COMERCIAL": "tipo_comercial",
        "TIPO": "tipo_comercial",
        "BAIRRO": "bairro",
        "CIDADE": "cidade",
    }

    # Converte colunas conhecidas
    rename_dict = {col: alias_map[col] for col in df.columns if col in alias_map}
    df = df.rename(columns=rename_dict)

    # Exige o núcleo mínimo
    ensure_columns(df, ["nome", "lat", "lon", "classe", "tipo_comercial"])

    # 2) Enriquecimento de POIs (opcional)
    cache = load_cache()
    if usar_api:
        if not API_KEY:
            raise SystemExit("Defina GOOGLE_API_KEY no .env para usar a API.")
        sess = requests.Session()
        all_new = []
        for i, row in df.iterrows():
            feats, new_records = enrich_row_with_pois(row, cache, sess)
            for k, v in feats.items():
                df.at[i, k] = v
            if new_records:
                all_new.extend(new_records)
            time.sleep(0.15)  # respeitar cota
        if all_new:
            cache = pd.concat([cache, pd.DataFrame(all_new)], ignore_index=True)
            save_cache(cache)
    else:
        # cria colunas zeradas p/ manter pipeline
        for label in PLACES_TYPES:
            for r in RADII:
                df[f"poi_{label}_{r}m"] = 0

    # 3) Salva base enriquecida
    enr_csv = f"{args.out_prefix}_enriquecidos.csv"
    df.to_csv(enr_csv, index=False)

    # 4) Features
    df_feat, num_cols, cat_cols = build_features(df)

    # 5) Pré-processamento + KMeans
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop")

    X = pre.fit_transform(df_feat[num_cols + cat_cols])

    km = fit_kmeans(X, n_clusters=args.n_clusters, random_state=42)
    labels = km.labels_
    sil = silhouette_score(X, labels) if args.n_clusters > 1 and X.shape[0] > args.n_clusters else np.nan
    print(f"KMeans n={args.n_clusters} | silhouette={sil:.3f}")

    # 6) Ranking de clusters
    rank_df = rank_clusters(df_feat, labels)
    print("\nRanking de clusters (1 = maior potencial):\n", rank_df)

    # 7) Salva clusterização
    out_df = df_feat.copy()
    out_df["cluster"] = labels
    out_csv = f"{args.out_prefix}_clusterizados.csv"
    out_df.to_csv(out_csv, index=False)

    # 8) Mapa
    out_html = make_map(out_df, rank_df, out_html=f"{args.out_prefix}_mapa_clusters.html")

    print("\nArquivos gerados:")
    print(" - Enriquecidos:", enr_csv)
    print(" - Clusterizados:", out_csv)
    print(" - Mapa:", out_html)

if __name__ == "__main__":
    main()

