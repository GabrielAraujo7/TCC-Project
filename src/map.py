import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np


def gerar_mapa(regioes: list):
    """
    Recebe lista [(lat, lon, nome)] e retorna um folium.Map.
    """
    if not regioes:
        return folium.Map(location=[-3.7319, -38.5267], zoom_start=12)

    center = [np.mean([r[0] for r in regioes]), np.mean([r[1] for r in regioes])]
    mapa = folium.Map(location=center, zoom_start=12)

    for lat, lon, nome in regioes:
        folium.Marker([lat, lon], popup=nome, icon=folium.Icon(color="green")).add_to(mapa)

    return mapa


file_path = "Projeto.xlsx"
df = pd.read_excel(file_path, sheet_name="BASE")
df.columns = [str(c).strip().upper() for c in df.columns]

col_lat, col_lon = "LATITUDE", "LONGITUDE"
col_class, col_cliente = "CLASSE SOCIAL", "CLIENTE"
col_tipo = "TIPO COMERCIAL"
possible_dev_cols = ["% DEV", "PCT_DEV", "PERCENT_DEV", "PERCENTUAL DEV", "PERCENT"]
col_dev = next((c for c in possible_dev_cols if c in df.columns), None)

df = df.dropna(subset=[col_class, col_lat, col_lon])


df[col_class] = df[col_class].astype(str).str.strip().str.upper().str[0]
df = df[df[col_class].isin(list("ABC"))]

def fix_coord(v):
    v = float(v)
    return v/1_000_000 if abs(v) > 1000 else v
df[col_lat] = df[col_lat].apply(fix_coord)
df[col_lon] = df[col_lon].apply(fix_coord)

def weight_by_class(classe):
    classe = str(classe).strip().upper()
    if classe == "A":
        return 50000
    elif classe == "B":
        return 20000
    elif classe == "C":
        return 1000
    else:
        return 0
df["HEAT_PESO"] = df[col_class].apply(weight_by_class)

def marker_color_by_percent(percent_dev):
    try:
        v = float(percent_dev)
    except Exception:
        return "blue"
    if v < 3:
        return "green"
    elif 3 <= v < 5:
        return "orange"
    else:
        return "red"

max_w = df["HEAT_PESO"].max() if len(df) else 1
df["HEAT_W"] = df["HEAT_PESO"] / max_w

center = [df[col_lat].mean(), df[col_lon].mean()]
m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

heat_data = df[[col_lat, col_lon, "HEAT_W"]].values.tolist()
HeatMap(
    heat_data,
    radius=22, blur=24, max_zoom=14, min_opacity=0.2,
    name="Heatmap (peso por classe — sua lógica)"
).add_to(m)

grupos = {}
tipos = sorted(df[col_tipo].dropna().unique().tolist())
for tipo in tipos:
    grupos[tipo] = folium.FeatureGroup(name=f"{tipo} (marcadores)", show=False)
    grupos[tipo].add_to(m)

for _, row in df.iterrows():
    tipo = row.get(col_tipo, "Sem Tipo")
    if pd.isna(tipo):
        tipo = "Sem Tipo"
        if tipo not in grupos:
            grupos[tipo] = folium.FeatureGroup(name=f"{tipo} (marcadores)", show=False)
            grupos[tipo].add_to(m)

    if col_dev and pd.notna(row.get(col_dev, None)):
        marker_color = marker_color_by_percent(row[col_dev])
        extra = f"<b>% DEV:</b> {row[col_dev]}<br>"
    else:
        class_color = {"A": "green", "B": "orange", "C": "red"}[row[col_class]]
        marker_color = class_color
        extra = ""

    popup_html = f"""
    <div style='font-size:13px'>
        <b>Cliente:</b> {row[col_cliente]}<br>
        <b>Classe:</b> {row[col_class]}<br>
        <b>Tipo Comercial:</b> {tipo}<br>
         </div>
    """
    folium.Marker(
        location=[row[col_lat], row[col_lon]],
        popup=folium.Popup(popup_html, max_width=320),
        icon=folium.Icon(color=marker_color, icon="shopping-cart", prefix="fa")
    ).add_to(grupos[tipo])

folium.LayerControl(collapsed=False).add_to(m)

legend_html = f"""
<div style="
    position: fixed; bottom: 20px; left: 20px; z-index: 9999;
    background: rgba(255,255,255,0.95); padding: 12px 14px; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: Arial, sans-serif; font-size: 13px;">
  <div style="font-weight: bold; margin-bottom: 6px;">Legenda</div>
  <div style="margin-bottom:6px;">
    <div><b>Heatmap</b>: intensidade proporcional ao peso por classe</div>
    <div>Classe A → 🌡️&nbsp; | &nbsp; Classe B → 🟠 &nbsp; | &nbsp; Classe C → ❄️</div>
  </div>
  <div style="border-top:1px solid #ddd; padding-top:6px; margin-top:6px;">
    <div style="margin-bottom:4px;"><b>Cor do marcador</b></div>
    
    
  </div>
  <div style="margin-top:6px;color:#666;">Use o painel no topo direito para filtrar por <b>Tipo Comercial</b>.</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

out_path = "mapa_heat_peso_user_logic_sem_gradient.html"
m.save(out_path)
out_path
