import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv_path = HERE / "dataset" / "fusion_minimal.csv"
out_html = HERE / "dataset" / "map_by_shiptype_colored.html"

df = pd.read_csv(csv_path)

# ---- auto-detect kolom tipe kapal ----
type_candidates = ["category", "Category", "Ship_type", "Ship_Type", "vessel_type", "label", "shiptype", "class"]
type_col = next((c for c in type_candidates if c in df.columns), None)
if type_col is None:
    raise ValueError(
        "Kolom tipe kapal tidak ditemukan di fusion_minimal.csv. "
        "Coba regenerate fusion_minimal.csv dengan memasukkan 'category' / 'Ship_type' / 'label'.\n"
        f"Kolom yang ada sekarang: {list(df.columns)}"
    )

# ---- bersihin koordinat ----
df = df.dropna(subset=["AIS_Latitude", "AIS_Longitude"])
df = df[(df["AIS_Latitude"].between(-90, 90)) & (df["AIS_Longitude"].between(-180, 180))]

# isi type kosong -> Unknown
df[type_col] = df[type_col].fillna("Unknown").astype(str)

center = [df["AIS_Latitude"].median(), df["AIS_Longitude"].median()]
m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

# ---- warna per jenis kapal (folium color names) ----
palette = [
    "red", "blue", "green", "orange", "purple", "cadetblue", "darkred", "darkblue",
    "darkgreen", "darkpurple", "pink", "gray", "black", "lightblue", "lightgreen",
    "beige", "lightgray"
]

types = sorted(df[type_col].unique().tolist())
color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}

# ---- layer per tipe + cluster di tiap layer ----
max_points = min(len(df), 4000)  # naikin kalau mau, tapi makin berat
df_plot = df.head(max_points)

layers = {}
clusters = {}

for t in types:
    fg = folium.FeatureGroup(name=f"{t}", show=True)
    fg.add_to(m)
    layers[t] = fg
    clusters[t] = MarkerCluster().add_to(fg)

for _, r in df_plot.iterrows():
    t = r[type_col]
    color = color_map.get(t, "black")

    popup = (
        f"<b>Type:</b> {t}<br>"
        f"<b>MMSI:</b> {r.get('MMSI')}<br>"
        f"<b>SOG:</b> {r.get('Sog')}<br>"
        f"<b>COG:</b> {r.get('Cog')}<br>"
        f"<b>Heading:</b> {r.get('True_Head')}<br><hr>"
        f"<b>SAR scene:</b> {r.get('scene')}<br>"
        f"<b>patch_cal:</b> {r.get('patch_cal')}<br>"
        f"<b>Incidence:</b> {r.get('Incidence')}<br>"
    )

    folium.CircleMarker(
        location=[r["AIS_Latitude"], r["AIS_Longitude"]],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup, max_width=350),
    ).add_to(clusters[t])

folium.LayerControl(collapsed=False).add_to(m)

m.save(out_html)
print("✅ Saved:", out_html)
print("Type column used:", type_col)
print("Types:", types)
print("Points plotted:", len(df_plot), "of", len(df))
