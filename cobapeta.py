import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv_path = HERE / "dataset" / "fusion_minimal.csv"
out_html = HERE / "dataset" / "map_fusion_points.html"

print("Reading:", csv_path)
df = pd.read_csv(csv_path)

df = df.dropna(subset=["AIS_Latitude", "AIS_Longitude"])
df = df[(df["AIS_Latitude"].between(-90, 90)) & (df["AIS_Longitude"].between(-180, 180))]

center = [df["AIS_Latitude"].median(), df["AIS_Longitude"].median()]
m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

cluster = MarkerCluster().add_to(m)

max_points = min(len(df), 5673)
for _, r in df.head(max_points).iterrows():
    popup = (
        f"<b>MMSI:</b> {r.get('MMSI')}<br>"
        f"<b>SOG:</b> {r.get('Sog')}<br>"
        f"<b>COG:</b> {r.get('Cog')}<br>"
        f"<b>Heading:</b> {r.get('True_Head')}<br><hr>"
        f"<b>SAR scene:</b> {r.get('scene')}<br>"
        f"<b>patch_cal:</b> {r.get('patch_cal')}<br>"
        f"<b>Incidence:</b> {r.get('Incidence')}<br>"
        f"<b>Ship LxW:</b> {r.get('AIS_Length')} x {r.get('AIS_Width')}<br>"
    )
    folium.CircleMarker(
        location=[r["AIS_Latitude"], r["AIS_Longitude"]],
        radius=3,
        fill=True,
        popup=folium.Popup(popup, max_width=350),
    ).add_to(cluster)

m.save(out_html)
print("✅ Saved:", out_html)
print("Points plotted:", max_points, "of", len(df))
