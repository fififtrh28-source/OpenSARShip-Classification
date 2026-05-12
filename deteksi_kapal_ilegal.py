import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# =========================
# LOKASI FILE
# =========================
HERE = Path(__file__).resolve().parent
csv_path = HERE / "dataset" / "fusion_minimal_with_type.csv"

out_csv = HERE / "dataset" / "hasil_deteksi_kapal_ikan_mencurigakan1.csv"
out_html = HERE / "dataset" / "peta_deteksi_kapal_ikan_mencurigakan1.html"
patch_cal_folder = HERE / "dataset" / "PATCH_CAL"
patch_rgb_folder = HERE / "dataset" / "PATCH_RGB"

print("Membaca file:", csv_path)

if not csv_path.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {csv_path}\n"
        "Pastikan fusion_minimal_with_type.csv ada di folder dataset."
    )

# =========================
# BACA DATA
# =========================
df = pd.read_csv(csv_path)

print("Jumlah data awal:", len(df))
print("Kolom:", list(df.columns))

# =========================
# BERSIHKAN KOORDINAT
# =========================
df = df.dropna(subset=["AIS_Latitude", "AIS_Longitude"])
df = df[
    (df["AIS_Latitude"].between(-90, 90)) &
    (df["AIS_Longitude"].between(-180, 180))
]

print("Jumlah data dengan koordinat valid:", len(df))

# =========================
# CEK KOLOM JENIS KAPAL
# =========================
type_col = "Ship_type"

if type_col not in df.columns:
    raise ValueError(
        f"Kolom {type_col} tidak ditemukan.\n"
        f"Kolom yang tersedia: {list(df.columns)}"
    )

df[type_col] = df[type_col].fillna("Unknown").astype(str)
# Filter hanya kapal ikan
df = df[df[type_col].str.contains("Fishing", case=False, na=False)]

print("Jumlah data kapal ikan:", len(df))

# =========================
# KONVERSI KOLOM NUMERIK
# =========================
numeric_cols = [
    "Sog",
    "Cog",
    "True_Head",
    "AIS_Length",
    "AIS_Width",
    "Draught",
    "Gross_tonnage",
    "Deadweight",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# ATURAN DETEKSI
# =========================
def hitung_skor(row):
    skor = 0
    alasan = []

    ship_type = str(row.get(type_col, "")).lower()
    mmsi = str(row.get("MMSI", "")).strip()
    sog = row.get("Sog")
    panjang = row.get("AIS_Length")
    lebar = row.get("AIS_Width")
    gross_tonnage = row.get("Gross_tonnage")

    # 1. Kapal penangkap ikan lebih relevan untuk dugaan illegal fishing
    if "fishing" in ship_type:
        skor += 2
        alasan.append("jenis kapal penangkap ikan")

    # 2. Kapal tidak dikenal / tipe lain
    if "other" in ship_type or "unknown" in ship_type:
        skor += 1
        alasan.append("jenis kapal tidak spesifik")

    # 3. Kecepatan sangat rendah
    if pd.notna(sog) and sog < 1:
        skor += 2
        alasan.append("kecepatan sangat rendah / hampir berhenti")

    # 4. Kecepatan rendah-menengah, sering muncul pada pola operasi kapal
    elif pd.notna(sog) and 1 <= sog <= 5:
        skor += 1
        alasan.append("kecepatan rendah-menengah")

    # 5. MMSI kosong atau tidak valid
    if mmsi in ["", "nan", "None", "NaN"] or len(mmsi) < 8:
        skor += 2
        alasan.append("MMSI kosong atau tidak valid")

    # 6. Ukuran kapal tidak lengkap
    if pd.isna(panjang) or pd.isna(lebar):
        skor += 1
        alasan.append("data panjang/lebar kapal tidak lengkap")

    # 7. Gross tonnage tidak tersedia
    if pd.isna(gross_tonnage):
        skor += 1
        alasan.append("gross tonnage tidak tersedia")

    # 8. Dimensi tidak wajar
    if pd.notna(panjang) and panjang <= 0:
        skor += 1
        alasan.append("panjang kapal tidak wajar")

    if pd.notna(lebar) and lebar <= 0:
        skor += 1
        alasan.append("lebar kapal tidak wajar")

    if len(alasan) == 0:
        alasan.append("tidak ada indikator mencurigakan utama")

    return skor, "; ".join(alasan)


df[["skor_mencurigakan", "alasan"]] = df.apply(
    lambda row: pd.Series(hitung_skor(row)),
    axis=1
)

# =========================
# KATEGORI RISIKO
# =========================
def kategori_risiko(skor):
    if skor >= 5:
        return "Tinggi"
    elif skor >= 3:
        return "Sedang"
    else:
        return "Rendah"


df["kategori_risiko"] = df["skor_mencurigakan"].apply(kategori_risiko)

# =========================
# SIMPAN CSV HASIL
# =========================
df.to_csv(out_csv, index=False)

def cari_gambar_patch(patch_name, base_folder):
    """
    Mencari gambar patch SAR secara rekursif sampai ke subfolder.
    Cocok untuk file .tif, .tiff, .png, .jpg, .jpeg.
    """
    if patch_name is None:
        return None

    patch_name = str(patch_name).strip()
    patch_stem = Path(patch_name).stem

    if not base_folder.exists():
        return None

    # Cari nama file persis dulu
    exact_match = list(base_folder.rglob(patch_name))
    if exact_match:
        return exact_match[0]

    # Cari berdasarkan stem tanpa ekstensi
    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    for ext in extensions:
        hasil = list(base_folder.rglob(f"{patch_stem}{ext}"))
        if hasil:
            return hasil[0]

    # Cari yang mengandung nama stem
    for ext in extensions:
        hasil = list(base_folder.rglob(f"*{patch_stem}*{ext}"))
        if hasil:
            return hasil[0]

    return None


def gambar_ke_base64(image_path, ukuran=(180, 180)):
    try:
        img = Image.open(image_path)
        img = img.convert("L")
        img.thumbnail(ukuran)

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    except Exception as e:
        print(f"Gagal membaca gambar {image_path}: {e}")
        return None


def buat_html_gambar_sar(patch_name):
    gambar_path = None

    if patch_rgb_folder.exists():
        gambar_path = cari_gambar_patch(patch_name, patch_rgb_folder)

    if gambar_path is None and patch_cal_folder.exists():
        gambar_path = cari_gambar_patch(patch_name, patch_cal_folder)

    if gambar_path is None:
        print("Gambar tidak ditemukan untuk:", patch_name)
        return "<br><i>Gambar SAR tidak ditemukan</i>"

    print("Gambar ditemukan:", gambar_path)

    img_base64 = gambar_ke_base64(gambar_path)

    if img_base64 is None:
        return "<br><i>Gambar SAR gagal dibaca</i>"

    return f"""
    <br>
    <div style="margin-top:8px;">
        <b>Gambar Patch SAR:</b><br>
        <img src="{img_base64}" width="180" style="border:1px solid #999; margin-top:5px;">
    </div>
    """
# =========================
# BUAT PETA
# =========================
center = [
    df["AIS_Latitude"].median(),
    df["AIS_Longitude"].median()
]

m = folium.Map(location=center, zoom_start=5, tiles=None)

# Peta dasar berwarna
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Peta Dasar Berwarna",
    control=True
).add_to(m)

warna_risiko = {
    "Tinggi": "red",
    "Sedang": "orange",
    "Rendah": "green",
}

# Layer checkbox berdasarkan risiko
for risiko in ["Tinggi", "Sedang", "Rendah"]:
    subset = df[df["kategori_risiko"] == risiko]

    fg = folium.FeatureGroup(
        name=f"Risiko {risiko}",
        show=True
    )

    cluster = MarkerCluster().add_to(fg)

    for _, r in subset.iterrows():
        img_html = buat_html_gambar_sar(r.get("patch_cal"))
        popup = (
            f"<b>Kategori Risiko:</b> {r.get('kategori_risiko')}<br>"
            f"<b>Skor Mencurigakan:</b> {r.get('skor_mencurigakan')}<br>"
            f"<b>Alasan:</b> {r.get('alasan')}<br>"
            f"<hr>"
            f"<b>Jenis Kapal:</b> {r.get(type_col)}<br>"
            f"<b>MMSI:</b> {r.get('MMSI')}<br>"
            f"<b>SOG / Kecepatan:</b> {r.get('Sog')}<br>"
            f"<b>COG / Arah Gerak:</b> {r.get('Cog')}<br>"
            f"<b>Haluan:</b> {r.get('True_Head')}<br>"
            f"<b>Panjang Kapal:</b> {r.get('AIS_Length')}<br>"
            f"<b>Lebar Kapal:</b> {r.get('AIS_Width')}<br>"
            f"<b>Draught / Sarat Air:</b> {r.get('Draught')}<br>"
            f"<b>Gross Tonnage:</b> {r.get('Gross_tonnage')}<br>"
            f"<b>Deadweight:</b> {r.get('Deadweight')}<br>"
            f"<b>Koordinat:</b> {r.get('AIS_Latitude')}, {r.get('AIS_Longitude')}<br>"
            f"<hr>"
            f"<b>Scene SAR:</b><br>{r.get('scene')}<br>"
            f"<b>Patch CAL:</b><br>{r.get('patch_cal')}<br>"
            f"<b>Sudut Insidensi:</b> {r.get('Incidence')}<br>"
            f"{img_html}"
        )

        folium.CircleMarker(
            location=[r["AIS_Latitude"], r["AIS_Longitude"]],
            radius=4,
            color=warna_risiko[risiko],
            fill=True,
            fill_color=warna_risiko[risiko],
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=500),
        ).add_to(cluster)

    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m.save(out_html)

# =========================
# OUTPUT TERMINAL
# =========================
print()
print("✅ Deteksi selesai")
print("CSV hasil:", out_csv)
print("Peta hasil:", out_html)
print()
print("Jumlah per kategori risiko:")
print(df["kategori_risiko"].value_counts())
print()
print("Contoh data risiko tinggi:")
print(
    df[df["kategori_risiko"] == "Tinggi"][
        ["MMSI", type_col, "Sog", "skor_mencurigakan", "alasan", "AIS_Latitude", "AIS_Longitude"]
    ].head(10)
)