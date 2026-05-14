import base64
import html
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from PIL import Image


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"

INPUT_PATH = DATASET_DIR / "fusion_kalman_with_type.csv"
OUTPUT_CSV = DATASET_DIR / "hasil_deteksi_kapal_ikan_mencurigakan_kalman.csv"
OUTPUT_HTML = DATASET_DIR / "peta_deteksi_kapal_ikan_mencurigakan_kalman.html"
OUTPUT_ALERT = DATASET_DIR / "alert_kapal_mencurigakan_kalman.csv"

PATCH_CAL_FOLDER = DATASET_DIR / "PATCH_CAL"
PATCH_RGB_FOLDER = DATASET_DIR / "PATCH_RGB"

KALMAN_SHIFT_THRESHOLD_M = 500.0


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan pada {label}: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )


def pick_type_column(df):
    candidates = [
        "Ship_type",
        "Ship_Type",
        "category",
        "Category",
        "vessel_type",
        "label",
        "shiptype",
        "class",
    ]
    return next((col for col in candidates if col in df.columns), None)


def ensure_optional_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            print(f"Peringatan: kolom '{col}' tidak ada. Nilai dianggap kosong.")
            df[col] = pd.NA


def safe_value(row, col, default="-"):
    if col not in row.index:
        return default
    value = row.get(col)
    if pd.isna(value):
        return default
    return html.escape(str(value))


def valid_lat_lon(lat_series, lon_series):
    return (
        lat_series.notna()
        & lon_series.notna()
        & lat_series.between(-90, 90)
        & lon_series.between(-180, 180)
    )


def add_plot_coordinates(df):
    kalman_valid = valid_lat_lon(df["Kalman_Lat"], df["Kalman_Lon"])
    ais_valid = valid_lat_lon(df["AIS_Latitude"], df["AIS_Longitude"])

    df["Plot_Lat"] = df["Kalman_Lat"].where(kalman_valid, df["AIS_Latitude"])
    df["Plot_Lon"] = df["Kalman_Lon"].where(kalman_valid, df["AIS_Longitude"])
    df["koordinat_sumber"] = "Kalman"
    df.loc[~kalman_valid & ais_valid, "koordinat_sumber"] = "Fallback AIS"

    return df, kalman_valid, ais_valid


def hitung_skor(row, type_col):
    skor = 0
    alasan = []

    ship_type = str(row.get(type_col, "")).lower()
    mmsi = str(row.get("MMSI", "")).strip()
    sog = pd.to_numeric(row.get("Sog"), errors="coerce")
    panjang = pd.to_numeric(row.get("AIS_Length"), errors="coerce")
    lebar = pd.to_numeric(row.get("AIS_Width"), errors="coerce")
    gross_tonnage = pd.to_numeric(row.get("Gross_tonnage"), errors="coerce")
    kalman_shift = pd.to_numeric(row.get("Kalman_Shift_m"), errors="coerce")

    if "fishing" in ship_type:
        skor += 2
        alasan.append("jenis kapal penangkap ikan")

    if "other" in ship_type or "unknown" in ship_type:
        skor += 1
        alasan.append("jenis kapal tidak spesifik")

    if pd.notna(sog) and sog < 1:
        skor += 2
        alasan.append("kecepatan sangat rendah / hampir berhenti")
    elif pd.notna(sog) and 1 <= sog <= 5:
        skor += 1
        alasan.append("kecepatan rendah-menengah")

    if mmsi in ["", "nan", "None", "NaN", "<NA>"] or len(mmsi) < 8:
        skor += 2
        alasan.append("MMSI kosong atau tidak valid")

    if pd.isna(panjang) or pd.isna(lebar):
        skor += 1
        alasan.append("data panjang/lebar kapal tidak lengkap")

    if pd.isna(gross_tonnage):
        skor += 1
        alasan.append("gross tonnage tidak tersedia")

    if pd.notna(panjang) and panjang <= 0:
        skor += 1
        alasan.append("panjang kapal tidak wajar")

    if pd.notna(lebar) and lebar <= 0:
        skor += 1
        alasan.append("lebar kapal tidak wajar")

    if pd.notna(kalman_shift) and kalman_shift > KALMAN_SHIFT_THRESHOLD_M:
        skor += 2
        alasan.append(
            f"pergeseran posisi Kalman besar ({kalman_shift:.2f} meter)"
        )

    if not alasan:
        alasan.append("tidak ada indikator mencurigakan utama")

    return skor, "; ".join(alasan)


def kategori_risiko(skor):
    if skor >= 5:
        return "Tinggi"
    if skor >= 3:
        return "Sedang"
    return "Rendah"


def cari_gambar_patch(patch_name, base_folder):
    if patch_name is None or pd.isna(patch_name):
        return None

    if not base_folder.exists():
        return None

    patch_text = str(patch_name).strip().replace("\\", "/")
    patch_file_name = Path(patch_text).name
    patch_stem = Path(patch_file_name).stem

    candidates = [patch_text, patch_file_name]
    for candidate in candidates:
        exact_match = list(base_folder.rglob(candidate))
        if exact_match:
            return exact_match[0]

    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    for ext in extensions:
        stem_match = list(base_folder.rglob(f"{patch_stem}{ext}"))
        if stem_match:
            return stem_match[0]

    for ext in extensions:
        contains_match = list(base_folder.rglob(f"*{patch_stem}*{ext}"))
        if contains_match:
            return contains_match[0]

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
    except Exception as exc:
        print(f"Gagal membaca gambar {image_path}: {exc}")
        return None


def buat_html_gambar_sar(patch_name):
    gambar_path = None

    if PATCH_RGB_FOLDER.exists():
        gambar_path = cari_gambar_patch(patch_name, PATCH_RGB_FOLDER)

    if gambar_path is None and PATCH_CAL_FOLDER.exists():
        gambar_path = cari_gambar_patch(patch_name, PATCH_CAL_FOLDER)

    if gambar_path is None:
        return "<br><i>Gambar SAR tidak ditemukan</i>"

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


def pilih_waktu_scene(row):
    for col in ["Detection_Time", "SAR_Start_Time", "scene"]:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    return pd.NA


def buat_peta(df, type_col):
    plot_valid = valid_lat_lon(df["Plot_Lat"], df["Plot_Lon"])
    df_plot = df.loc[plot_valid].copy()

    if df_plot.empty:
        print("Tidak ada titik valid untuk peta. Peta kosong tetap dibuat.")
        fmap = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles="OpenStreetMap")
        fmap.save(OUTPUT_HTML)
        return

    center = [df_plot["Plot_Lat"].median(), df_plot["Plot_Lon"].median()]
    fmap = folium.Map(location=center, zoom_start=5, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Peta Dasar Berwarna",
        control=True,
    ).add_to(fmap)

    warna_risiko = {
        "Tinggi": "red",
        "Sedang": "orange",
        "Rendah": "green",
    }

    for risiko in ["Tinggi", "Sedang", "Rendah"]:
        subset = df_plot[df_plot["kategori_risiko"] == risiko]
        fg = folium.FeatureGroup(name=f"Risiko {risiko}", show=True)
        cluster = MarkerCluster().add_to(fg)

        for _, row in subset.iterrows():
            img_html = buat_html_gambar_sar(row.get("patch_cal"))
            ship_type = safe_value(row, type_col)

            popup = f"""
            <b>Kategori Risiko:</b> {safe_value(row, "kategori_risiko")}<br>
            <b>Skor Mencurigakan:</b> {safe_value(row, "skor_mencurigakan")}<br>
            <b>Alasan:</b> {safe_value(row, "alasan")}<br>
            <hr>
            <b>MMSI:</b> {safe_value(row, "MMSI")}<br>
            <b>Jenis Kapal:</b> {ship_type}<br>
            <b>SOG:</b> {safe_value(row, "Sog")}<br>
            <b>COG:</b> {safe_value(row, "Cog")}<br>
            <b>True Head:</b> {safe_value(row, "True_Head")}<br>
            <hr>
            <b>Koordinat AIS asli:</b><br>
            Lat {safe_value(row, "AIS_Latitude")}, Lon {safe_value(row, "AIS_Longitude")}<br>
            <b>Koordinat Kalman:</b><br>
            Lat {safe_value(row, "Kalman_Lat")}, Lon {safe_value(row, "Kalman_Lon")}<br>
            <b>Kalman Shift:</b> {safe_value(row, "Kalman_Shift_m")} meter<br>
            <b>Sumber marker:</b> {safe_value(row, "koordinat_sumber")}<br>
            <hr>
            <b>Scene SAR:</b><br>{safe_value(row, "scene")}<br>
            <b>Patch CAL:</b><br>{safe_value(row, "patch_cal")}<br>
            <b>Incidence:</b> {safe_value(row, "Incidence")}<br>
            {img_html}
            """

            folium.CircleMarker(
                location=[row["Plot_Lat"], row["Plot_Lon"]],
                radius=4,
                color=warna_risiko.get(risiko, "blue"),
                fill=True,
                fill_color=warna_risiko.get(risiko, "blue"),
                fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=520),
                tooltip=f"{risiko} | MMSI {safe_value(row, 'MMSI')}",
            ).add_to(cluster)

        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(OUTPUT_HTML)


def buat_alert_ringkas(df, type_col):
    alert_df = df[df["kategori_risiko"].isin(["Tinggi", "Sedang"])].copy()
    alert_df["waktu_scene"] = alert_df.apply(pilih_waktu_scene, axis=1)
    alert_df["Ship_type_category"] = alert_df[type_col]

    output_cols = [
        "waktu_scene",
        "scene",
        "MMSI",
        "Ship_type_category",
        "kategori_risiko",
        "skor_mencurigakan",
        "alasan",
        "Kalman_Lat",
        "Kalman_Lon",
        "AIS_Latitude",
        "AIS_Longitude",
        "Kalman_Shift_m",
        "patch_cal",
    ]

    for col in output_cols:
        if col not in alert_df.columns:
            alert_df[col] = pd.NA

    alert_df[output_cols].to_csv(OUTPUT_ALERT, index=False)
    return len(alert_df)


def main():
    print("deteksi_kapal_ilegal_kalman_alert.py mulai dijalankan...")
    print("Input      :", INPUT_PATH)
    print("Output CSV :", OUTPUT_CSV)
    print("Output peta:", OUTPUT_HTML)
    print("Output alert ringkas:", OUTPUT_ALERT)
    print("Catatan: alert CSV ini adalah hasil analisis sistem, bukan SSAS/distress kapal.")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}. "
            "Jalankan integrasi_kalman_ke_fusion.py terlebih dahulu."
        )

    df = pd.read_csv(INPUT_PATH)
    print("Jumlah data awal:", len(df))
    print("Kolom input:", list(df.columns))

    require_columns(
        df,
        [
            "MMSI",
            "AIS_Latitude",
            "AIS_Longitude",
            "Kalman_Lat",
            "Kalman_Lon",
            "Kalman_Shift_m",
        ],
        "dataset/fusion_kalman_with_type.csv",
    )

    type_col = pick_type_column(df)
    if type_col is None:
        raise KeyError(
            "Kolom tipe kapal tidak ditemukan. Butuh salah satu dari "
            "Ship_type, Ship_Type, category, Category, vessel_type, label, shiptype, class."
        )
    print("Kolom tipe kapal yang dipakai:", type_col)

    ensure_optional_columns(
        df,
        [
            "Sog",
            "Cog",
            "True_Head",
            "AIS_Length",
            "AIS_Width",
            "Gross_tonnage",
            "Deadweight",
            "Draught",
            "scene",
            "patch_cal",
            "Incidence",
        ],
    )

    numeric_cols = [
        "Sog",
        "Cog",
        "True_Head",
        "AIS_Latitude",
        "AIS_Longitude",
        "AIS_Length",
        "AIS_Width",
        "Draught",
        "Gross_tonnage",
        "Deadweight",
        "Kalman_Lat",
        "Kalman_Lon",
        "Kalman_Shift_m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[type_col] = df[type_col].fillna("Unknown").astype(str)

    fishing_mask = df[type_col].str.contains("Fishing", case=False, na=False)
    print("Jumlah data fishing:", int(fishing_mask.sum()))
    if fishing_mask.any():
        df = df.loc[fishing_mask].copy()
        print("Mengikuti aturan lama: hanya data fishing yang diproses.")
    else:
        print("Peringatan: tidak ada tipe Fishing. Semua data diproses.")

    df, kalman_valid, ais_valid = add_plot_coordinates(df)
    plot_valid = valid_lat_lon(df["Plot_Lat"], df["Plot_Lon"])

    print("Titik memakai koordinat Kalman:", int((kalman_valid & plot_valid).sum()))
    print("Titik fallback ke AIS mentah  :", int((~kalman_valid & ais_valid).sum()))
    print("Titik koordinat tidak valid    :", int((~plot_valid).sum()))

    df[["skor_mencurigakan", "alasan"]] = df.apply(
        lambda row: pd.Series(hitung_skor(row, type_col)),
        axis=1,
    )
    df["kategori_risiko"] = df["skor_mencurigakan"].apply(kategori_risiko)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    buat_peta(df, type_col)
    jumlah_alert = buat_alert_ringkas(df, type_col)

    print("CSV hasil deteksi berhasil dibuat:", OUTPUT_CSV)
    print("Peta hasil deteksi berhasil dibuat:", OUTPUT_HTML)
    print("Alert ringkas berhasil dibuat:", OUTPUT_ALERT)
    print("Jumlah alert Tinggi/Sedang:", jumlah_alert)
    print("\nJumlah per kategori risiko:")
    print(df["kategori_risiko"].value_counts())

    if len(df) > 0:
        preview_cols = [
            "MMSI",
            type_col,
            "Sog",
            "Kalman_Shift_m",
            "skor_mencurigakan",
            "kategori_risiko",
            "alasan",
            "Kalman_Lat",
            "Kalman_Lon",
        ]
        print("\nContoh hasil deteksi:")
        print(df[preview_cols].head(10))

    print("deteksi_kapal_ilegal_kalman_alert.py selesai dijalankan.")


if __name__ == "__main__":
    main()
