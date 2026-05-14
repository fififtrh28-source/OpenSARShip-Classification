import html
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster


HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "dataset" / "fusion_kalman_with_type.csv"
OUTPUT_HTML = HERE / "dataset" / "map_fusion_kalman_points.html"


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


def main():
    print("cobapeta_kalman.py mulai dijalankan...")
    print("Input :", INPUT_PATH)
    print("Output:", OUTPUT_HTML)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}. "
            "Jalankan integrasi_kalman_ke_fusion.py terlebih dahulu."
        )

    df = pd.read_csv(INPUT_PATH)
    print("Jumlah data input:", len(df))
    print("Kolom input:", list(df.columns))

    require_columns(
        df,
        ["AIS_Latitude", "AIS_Longitude", "Kalman_Lat", "Kalman_Lon"],
        "dataset/fusion_kalman_with_type.csv",
    )

    type_col = pick_type_column(df)
    if type_col is None:
        print("Peringatan: kolom tipe kapal tidak ditemukan. Popup akan memakai nilai Unknown.")
    else:
        print("Kolom tipe kapal yang dipakai:", type_col)

    numeric_cols = [
        "AIS_Latitude",
        "AIS_Longitude",
        "Kalman_Lat",
        "Kalman_Lon",
        "Kalman_Shift_m",
        "Sog",
        "Cog",
        "True_Head",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    kalman_valid = valid_lat_lon(df["Kalman_Lat"], df["Kalman_Lon"])
    ais_valid = valid_lat_lon(df["AIS_Latitude"], df["AIS_Longitude"])

    df["Plot_Lat"] = df["Kalman_Lat"].where(kalman_valid, df["AIS_Latitude"])
    df["Plot_Lon"] = df["Kalman_Lon"].where(kalman_valid, df["AIS_Longitude"])
    df["koordinat_sumber"] = "Kalman"
    df.loc[~kalman_valid & ais_valid, "koordinat_sumber"] = "Fallback AIS"

    plot_valid = valid_lat_lon(df["Plot_Lat"], df["Plot_Lon"])
    fallback_count = int((~kalman_valid & ais_valid).sum())
    invalid_count = int((~plot_valid).sum())

    print("Titik memakai koordinat Kalman:", int((kalman_valid & plot_valid).sum()))
    print("Titik fallback ke AIS mentah  :", fallback_count)
    print("Titik koordinat tidak valid    :", invalid_count)

    df_plot = df.loc[plot_valid].copy()
    if df_plot.empty:
        raise ValueError("Tidak ada koordinat valid untuk dibuat menjadi peta.")

    center = [df_plot["Plot_Lat"].median(), df_plot["Plot_Lon"].median()]
    fmap = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
    cluster = MarkerCluster(name="Titik AIS hasil Kalman").add_to(fmap)

    for _, row in df_plot.iterrows():
        ship_type = safe_value(row, type_col) if type_col else "Unknown"
        popup = f"""
        <b>MMSI:</b> {safe_value(row, "MMSI")}<br>
        <b>Jenis kapal:</b> {ship_type}<br>
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
        """

        folium.CircleMarker(
            location=[row["Plot_Lat"], row["Plot_Lon"]],
            radius=4,
            color="red" if row["koordinat_sumber"] == "Fallback AIS" else "blue",
            fill=True,
            fill_color="red" if row["koordinat_sumber"] == "Fallback AIS" else "blue",
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=430),
            tooltip=f"MMSI {safe_value(row, 'MMSI')} | {ship_type}",
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(OUTPUT_HTML)

    print("Peta berhasil dibuat:", OUTPUT_HTML)
    print("Jumlah titik diplot:", len(df_plot), "dari", len(df))
    print("cobapeta_kalman.py selesai dijalankan.")


if __name__ == "__main__":
    main()
