from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
OUTPUT_DIR = HERE / "output"

DASHBOARD_OUTPUT = OUTPUT_DIR / "dashboard_kapal_ikan_ais_sar.csv"
ALERT_OUTPUT = OUTPUT_DIR / "alert_kapal_ikan_ais_sar.csv"

DEFAULT_DISTANCE_THRESHOLD_KM = 5.0
DEFAULT_KALMAN_SHIFT_THRESHOLD_M = 500.0
DEFAULT_ABNORMAL_SOG_KNOTS = 20.0


DATA_CANDIDATES = {
    "fusion_kalman": [
        DATASET_DIR / "fusion_kalman_with_type.csv",
    ],
    "fusion": [
        DATASET_DIR / "fusion_minimal.csv",
        HERE / "fusion_minimal.csv",
        DATASET_DIR / "fusion_minimal_with_type.csv",
    ],
    "metadata": [
        DATASET_DIR / "metadata.csv",
    ],
    "kalman": [
        HERE / "kalman_ais_result.csv",
        DATASET_DIR / "kalman_ais_result.csv",
    ],
    "existing_alert": [
        DATASET_DIR / "alert_kapal_mencurigakan_kalman.csv",
        DATASET_DIR / "dark_vessel_alerts_v2.csv",
        DATASET_DIR / "dark_vessel_alerts.csv",
        DATASET_DIR / "hasil_deteksi_kapal_ikan_mencurigakan_kalman.csv",
        DATASET_DIR / "hasil_deteksi_kapal_ikan_mencurigakan1.csv",
        DATASET_DIR / "hasil_deteksi_kapal_mencurigakan.csv",
    ],
}


COLUMN_ALIASES = {
    "scene": ["scene", "waktu_scene"],
    "patch_cal": ["patch_cal", "patch", "patch_name"],
    "mmsi": ["MMSI", "mmsi"],
    "ship_type": [
        "Ship_type",
        "ship_type",
        "Ship_type_category",
        "category",
        "Category",
        "Elaborated_type",
        "vessel_type",
        "Vessel_Type",
        "label",
        "class",
        "Ship_Type",
    ],
    "sar_pred_type": ["predicted_class", "sar_class", "sar_ship_type", "classification"],
    "category": ["category", "Category"],
    "ais_lat": ["AIS_Latitude", "ais_latitude", "latitude_ais", "AIS_Lat", "ais_lat"],
    "ais_lon": ["AIS_Longitude", "ais_longitude", "longitude_ais", "AIS_Lon", "ais_lon"],
    "sar_lat": ["SAR_Latitude", "Sar_Latitude", "Center_latitude", "center_latitude", "sar_lat", "latitude_sar"],
    "sar_lon": ["SAR_Longitude", "Sar_Longitude", "Center_longitude", "center_longitude", "sar_lon", "longitude_sar"],
    "kalman_lat": ["Kalman_Lat", "kalman_lat"],
    "kalman_lon": ["Kalman_Lon", "kalman_lon"],
    "pred_lat": ["Pred_Lat", "pred_lat"],
    "pred_lon": ["Pred_Lon", "pred_lon"],
    "kalman_shift_m": ["Kalman_Shift_m", "kalman_shift_m"],
    "sog": ["Sog", "SOG", "sog"],
    "cog": ["Cog", "COG", "cog"],
    "true_head": ["True_Head", "true_head", "heading", "Heading"],
    "nav_status": ["Nav_Status", "nav_status", "status_navigasi"],
    "status": ["status", "Status"],
    "incidence": ["Incidence", "incidence"],
    "polarization": ["polarization", "Polarization"],
    "product_type": ["product_type", "Product_Type", "productType"],
    "detection_time": ["Detection_Time", "detection_time", "SAR_Start_Time", "Imaging_time", "alert_time"],
    "has_ais": ["has_ais", "Has_AIS"],
    "risk_level": ["kategori_risiko", "alert_level", "risk_level", "Risk_Level"],
    "risk_score": ["skor_mencurigakan", "risk_score", "score"],
    "alert_reason": ["alasan", "reason", "alert_reason", "keterangan"],
}


DISPLAY_COLUMNS = [
    "source_file",
    "scene",
    "patch_cal",
    "mmsi",
    "ship_type",
    "ship_type_normalized",
    "is_fishing_vessel",
    "sar_pred_type",
    "is_sar_predicted_fishing",
    "ais_lat",
    "ais_lon",
    "ais_final_lat",
    "ais_final_lon",
    "ais_final_source",
    "sar_lat",
    "sar_lon",
    "kalman_lat",
    "kalman_lon",
    "kalman_shift_m",
    "ais_without_kalman",
    "navigation_incomplete_for_kalman",
    "sog_raw",
    "sog_missing",
    "cog_missing",
    "sog_knots",
    "cog",
    "true_head",
    "nav_status",
    "incidence",
    "polarization",
    "product_type",
    "detection_time",
    "has_ais",
    "final_ais_sar_distance_km",
    "fusion_status",
]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def read_csv_if_exists(paths: Iterable[Path]) -> tuple[pd.DataFrame | None, Path | None]:
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path), path
            except Exception as exc:
                print(f"Gagal membaca {path}: {exc}")
    return None, None


def normalize_mmsi(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    out = raw.copy()
    if integer_like.any():
        out.loc[integer_like] = np.round(numeric.loc[integer_like]).astype("int64").astype(str)
    return out.replace({"<NA>": "", "nan": "", "NaN": "", "None": "", "none": ""})


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def normalize_ship_type(series: pd.Series) -> pd.Series:
    text = normalize_text(series).str.lower()
    text = text.str.replace("_", " ", regex=False)
    text = text.str.replace("-", " ", regex=False)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text


def is_fishing_ship_type(series: pd.Series) -> pd.Series:
    normalized = normalize_ship_type(series)
    return normalized.str.contains(r"fishing|fish|kapal ikan", na=False)


def is_valid_mmsi(series: pd.Series) -> pd.Series:
    text = normalize_mmsi(series)
    clean = text.str.replace(r"\.0$", "", regex=True)
    invalid_values = {"", "0", "nan", "none", "<na>"}
    return (~clean.str.lower().isin(invalid_values)) & clean.str.len().ge(8)


def standardize_dataframe(df: pd.DataFrame, source_file: Path | None = None) -> pd.DataFrame:
    work = df.copy()
    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in work.columns:
            continue
        found = find_column(work, aliases)
        if found is not None:
            work[standard_name] = work[found]

    for col in COLUMN_ALIASES:
        if col not in work.columns:
            work[col] = pd.NA

    work["source_file"] = str(source_file.name if source_file else "unknown")
    work["mmsi"] = normalize_mmsi(work["mmsi"])
    work["ship_type"] = normalize_text(work["ship_type"]).replace("", "Unknown")
    work["sar_pred_type"] = normalize_text(work["sar_pred_type"]).replace("", pd.NA)
    work["ship_type_normalized"] = normalize_ship_type(work["ship_type"])
    work["is_fishing_vessel"] = is_fishing_ship_type(work["ship_type"])
    work["is_sar_predicted_fishing"] = is_fishing_ship_type(work["sar_pred_type"])
    work["scene"] = normalize_text(work["scene"])
    work["patch_cal"] = normalize_text(work["patch_cal"]).str.replace("\\", "/", regex=False)

    numeric_cols = [
        "ais_lat",
        "ais_lon",
        "sar_lat",
        "sar_lon",
        "kalman_lat",
        "kalman_lon",
        "pred_lat",
        "pred_lon",
        "kalman_shift_m",
        "sog",
        "cog",
        "true_head",
        "incidence",
        "risk_score",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["has_valid_mmsi"] = is_valid_mmsi(work["mmsi"])
    if work["has_ais"].notna().any():
        has_ais_text = work["has_ais"].astype("string").fillna("").str.lower().str.strip()
        work["has_ais"] = has_ais_text.isin(["true", "1", "yes", "y"]) | work["has_valid_mmsi"]
    else:
        work["has_ais"] = work["has_valid_mmsi"]

    return normalize_navigation_values(work)


def normalize_navigation_values(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["sog_raw"] = pd.to_numeric(work["sog"], errors="coerce")
    sentinel_sog = work["sog_raw"].eq(1023)
    work["sog_missing"] = work["sog_raw"].isna() | sentinel_sog

    non_sentinel = work.loc[~sentinel_sog, "sog_raw"].dropna()
    needs_sog_scale = False
    if not non_sentinel.empty:
        needs_sog_scale = non_sentinel.quantile(0.95) > 60 or non_sentinel.max() > 80

    work["sog_knots"] = work["sog_raw"]
    if needs_sog_scale:
        work["sog_knots"] = work["sog_raw"] / 10.0
    work.loc[sentinel_sog, "sog_knots"] = np.nan
    work["sog_sentinel"] = sentinel_sog

    cog_raw = pd.to_numeric(work["cog"], errors="coerce")
    work["cog_raw"] = cog_raw
    sentinel_cog = cog_raw.eq(3600)
    work["cog_missing"] = cog_raw.isna() | sentinel_cog
    non_sentinel_cog = cog_raw.loc[~sentinel_cog].dropna()
    if not non_sentinel_cog.empty and non_sentinel_cog.max() > 360:
        cog_raw = cog_raw / 10.0
    cog_raw.loc[sentinel_cog] = np.nan
    work["cog"] = cog_raw
    work["cog_sentinel"] = sentinel_cog

    return work


def valid_lat_lon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    lat = pd.to_numeric(df.get(lat_col), errors="coerce")
    lon = pd.to_numeric(df.get(lon_col), errors="coerce")
    return lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = pd.to_numeric(lat1, errors="coerce")
    lon1 = pd.to_numeric(lon1, errors="coerce")
    lat2 = pd.to_numeric(lat2, errors="coerce")
    lon2 = pd.to_numeric(lon2, errors="coerce")

    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def load_sources() -> dict[str, tuple[pd.DataFrame | None, Path | None]]:
    return {name: read_csv_if_exists(paths) for name, paths in DATA_CANDIDATES.items()}


def choose_primary_data(sources: dict[str, tuple[pd.DataFrame | None, Path | None]]) -> tuple[pd.DataFrame, Path | None, list[str]]:
    notes = []
    for source_name in ["fusion_kalman", "fusion", "metadata"]:
        df, path = sources[source_name]
        if df is not None and not df.empty:
            notes.append(f"Sumber utama: {path}")
            return standardize_dataframe(df, path), path, notes
    notes.append("Tidak ada file fusion/metadata yang dapat dibaca. Dashboard memakai data kosong.")
    return standardize_dataframe(pd.DataFrame(), None), None, notes


def build_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=df.index)
    keys["_key_scene"] = normalize_text(df.get("scene", pd.Series(index=df.index, dtype="object")))
    keys["_key_patch"] = normalize_text(df.get("patch_cal", pd.Series(index=df.index, dtype="object"))).str.replace("\\", "/", regex=False)
    keys["_key_mmsi"] = normalize_mmsi(df.get("mmsi", pd.Series(index=df.index, dtype="object")))
    keys["_key_ais_lat"] = pd.to_numeric(df.get("ais_lat"), errors="coerce").round(6)
    keys["_key_ais_lon"] = pd.to_numeric(df.get("ais_lon"), errors="coerce").round(6)
    return keys


def enrich_from_metadata(main_df: pd.DataFrame, metadata_df: pd.DataFrame | None, metadata_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    if metadata_df is None or metadata_df.empty:
        notes.append("metadata.csv tidak tersedia; layer SAR memakai kolom SAR yang ada saja.")
        return main_df, notes

    meta = standardize_dataframe(metadata_df, metadata_path)
    main = main_df.copy()

    main_keys = build_merge_keys(main)
    meta_keys = build_merge_keys(meta)
    main = pd.concat([main, main_keys], axis=1)
    meta = pd.concat([meta, meta_keys], axis=1)

    enrich_cols = [
        "_key_scene",
        "_key_patch",
        "sar_lat",
        "sar_lon",
        "incidence",
        "polarization",
        "product_type",
        "detection_time",
        "category",
    ]
    meta_small = meta[enrich_cols].drop_duplicates(subset=["_key_scene", "_key_patch"])

    merged = main.merge(
        meta_small,
        on=["_key_scene", "_key_patch"],
        how="left",
        suffixes=("", "_meta"),
    )

    for col in ["sar_lat", "sar_lon", "incidence", "polarization", "product_type", "detection_time", "category"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            merged = merged.drop(columns=[meta_col])

    helper_cols = [col for col in merged.columns if col.startswith("_key_")]
    merged = merged.drop(columns=helper_cols, errors="ignore")
    notes.append("Koordinat dan metadata SAR diperkaya dari metadata.csv jika kolom belum tersedia.")
    return merged, notes


def enrich_from_kalman(main_df: pd.DataFrame, kalman_df: pd.DataFrame | None, kalman_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    if kalman_df is None or kalman_df.empty:
        notes.append("kalman_ais_result.csv tidak tersedia; fusion AIS-SAR berbasis Kalman tidak dapat dihitung.")
        return main_df, notes

    main_has_kalman = main_df["kalman_lat"].notna().any() and main_df["kalman_lon"].notna().any()
    if main_has_kalman:
        notes.append("Kolom Kalman sudah tersedia pada sumber utama.")
        return main_df, notes

    kalman = standardize_dataframe(kalman_df, kalman_path)
    main = main_df.copy()
    main_keys = build_merge_keys(main)
    kalman_keys = build_merge_keys(kalman)
    main = pd.concat([main, main_keys], axis=1)
    kalman = pd.concat([kalman, kalman_keys], axis=1)

    payload = [
        "kalman_lat",
        "kalman_lon",
        "pred_lat",
        "pred_lon",
        "kalman_shift_m",
        "_key_scene",
        "_key_patch",
        "_key_mmsi",
        "_key_ais_lat",
        "_key_ais_lon",
    ]
    kalman_payload = kalman[payload].drop_duplicates(
        subset=["_key_scene", "_key_patch", "_key_mmsi", "_key_ais_lat", "_key_ais_lon"]
    )

    merged = main.merge(
        kalman_payload,
        on=["_key_scene", "_key_patch", "_key_mmsi", "_key_ais_lat", "_key_ais_lon"],
        how="left",
        suffixes=("", "_kalman_src"),
    )

    for col in ["kalman_lat", "kalman_lon", "pred_lat", "pred_lon", "kalman_shift_m"]:
        src_col = f"{col}_kalman_src"
        if src_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[src_col])
            merged = merged.drop(columns=[src_col])

    helper_cols = [col for col in merged.columns if col.startswith("_key_")]
    merged = merged.drop(columns=helper_cols, errors="ignore")
    matched = int(merged["kalman_lat"].notna().sum())
    notes.append(f"Hasil Kalman digabung dari {kalman_path}; baris dengan Kalman: {matched}.")
    return merged, notes


def build_dashboard_data(threshold_km: float = DEFAULT_DISTANCE_THRESHOLD_KM) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sources = load_sources()
    main_df, _, notes = choose_primary_data(sources)

    metadata_df, metadata_path = sources["metadata"]
    main_df, meta_notes = enrich_from_metadata(main_df, metadata_df, metadata_path)
    notes.extend(meta_notes)

    kalman_df, kalman_path = sources["kalman"]
    main_df, kalman_notes = enrich_from_kalman(main_df, kalman_df, kalman_path)
    notes.extend(kalman_notes)

    if main_df.empty:
        return main_df, pd.DataFrame(), notes

    main_df["ais_valid"] = valid_lat_lon(main_df, "ais_lat", "ais_lon")
    main_df["sar_valid"] = valid_lat_lon(main_df, "sar_lat", "sar_lon")
    main_df["kalman_valid"] = valid_lat_lon(main_df, "kalman_lat", "kalman_lon")

    # Konsep penelitian: Kalman hanya memperhalus/mengoreksi AIS.
    # Posisi AIS final untuk fusion wajib berasal dari Kalman.
    # AIS mentah tidak dipakai sebagai fallback matching/fusion.
    main_df["ais_final_lat"] = main_df["kalman_lat"].where(main_df["kalman_valid"], np.nan)
    main_df["ais_final_lon"] = main_df["kalman_lon"].where(main_df["kalman_valid"], np.nan)
    main_df["ais_final_source"] = np.where(main_df["kalman_valid"], "Kalman", "Tidak ada Kalman")
    main_df["ais_final_valid"] = main_df["kalman_valid"]

    has_raw_ais = main_df["has_valid_mmsi"] & main_df["has_ais"] & main_df["ais_valid"]
    has_kalman_ais = main_df["has_valid_mmsi"] & main_df["has_ais"] & main_df["ais_final_valid"]
    main_df["unmatched_sar"] = main_df["sar_valid"] & (~main_df["has_valid_mmsi"] | ~main_df["has_ais"])
    main_df["unmatched_sar_fishing_candidate"] = main_df["unmatched_sar"] & main_df["is_sar_predicted_fishing"]
    main_df["ais_without_kalman"] = main_df["sar_valid"] & has_raw_ais & ~main_df["kalman_valid"]
    main_df["navigation_incomplete_for_kalman"] = (
        main_df["ais_without_kalman"]
        & (main_df["sog_missing"].fillna(False) | main_df["cog_missing"].fillna(False))
    )

    main_df["final_ais_sar_distance_km"] = np.nan
    distance_mask = has_kalman_ais & main_df["sar_valid"]
    if distance_mask.any():
        main_df.loc[distance_mask, "final_ais_sar_distance_km"] = haversine_km(
            main_df.loc[distance_mask, "ais_final_lat"],
            main_df.loc[distance_mask, "ais_final_lon"],
            main_df.loc[distance_mask, "sar_lat"],
            main_df.loc[distance_mask, "sar_lon"],
        )

    main_df["fusion_matched"] = (
        distance_mask
        & main_df["final_ais_sar_distance_km"].notna()
        & main_df["final_ais_sar_distance_km"].le(threshold_km)
    )
    main_df["position_anomaly"] = (
        distance_mask
        & main_df["final_ais_sar_distance_km"].notna()
        & main_df["final_ais_sar_distance_km"].gt(threshold_km)
    )
    main_df["fusion_status"] = "Data belum lengkap"
    main_df.loc[main_df["unmatched_sar"], "fusion_status"] = "SAR tanpa AIS - tipe kapal belum terklasifikasi"
    main_df.loc[main_df["unmatched_sar_fishing_candidate"], "fusion_status"] = "Kandidat Dark Vessel Kapal Ikan"
    main_df.loc[main_df["ais_without_kalman"], "fusion_status"] = "AIS tanpa estimasi Kalman"
    main_df.loc[main_df["navigation_incomplete_for_kalman"], "fusion_status"] = "Data AIS tidak lengkap untuk Kalman"
    main_df.loc[main_df["fusion_matched"], "fusion_status"] = "Matched AIS-SAR"
    main_df.loc[main_df["position_anomaly"], "fusion_status"] = "Anomali posisi"

    alerts = generate_alerts(main_df, threshold_km=threshold_km)
    return main_df, alerts, notes


def risk_priority(risk: str) -> int:
    return {"Tinggi": 3, "Sedang": 2, "Rendah": 1}.get(str(risk), 0)


def status_for_risk(risk: str) -> str:
    if risk == "Tinggi":
        return "Perlu Investigasi"
    if risk == "Sedang":
        return "Monitoring"
    return "Observasi"


def make_alert_record(row: pd.Series, alert_type: str, risk: str, reason: str) -> dict:
    lat = row.get("ais_final_lat")
    lon = row.get("ais_final_lon")
    coord_source = row.get("ais_final_source", "AIS Final")

    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("ais_lat")
        lon = row.get("ais_lon")
        coord_source = "AIS Mentah untuk verifikasi"

    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("sar_lat")
        lon = row.get("sar_lon")
        coord_source = "SAR"

    return {
        "prioritas": risk,
        "jenis_alert": alert_type,
        "status": status_for_risk(risk),
        "keterangan": reason,
        "scene": row.get("scene"),
        "waktu": row.get("detection_time"),
        "MMSI": row.get("mmsi"),
        "Ship_Type": row.get("ship_type"),
        "latitude": lat,
        "longitude": lon,
        "sumber_koordinat": coord_source,
        "AIS_Latitude": row.get("ais_lat"),
        "AIS_Longitude": row.get("ais_lon"),
        "AIS_Final_Latitude": row.get("ais_final_lat"),
        "AIS_Final_Longitude": row.get("ais_final_lon"),
        "AIS_Final_Source": row.get("ais_final_source"),
        "SAR_Latitude": row.get("sar_lat"),
        "SAR_Longitude": row.get("sar_lon"),
        "Kalman_Lat": row.get("kalman_lat"),
        "Kalman_Lon": row.get("kalman_lon"),
        "SOG": row.get("sog_knots"),
        "COG": row.get("cog"),
        "Kalman_Shift_m": row.get("kalman_shift_m"),
        "Final_AIS_SAR_Distance_km": row.get("final_ais_sar_distance_km"),
        "Fusion_Status": row.get("fusion_status"),
        "patch_cal": row.get("patch_cal"),
    }


def generate_alerts(
    df: pd.DataFrame,
    threshold_km: float = DEFAULT_DISTANCE_THRESHOLD_KM,
    abnormal_sog_threshold: float = DEFAULT_ABNORMAL_SOG_KNOTS,
    kalman_shift_threshold_m: float = DEFAULT_KALMAN_SHIFT_THRESHOLD_M,
) -> pd.DataFrame:
    records = []
    if df.empty:
        return pd.DataFrame(records)

    for _, row in df.iterrows():
        if bool(row.get("unmatched_sar_fishing_candidate", False)):
            records.append(
                make_alert_record(
                    row,
                    "Kandidat Dark Vessel Kapal Ikan",
                    "Tinggi",
                    "Deteksi SAR terklasifikasi sebagai fishing vessel tetapi MMSI/pasangan AIS tidak valid atau tidak tersedia.",
                )
            )

        if bool(row.get("ais_without_kalman", False)):
            missing_parts = []
            if bool(row.get("sog_missing", False)):
                missing_parts.append("SOG kosong/invalid")
            if bool(row.get("cog_missing", False)):
                missing_parts.append("COG kosong/invalid")
            if not missing_parts:
                missing_parts.append("estimasi Kalman tidak tersedia")

            records.append(
                make_alert_record(
                    row,
                    "Kapal Ikan Perlu Investigasi",
                    "Sedang",
                    "Data AIS memiliki pasangan SAR tetapi tidak dapat dipakai untuk estimasi Kalman: "
                    + ", ".join(missing_parts)
                    + ". Perlu verifikasi data AIS sebelum proses fusion.",
                )
            )

        distance = row.get("final_ais_sar_distance_km")
        if pd.notna(distance) and distance > threshold_km:
            risk = "Tinggi" if distance > threshold_km * 2 else "Sedang"
            records.append(
                make_alert_record(
                    row,
                    "Anomali Posisi Kapal Ikan AIS-SAR",
                    risk,
                    f"Jarak posisi AIS final ke SAR {distance:.2f} km melebihi threshold {threshold_km:.2f} km.",
                )
            )

        sog = row.get("sog_knots")
        if bool(row.get("sog_sentinel", False)) or (pd.notna(sog) and sog > abnormal_sog_threshold):
            records.append(
                make_alert_record(
                    row,
                    "Indikasi Aktivitas Mencurigakan",
                    "Sedang",
                    f"Nilai SOG kapal ikan tidak wajar atau melebihi threshold {abnormal_sog_threshold:.2f} knot.",
                )
            )

        kalman_shift = row.get("kalman_shift_m")
        if pd.notna(kalman_shift) and kalman_shift > kalman_shift_threshold_m:
            records.append(
                make_alert_record(
                    row,
                    "Kapal Ikan Perlu Investigasi",
                    "Sedang",
                    f"Kalman_Shift_m kapal ikan {kalman_shift:.2f} meter melebihi threshold {kalman_shift_threshold_m:.2f} meter.",
                )
            )

    alerts = pd.DataFrame(records)
    if alerts.empty:
        return alerts

    alerts["_risk_order"] = alerts["prioritas"].map(risk_priority)
    alerts = alerts.sort_values(["_risk_order", "jenis_alert"], ascending=[False, True])
    alerts = alerts.drop(columns=["_risk_order"])
    return alerts.reset_index(drop=True)


def save_outputs(dashboard_df: pd.DataFrame, alerts_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_cols = [col for col in DISPLAY_COLUMNS if col in dashboard_df.columns]
    if dashboard_cols:
        dashboard_df[dashboard_cols].to_csv(DASHBOARD_OUTPUT, index=False)
    else:
        dashboard_df.to_csv(DASHBOARD_OUTPUT, index=False)
    alerts_df.to_csv(ALERT_OUTPUT, index=False)


def safe_html(value) -> str:
    if pd.isna(value):
        return "-"
    return html.escape(str(value))


def format_float(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return safe_html(value)


def marker_popup(row: pd.Series, marker_type: str) -> str:
    return f"""
    <b>{safe_html(marker_type)}</b><br>
    <b>MMSI:</b> {safe_html(row.get("mmsi"))}<br>
    <b>Ship Type:</b> {safe_html(row.get("ship_type"))}<br>
    <b>AIS mentah:</b> {format_float(row.get("ais_lat"))}, {format_float(row.get("ais_lon"))}<br>
    <b>AIS final:</b> {format_float(row.get("ais_final_lat"))}, {format_float(row.get("ais_final_lon"))}<br>
    <b>Sumber AIS final:</b> {safe_html(row.get("ais_final_source"))}<br>
    <b>SAR:</b> {format_float(row.get("sar_lat"))}, {format_float(row.get("sar_lon"))}<br>
    <b>SOG:</b> {format_float(row.get("sog_knots"), 2)} kn<br>
    <b>COG:</b> {format_float(row.get("cog"), 2)}<br>
    <b>Nav Status:</b> {safe_html(row.get("nav_status"))}<br>
    <b>Status fusion:</b> {safe_html(row.get("fusion_status"))}<br>
    <b>Jarak AIS final-SAR:</b> {format_float(row.get("final_ais_sar_distance_km"), 2)} km<br>
    <b>Kalman Shift:</b> {format_float(row.get("kalman_shift_m"), 2)} m<br>
    <b>Incidence:</b> {format_float(row.get("incidence"), 2)}<br>
    <b>Polarization:</b> {safe_html(row.get("polarization"))}<br>
    <b>Product Type:</b> {safe_html(row.get("product_type"))}<br>
    <hr>
    <b>Scene:</b><br>{safe_html(row.get("scene"))}<br>
    <b>Patch:</b> {safe_html(row.get("patch_cal"))}
    """


def alert_popup(row: pd.Series) -> str:
    return f"""
    <b>Alert: {safe_html(row.get("jenis_alert"))}</b><br>
    <b>Prioritas:</b> {safe_html(row.get("prioritas"))}<br>
    <b>Status:</b> {safe_html(row.get("status"))}<br>
    <b>MMSI:</b> {safe_html(row.get("MMSI"))}<br>
    <b>Ship Type:</b> {safe_html(row.get("Ship_Type"))}<br>
    <b>Koordinat:</b> {format_float(row.get("latitude"))}, {format_float(row.get("longitude"))}<br>
    <b>Keterangan:</b> {safe_html(row.get("keterangan"))}<br>
    <hr>
    <b>Scene:</b><br>{safe_html(row.get("scene"))}
    """


def add_circle_marker(group, lat, lon, color, popup, tooltip, radius=4):
    if pd.isna(lat) or pd.isna(lon):
        return
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.86,
        weight=1,
        popup=folium.Popup(popup, max_width=440),
        tooltip=tooltip,
    ).add_to(group)


def build_map(df: pd.DataFrame, alerts: pd.DataFrame, selected_layers: list[str], max_rows: int):
    map_df = df.head(max_rows).copy()
    center_lat = pd.concat(
        [
            pd.to_numeric(map_df.get("kalman_lat"), errors="coerce"),
            pd.to_numeric(map_df.get("ais_lat"), errors="coerce"),
            pd.to_numeric(map_df.get("sar_lat"), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    center_lon = pd.concat(
        [
            pd.to_numeric(map_df.get("kalman_lon"), errors="coerce"),
            pd.to_numeric(map_df.get("ais_lon"), errors="coerce"),
            pd.to_numeric(map_df.get("sar_lon"), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()

    if center_lat.empty or center_lon.empty:
        center = [-2.5, 118.0]
    else:
        center = [float(center_lat.median()), float(center_lon.median())]

    fmap = folium.Map(location=center, zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB dark_matter", name="Dark Map", control=True).add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(fmap)

    layer_names = ["AIS Mentah", "AIS Kalman", "SAR", "Fusion", "Alert"]
    layer_flags = {name: name in selected_layers for name in layer_names}

    ais_raw_group = folium.FeatureGroup(name="AIS Kapal Ikan Mentah - Biru Muda", show=layer_flags["AIS Mentah"])
    ais_kalman_group = folium.FeatureGroup(name="AIS Kapal Ikan Kalman - Cyan/Ungu", show=layer_flags["AIS Kalman"])
    sar_group = folium.FeatureGroup(name="SAR Kapal Ikan - Oranye", show=layer_flags["SAR"])
    fusion_group = folium.FeatureGroup(name="Fusion Kapal Ikan - Hijau", show=layer_flags["Fusion"])
    alert_group = folium.FeatureGroup(name="Alert Kapal Ikan - Merah", show=layer_flags["Alert"])

    ais_raw_cluster = MarkerCluster(name="AIS Kapal Ikan Mentah").add_to(ais_raw_group)
    ais_kalman_cluster = MarkerCluster(name="AIS Kapal Ikan Kalman").add_to(ais_kalman_group)
    sar_cluster = MarkerCluster(name="SAR Kapal Ikan").add_to(sar_group)

    for _, row in map_df.iterrows():
        if bool(row.get("ais_valid", False)):
            add_circle_marker(
                ais_raw_cluster,
                row.get("ais_lat"),
                row.get("ais_lon"),
                "#57c7ff",
                marker_popup(row, "AIS kapal ikan mentah"),
                f"AIS mentah | {safe_html(row.get('mmsi'))}",
            )

        if bool(row.get("kalman_valid", False)):
            add_circle_marker(
                ais_kalman_cluster,
                row.get("kalman_lat"),
                row.get("kalman_lon"),
                "#7c4dff",
                marker_popup(row, "AIS kapal ikan hasil Kalman"),
                f"Kalman | {safe_html(row.get('mmsi'))}",
            )

        if bool(row.get("sar_valid", False)):
            add_circle_marker(
                sar_cluster,
                row.get("sar_lat"),
                row.get("sar_lon"),
                "#f39c12",
                marker_popup(row, "SAR terkait kapal ikan"),
                "SAR kapal ikan",
            )

        if bool(row.get("fusion_matched", False)):
            mid_lat = np.nanmean([row.get("kalman_lat"), row.get("sar_lat")])
            mid_lon = np.nanmean([row.get("kalman_lon"), row.get("sar_lon")])
            add_circle_marker(
                fusion_group,
                mid_lat,
                mid_lon,
                "#2ecc71",
                marker_popup(row, "Fusion kapal ikan AIS-SAR"),
                f"Fusion kapal ikan | {safe_html(row.get('mmsi'))}",
                radius=5,
            )
            if all(pd.notna(row.get(col)) for col in ["kalman_lat", "kalman_lon", "sar_lat", "sar_lon"]):
                folium.PolyLine(
                    locations=[
                        [row.get("kalman_lat"), row.get("kalman_lon")],
                        [row.get("sar_lat"), row.get("sar_lon")],
                    ],
                    color="#2ecc71",
                    weight=1,
                    opacity=0.55,
                ).add_to(fusion_group)

    for _, row in alerts.head(max_rows).iterrows():
        color = "#ff3b30" if row.get("prioritas") == "Tinggi" else "#ff7a45"
        add_circle_marker(
            alert_group,
            row.get("latitude"),
            row.get("longitude"),
            color,
            alert_popup(row),
            f"{safe_html(row.get('prioritas'))} | {safe_html(row.get('jenis_alert'))}",
            radius=7 if row.get("prioritas") == "Tinggi" else 5,
        )

    for group in [ais_raw_group, ais_kalman_group, sar_group, fusion_group, alert_group]:
        group.add_to(fmap)

    legend_html = """
    <div style="
        position: fixed; bottom: 35px; left: 35px; z-index: 9999;
        background: rgba(7, 18, 34, 0.88); color: #e6edf7;
        padding: 12px 14px; border-radius: 8px; border: 1px solid #24364f;
        font-size: 13px; box-shadow: 0 8px 24px rgba(0,0,0,.35);">
        <b>Legenda Marker Kapal Ikan</b><br>
        <span style="color:#57c7ff;">&bull;</span> AIS mentah kapal ikan<br>
        <span style="color:#7c4dff;">&bull;</span> AIS Kalman kapal ikan<br>
        <span style="color:#f39c12;">&bull;</span> SAR terkait kapal ikan<br>
        <span style="color:#2ecc71;">&bull;</span> Fusion kapal ikan<br>
        <span style="color:#ff3b30;">&bull;</span> Alert kapal ikan
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def apply_data_filters(
    df: pd.DataFrame,
    ship_types: list[str],
    sog_range: tuple[float, float],
    only_fishing: bool = True,
) -> pd.DataFrame:
    filtered = df.copy()
    if only_fishing:
        filtered = filtered[filtered["is_fishing_vessel"]]
    elif ship_types:
        filtered = filtered[filtered["ship_type"].fillna("Unknown").isin(ship_types)]

    low, high = sog_range
    sog = pd.to_numeric(filtered["sog_knots"], errors="coerce")
    filtered = filtered[sog.isna() | sog.between(low, high)]
    return filtered


def apply_alert_filters(alerts: pd.DataFrame, risk_levels: list[str], statuses: list[str]) -> pd.DataFrame:
    filtered = alerts.copy()
    if filtered.empty:
        return filtered
    if risk_levels:
        filtered = filtered[filtered["prioritas"].isin(risk_levels)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    return filtered


def summary_values(df: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, int]:
    valid_ais = df[df["has_valid_mmsi"]]
    total_ais = int(valid_ais["mmsi"].nunique()) if not valid_ais.empty else 0
    total_fusion = int(df.get("fusion_matched", pd.Series(dtype=bool)).sum())
    sar_fishing = int(df.get("sar_valid", pd.Series(dtype=bool)).sum())
    active_alert = int(len(alerts))
    dark_candidates = int((alerts.get("jenis_alert", pd.Series(dtype=str)) == "Kandidat Dark Vessel Kapal Ikan").sum())
    return {
        "Total Kapal Ikan AIS": total_ais,
        "Kapal Ikan Fusion / Matched": total_fusion,
        "SAR Kapal Ikan": sar_fishing,
        "Alert Kapal Ikan Aktif": active_alert,
        "Kandidat Dark Vessel Kapal Ikan": dark_candidates,
    }


def run_streamlit_app() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    try:
        from streamlit_folium import st_folium
    except Exception:
        st_folium = None

    st.set_page_config(
        page_title="Sistem Pemantauan Kapal Ikan AIS-SAR",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #07111f;
            color: #e6edf7;
        }
        [data-testid="stSidebar"] {
            background: #0b1728;
            border-right: 1px solid #20324a;
        }
        .hero {
            padding: 18px 22px;
            border: 1px solid #1f3656;
            border-radius: 8px;
            background: linear-gradient(135deg, #0d1d33 0%, #10243f 55%, #122b48 100%);
            margin-bottom: 14px;
        }
        .hero h1 {
            margin: 0;
            font-size: 30px;
            letter-spacing: 0;
            color: #f8fbff;
        }
        .hero p {
            margin: 6px 0 0 0;
            color: #a9bad2;
            font-size: 15px;
        }
        .metric-card {
            border: 1px solid #223956;
            border-radius: 8px;
            background: #0d1b2d;
            padding: 14px 16px;
            min-height: 92px;
        }
        .metric-card .label {
            color: #91a7c2;
            font-size: 13px;
        }
        .metric-card .value {
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin-top: 6px;
        }
        .alert-card {
            border: 1px solid #283f5d;
            border-radius: 8px;
            background: #0d1b2d;
            padding: 12px;
            margin-bottom: 10px;
        }
        .risk-high { color: #ff6b6b; font-weight: 700; }
        .risk-medium { color: #ffd166; font-weight: 700; }
        .risk-low { color: #82d173; font-weight: 700; }
        .small-muted { color: #9fb1c7; font-size: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Sistem Pemantauan Kapal Ikan AIS-SAR</h1>
            <p>Output Integrasi AIS-SAR, Kalman Filter, dan Alert Indikasi Kapal Ikan Mencurigakan</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Filter Dashboard")
    threshold_km = st.sidebar.slider(
        "Threshold jarak AIS-SAR (km)",
        min_value=0.5,
        max_value=50.0,
        value=DEFAULT_DISTANCE_THRESHOLD_KM,
        step=0.5,
    )
    kalman_shift_threshold = st.sidebar.slider(
        "Threshold Kalman Shift (meter)",
        min_value=100.0,
        max_value=10000.0,
        value=DEFAULT_KALMAN_SHIFT_THRESHOLD_M,
        step=100.0,
    )
    abnormal_sog_threshold = st.sidebar.slider(
        "Threshold SOG tidak normal (knots)",
        min_value=5.0,
        max_value=60.0,
        value=DEFAULT_ABNORMAL_SOG_KNOTS,
        step=1.0,
    )
    dashboard_df, alerts_df, notes = build_dashboard_data(threshold_km=threshold_km)

    if dashboard_df.empty:
        st.error("Tidak ada data yang dapat dibaca. Pastikan file CSV tersedia di folder dataset.")
        return

    ship_type_options = sorted([str(x) for x in dashboard_df["ship_type"].fillna("Unknown").unique()])
    fishing_type_options = sorted([
        str(x)
        for x in dashboard_df.loc[dashboard_df["is_fishing_vessel"], "ship_type"].fillna("Unknown").unique()
    ])
    only_fishing = st.sidebar.checkbox("Tampilkan hanya kapal ikan", value=True)
    selected_ship_types = st.sidebar.multiselect(
        "Ship Type",
        ship_type_options,
        default=fishing_type_options if fishing_type_options else [],
    )

    sog_series = pd.to_numeric(dashboard_df["sog_knots"], errors="coerce").dropna()
    if sog_series.empty:
        sog_min, sog_max = 0.0, 60.0
    else:
        sog_min = float(max(0.0, math.floor(sog_series.min())))
        sog_max = float(max(sog_min + 1.0, math.ceil(sog_series.quantile(0.99))))
    sog_range = st.sidebar.slider(
        "Rentang SOG (knots)",
        min_value=sog_min,
        max_value=sog_max,
        value=(sog_min, sog_max),
        step=0.5,
    )

    risk_options = ["Tinggi", "Sedang", "Rendah"]
    selected_risks = st.sidebar.multiselect("Risk Level", risk_options, default=risk_options)
    status_options = ["Perlu Investigasi", "Monitoring", "Observasi"]
    selected_statuses = st.sidebar.multiselect("Status", status_options, default=status_options)
    selected_layers = st.sidebar.multiselect(
        "Layer peta",
        ["AIS Mentah", "AIS Kalman", "SAR", "Fusion", "Alert"],
        default=["AIS Mentah", "AIS Kalman", "SAR", "Fusion", "Alert"],
    )
    max_marker_rows = st.sidebar.slider(
        "Batas marker per layer",
        min_value=250,
        max_value=max(250, min(6000, len(dashboard_df))),
        value=min(2500, len(dashboard_df)),
        step=250,
    )

    filtered_df = apply_data_filters(dashboard_df, selected_ship_types, sog_range, only_fishing=only_fishing)
    alerts_df = generate_alerts(
        filtered_df,
        threshold_km=threshold_km,
        abnormal_sog_threshold=abnormal_sog_threshold,
        kalman_shift_threshold_m=kalman_shift_threshold,
    )
    filtered_alerts = apply_alert_filters(alerts_df, selected_risks, selected_statuses)

    save_outputs(filtered_df, filtered_alerts)

    for note in notes:
        st.caption(note)

    st.info(
        "Dashboard ini hanya menampilkan kapal dengan kategori Fishing/Fishing Vessel. "
        "Alert yang ditampilkan merupakan indikasi awal untuk investigasi, bukan vonis IUU Fishing."
    )

    if int(filtered_df.get("unmatched_sar_fishing_candidate", pd.Series(dtype=bool)).sum()) == 0:
        st.warning(
            "Data saat ini belum menunjukkan SAR tanpa AIS yang benar-benar terklasifikasi sebagai kapal ikan. "
            "Kandidat dark vessel kapal ikan hanya dibuat jika label/prediksi SAR mendukung."
        )

    values = summary_values(filtered_df, filtered_alerts)
    metric_cols = st.columns(5)
    for col, (label, value) in zip(metric_cols, values.items()):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    map_col, alert_col = st.columns([2.25, 1.0], gap="large")
    with map_col:
        st.subheader("Peta Interaktif Kapal Ikan AIS-SAR-Kalman")
        fmap = build_map(filtered_df, filtered_alerts, selected_layers, max_rows=max_marker_rows)
        if st_folium is not None:
            st_folium(fmap, width=None, height=720, returned_objects=[])
        else:
            st.warning("streamlit-folium belum terpasang. Peta ditampilkan sebagai HTML statis.")
            components.html(fmap._repr_html_(), height=720, scrolling=False)

    with alert_col:
        st.subheader("Panel Alert Kapal Ikan")
        st.caption("Alert kapal ikan adalah indikasi awal untuk investigasi, bukan vonis IUU Fishing.")

        if filtered_alerts.empty:
            st.info("Tidak ada alert aktif berdasarkan filter saat ini.")
        else:
            for _, alert in filtered_alerts.head(12).iterrows():
                risk = alert.get("prioritas")
                risk_class = "risk-high" if risk == "Tinggi" else "risk-medium" if risk == "Sedang" else "risk-low"
                st.markdown(
                    f"""
                    <div class="alert-card">
                        <div class="{risk_class}">{safe_html(risk)} - {safe_html(alert.get("jenis_alert"))}</div>
                        <div><b>Status:</b> {safe_html(alert.get("status"))}</div>
                        <div><b>MMSI:</b> {safe_html(alert.get("MMSI"))}</div>
                        <div><b>Koordinat:</b> {format_float(alert.get("latitude"))}, {format_float(alert.get("longitude"))}</div>
                        <div class="small-muted">{safe_html(alert.get("keterangan"))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Distribusi Alert")
        if not filtered_alerts.empty:
            st.bar_chart(filtered_alerts["prioritas"].value_counts())

    with st.expander("Data Alert dan Output CSV", expanded=False):
        st.write(f"Output alert disimpan ke: `{ALERT_OUTPUT}`")
        st.write(f"Data dashboard dirapikan disimpan ke: `{DASHBOARD_OUTPUT}`")
        st.dataframe(filtered_alerts, use_container_width=True, height=320)

    with st.expander("Data Dashboard Dirapikan", expanded=False):
        cols = [col for col in DISPLAY_COLUMNS if col in filtered_df.columns]
        st.dataframe(filtered_df[cols], use_container_width=True, height=360)


def cli_build_outputs() -> None:
    dashboard_df, alerts_df, notes = build_dashboard_data(threshold_km=DEFAULT_DISTANCE_THRESHOLD_KM)
    fishing_df = dashboard_df[dashboard_df["is_fishing_vessel"]].copy() if not dashboard_df.empty else dashboard_df
    alerts_df = generate_alerts(
        fishing_df,
        threshold_km=DEFAULT_DISTANCE_THRESHOLD_KM,
        abnormal_sog_threshold=DEFAULT_ABNORMAL_SOG_KNOTS,
        kalman_shift_threshold_m=DEFAULT_KALMAN_SHIFT_THRESHOLD_M,
    )
    save_outputs(fishing_df, alerts_df)
    print("Dashboard fishing data rows:", len(fishing_df))
    print("Fishing alert rows:", len(alerts_df))
    print("Dashboard output:", DASHBOARD_OUTPUT)
    print("Alert output:", ALERT_OUTPUT)
    for note in notes:
        print("-", note)
    if not fishing_df.empty and int(fishing_df.get("unmatched_sar_fishing_candidate", pd.Series(dtype=bool)).sum()) == 0:
        print(
            "Catatan: data belum menunjukkan SAR tanpa AIS yang terklasifikasi sebagai kapal ikan; "
            "kandidat dark vessel kapal ikan belum tersedia."
        )


def main() -> None:
    if "--build-output" in sys.argv:
        cli_build_outputs()
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()
