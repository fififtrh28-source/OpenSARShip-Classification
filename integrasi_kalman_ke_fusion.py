import numpy as np
import pandas as pd
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"

FUSION_PATH = DATASET_DIR / "fusion_minimal_with_type.csv"
KALMAN_PATH = HERE / "kalman_ais_result.csv"
OUTPUT_PATH = DATASET_DIR / "fusion_kalman_with_type.csv"

KALMAN_PAYLOAD_COLS = [
    "Pred_Lat",
    "Pred_Lon",
    "Kalman_Lat",
    "Kalman_Lon",
    "Kalman_Shift_m",
]


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan pada {label}: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )


def normalize_id(series):
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))

    out = text.fillna("")
    if integer_like.any():
        out.loc[integer_like] = (
            np.round(numeric.loc[integer_like])
            .astype("int64")
            .astype(str)
        )

    return out.replace({"<NA>": "", "nan": "", "None": "", "NaN": ""})


def normalize_text(series):
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace("\\", "/", regex=False)
    )


def add_match_columns(df, is_kalman=False):
    work = df.copy()
    work["_key_mmsi"] = normalize_id(work["MMSI"])
    work["_key_lat"] = pd.to_numeric(work["AIS_Latitude"], errors="coerce").round(6)
    work["_key_lon"] = pd.to_numeric(work["AIS_Longitude"], errors="coerce").round(6)

    if "scene" in work.columns:
        work["_key_scene"] = normalize_text(work["scene"])

    if "patch_cal" in work.columns:
        work["_key_patch"] = normalize_text(work["patch_cal"])

    if is_kalman and "_orig_order" in work.columns:
        work["_key_order"] = pd.to_numeric(work["_orig_order"], errors="coerce")
    elif not is_kalman:
        work["_key_order"] = work["_fusion_row_id"].astype(float)

    return work


def valid_key_mask(df, keys):
    mask = pd.Series(True, index=df.index)
    for key in keys:
        if key not in df.columns:
            return pd.Series(False, index=df.index)

        if pd.api.types.is_numeric_dtype(df[key]):
            mask &= df[key].notna()
        else:
            values = df[key].astype("string").fillna("").str.strip()
            mask &= values != ""

    return mask


def attach_with_strategy(fusion, kalman, keys, method, used_kalman_ids):
    pending = fusion[fusion["kalman_merge_method"].eq("")]
    available = kalman[~kalman["_kalman_row_id"].isin(used_kalman_ids)]

    if pending.empty or available.empty:
        return 0, used_kalman_ids

    left_mask = valid_key_mask(pending, keys)
    right_mask = valid_key_mask(available, keys)

    left = pending.loc[left_mask, ["_fusion_row_id"] + keys].copy()
    right = available.loc[
        right_mask,
        ["_kalman_row_id"] + keys + KALMAN_PAYLOAD_COLS,
    ].copy()

    if left.empty or right.empty:
        return 0, used_kalman_ids

    left["_key_occurrence"] = left.groupby(keys, dropna=False).cumcount()
    right["_key_occurrence"] = right.groupby(keys, dropna=False).cumcount()

    merged = left.merge(
        right,
        on=keys + ["_key_occurrence"],
        how="left",
        validate="one_to_one",
    )

    matched = merged[merged["_kalman_row_id"].notna()].copy()
    if matched.empty:
        return 0, used_kalman_ids

    matched["_kalman_row_id"] = matched["_kalman_row_id"].astype(int)
    matched = matched.drop_duplicates(subset=["_fusion_row_id"], keep="first")
    matched_index = matched.set_index("_fusion_row_id")

    update_mask = fusion["_fusion_row_id"].isin(matched_index.index)
    row_ids = fusion.loc[update_mask, "_fusion_row_id"]

    for col in KALMAN_PAYLOAD_COLS:
        fusion.loc[update_mask, col] = row_ids.map(matched_index[col]).to_numpy()

    fusion.loc[update_mask, "kalman_merge_method"] = method
    fusion.loc[update_mask, "matched_kalman_row_id"] = (
        row_ids.map(matched_index["_kalman_row_id"]).to_numpy()
    )

    new_used = set(matched["_kalman_row_id"].tolist())
    used_kalman_ids.update(new_used)

    return len(matched), used_kalman_ids


def main():
    print("integrasi_kalman_ke_fusion.py mulai dijalankan...")
    print("Input fusion :", FUSION_PATH)
    print("Input Kalman :", KALMAN_PATH)
    print("Output       :", OUTPUT_PATH)

    if not FUSION_PATH.exists():
        raise FileNotFoundError(f"File fusion tidak ditemukan: {FUSION_PATH}")
    if not KALMAN_PATH.exists():
        raise FileNotFoundError(f"File Kalman tidak ditemukan: {KALMAN_PATH}")

    fusion_df = pd.read_csv(FUSION_PATH)
    kalman_df = pd.read_csv(KALMAN_PATH)

    print("Jumlah data fusion :", len(fusion_df))
    print("Jumlah data Kalman :", len(kalman_df))

    require_columns(
        fusion_df,
        ["MMSI", "AIS_Latitude", "AIS_Longitude"],
        "dataset/fusion_minimal_with_type.csv",
    )
    require_columns(
        kalman_df,
        ["MMSI", "AIS_Latitude", "AIS_Longitude", "Kalman_Lat", "Kalman_Lon"],
        "kalman_ais_result.csv",
    )

    missing_payload = [col for col in KALMAN_PAYLOAD_COLS if col not in kalman_df.columns]
    if missing_payload:
        raise KeyError(
            f"Kolom hasil Kalman tidak lengkap: {missing_payload}. "
            "Pastikan cobakalman.py sudah menghasilkan Pred_Lat, Pred_Lon, "
            "Kalman_Lat, Kalman_Lon, dan Kalman_Shift_m."
        )

    for col in ["scene", "patch_cal", "Incidence"]:
        if col not in fusion_df.columns:
            print(f"Peringatan: kolom SAR '{col}' tidak ada di fusion, jadi tidak ikut output.")

    type_cols = [col for col in ["Ship_type", "Ship_Type", "category"] if col in fusion_df.columns]
    if not type_cols:
        print("Peringatan: kolom tipe kapal Ship_type/Ship_Type/category tidak ditemukan di fusion.")
    else:
        print("Kolom tipe kapal yang ikut output:", type_cols)

    fusion_work = fusion_df.copy()
    fusion_work["_fusion_row_id"] = np.arange(len(fusion_work))

    kalman_work = kalman_df.copy()
    kalman_work["_kalman_row_id"] = np.arange(len(kalman_work))

    fusion_work = add_match_columns(fusion_work, is_kalman=False)
    kalman_work = add_match_columns(kalman_work, is_kalman=True)

    for col in KALMAN_PAYLOAD_COLS:
        fusion_work[col] = pd.NA

    fusion_work["kalman_merge_method"] = ""
    fusion_work["matched_kalman_row_id"] = pd.NA

    strategies = []
    if {"_key_scene", "_key_patch"}.issubset(fusion_work.columns) and {
        "_key_scene",
        "_key_patch",
    }.issubset(kalman_work.columns):
        strategies.append(
            (
                ["_key_scene", "_key_patch", "_key_mmsi", "_key_lat", "_key_lon"],
                "scene_patch_mmsi_koordinat",
            )
        )

    if "_key_order" in fusion_work.columns and "_key_order" in kalman_work.columns:
        strategies.append((["_key_order", "_key_mmsi"], "orig_order_mmsi"))

    strategies.append((["_key_mmsi", "_key_lat", "_key_lon"], "mmsi_koordinat"))
    strategies.append((["_key_mmsi"], "urutan_kemunculan_per_mmsi"))

    used_kalman_ids = set()
    for keys, method in strategies:
        matched_count, used_kalman_ids = attach_with_strategy(
            fusion_work,
            kalman_work,
            keys,
            method,
            used_kalman_ids,
        )
        print(f"Match dengan {method}: {matched_count} baris")

    matched_total = int(fusion_work["kalman_merge_method"].ne("").sum())
    unmatched_total = len(fusion_work) - matched_total

    print("Total berhasil tergabung :", matched_total)
    print("Total tidak tergabung    :", unmatched_total)

    if unmatched_total > 0:
        sample_cols = [
            col
            for col in ["scene", "patch_cal", "MMSI", "AIS_Latitude", "AIS_Longitude"]
            if col in fusion_work.columns
        ]
        print("\nContoh data yang tidak berhasil tergabung:")
        print(
            fusion_work.loc[
                fusion_work["kalman_merge_method"].eq(""),
                sample_cols,
            ].head(10)
        )

    helper_cols = [
        col
        for col in fusion_work.columns
        if col.startswith("_key_") or col in ["_fusion_row_id", "matched_kalman_row_id"]
    ]
    output_df = fusion_work.drop(columns=helper_cols, errors="ignore")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("File berhasil disimpan:", OUTPUT_PATH)
    print("Kolom output:", list(output_df.columns))
    print("integrasi_kalman_ke_fusion.py selesai dijalankan.")


if __name__ == "__main__":
    main()
