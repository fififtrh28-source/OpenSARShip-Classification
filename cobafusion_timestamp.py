import pandas as pd
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta_path = HERE / "dataset" / "metadata.csv"
out_path = HERE / "dataset" / "fusion_minimal.csv"

if not meta_path.exists():
    raise FileNotFoundError(f"metadata.csv tidak ditemukan: {meta_path}")

df = pd.read_csv(meta_path)


def extract_scene_start_time(scene_value):
    text = str(scene_value)
    matches = re.findall(r"\d{8}T\d{6}", text)

    if len(matches) >= 1:
        return pd.to_datetime(matches[0], format="%Y%m%dT%H%M%S", errors="coerce")

    return pd.NaT


def extract_scene_end_time(scene_value):
    text = str(scene_value)
    matches = re.findall(r"\d{8}T\d{6}", text)

    if len(matches) >= 2:
        return pd.to_datetime(matches[1], format="%Y%m%dT%H%M%S", errors="coerce")

    return pd.NaT


# =============================
# RAPIKAN NILAI AIS MENTAH
# =============================

if "Sog" in df.columns:
    df["Sog"] = pd.to_numeric(df["Sog"], errors="coerce")
    df.loc[df["Sog"] == 1023, "Sog"] = pd.NA
    df["Sog"] = df["Sog"] / 10

if "Cog" in df.columns:
    df["Cog"] = pd.to_numeric(df["Cog"], errors="coerce")
    df.loc[df["Cog"] == 3600, "Cog"] = pd.NA
    df["Cog"] = df["Cog"] / 10

if "True_Head" in df.columns:
    df["True_Head"] = pd.to_numeric(df["True_Head"], errors="coerce")
    df.loc[df["True_Head"] == 511, "True_Head"] = pd.NA


# =============================
# AMBIL TIMESTAMP DARI SCENE
# =============================

df["SAR_Start_Time"] = df["scene"].apply(extract_scene_start_time)
df["SAR_End_Time"] = df["scene"].apply(extract_scene_end_time)
df["Detection_Time"] = df["SAR_Start_Time"]


# =============================
# KOORDINAT SAR
# =============================

df["SAR_Longitude"] = pd.to_numeric(df["Center_longitude"], errors="coerce")
df["SAR_Latitude"] = pd.to_numeric(df["Center_latitude"], errors="coerce")


# =============================
# PILIH KOLOM OUTPUT
# =============================

output_cols = [
    "scene",
    "patch_cal",
    "SAR_Longitude",
    "SAR_Latitude",
    "SAR_Start_Time",
    "SAR_End_Time",
    "Detection_Time",
    "MMSI",
    "Sog",
    "Cog",
    "True_Head",
    "AIS_Longitude",
    "AIS_Latitude",
    "AIS_Length",
    "AIS_Width",
    "Ship_Type",
    "Nav_Status",
    "Draught",
    "Gross_tonnage",
    "Deadweight",
    "category",
]

output_cols = [col for col in output_cols if col in df.columns]

fusion = df[output_cols].copy()

if "patch_cal" in fusion.columns:
    fusion["patch_cal"] = fusion["patch_cal"].astype(str).str.replace("\\", "/", regex=False)

if "MMSI" in fusion.columns:
    fusion["has_ais"] = fusion["MMSI"].notna() & (fusion["MMSI"].astype(str).str.strip() != "")
else:
    fusion["has_ais"] = False

fusion.to_csv(out_path, index=False)

print("Input :", meta_path)
print("Output:", out_path)
print("Rows  :", len(fusion), "Cols:", fusion.shape[1])
print("Kolom output:", list(fusion.columns))

print("\nContoh timestamp:")
print(fusion[["scene", "SAR_Start_Time", "SAR_End_Time", "Detection_Time"]].head())
