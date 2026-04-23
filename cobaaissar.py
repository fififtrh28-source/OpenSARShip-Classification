import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta_path = HERE / "dataset" / "metadata.csv"
out_path  = HERE / "dataset" / "fusion_minimal.csv"

if not meta_path.exists():
    raise FileNotFoundError(f"metadata.csv tidak ditemukan: {meta_path}")

df = pd.read_csv(meta_path)

if "Cog" in df.columns:
    df["Cog"] = df["Cog"] / 10

# --- pilih kolom SAR + AIS yang paling “minimal tapi kuat” ---
# (kalau ada kolom yang tidak ada di metadata-mu, script akan otomatis skip)
sar_cols_candidates = [
    "scene", "patch_cal", "product_type", "satellite", "polarization",
    "Incidence", "Imaging_time",
    "chip_x", "chip_y", "chip_width", "chip_height",
    "ship_x", "ship_y", "ship_width", "ship_height",
    "Sar_Longitude", "Sar_Latitude"
]

ais_cols_candidates = [
    "MMSI", "Sog", "Cog", "True_Head",
    "AIS_Longitude", "AIS_Latitude",
    "AIS_Length", "AIS_Width",
    "Ship_type", "Nav_status",
    "Draught", "Gross_tonnage", "Deadweight"
]

label_cols_candidates = ["label", "vessel_type", "shiptype", "class"]

keep = [c for c in (sar_cols_candidates + ais_cols_candidates + label_cols_candidates) if c in df.columns]

fusion = df[keep].copy()

# --- rapihin sedikit (opsional tapi bagus buat ditunjukin) ---
# normalisasi path windows biar konsisten
if "patch_cal" in fusion.columns:
    fusion["patch_cal"] = fusion["patch_cal"].astype(str).str.replace("\\", "/", regex=False)

# drop baris yang MMSI kosong (biar fusion AIS beneran ada)
if "MMSI" in fusion.columns:
    fusion = fusion.dropna(subset=["MMSI"])

fusion.to_csv(out_path, index=False)

print("Input :", meta_path)
print("Output:", out_path)
print("Rows  :", len(fusion), "Cols:", fusion.shape[1])
print("Kolom output:", list(fusion.columns))
