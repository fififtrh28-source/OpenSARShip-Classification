import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta_path = HERE / "dataset" / "metadata.csv"
out_path  = HERE / "dataset" / "fusion_minimal.csv"

df = pd.read_csv(meta_path)

sar_cols = ["scene", "patch_cal", "Incidence"]
ais_cols = [
    "MMSI", "Sog", "Cog", "True_Head",
    "AIS_Longitude", "AIS_Latitude",
    "AIS_Length", "AIS_Width",
    "Draught", "Gross_tonnage", "Deadweight"
]

# kolom label/tipe kapal (coba beberapa kemungkinan nama)
type_cols = ["category", "Category", "Ship_type", "Ship_Type", "vessel_type", "label"]

keep = [c for c in (sar_cols + ais_cols + type_cols) if c in df.columns]

fusion = df[keep].copy()

# rapihin path
if "patch_cal" in fusion.columns:
    fusion["patch_cal"] = fusion["patch_cal"].astype(str).str.replace("\\", "/", regex=False)

# drop yang MMSI kosong
if "MMSI" in fusion.columns:
    fusion = fusion.dropna(subset=["MMSI"])

fusion.to_csv(out_path, index=False)

print("✅ Fusion minimal (dengan tipe kapal) selesai!")
print("Output:", out_path)
print("Cols:", list(fusion.columns))
print("Unique type col preview:")
for c in ["category", "Category", "Ship_type", "Ship_Type", "vessel_type", "label"]:
    if c in fusion.columns:
        print(c, fusion[c].dropna().astype(str).value_counts().head(10))
        break
