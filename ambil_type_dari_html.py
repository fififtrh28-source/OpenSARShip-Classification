import re
import html
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
dataset = HERE / "dataset"

fusion_path = dataset / "fusion_minimal.csv"
html_path = dataset / "map_by_shiptype_colored.html"
out_path = dataset / "fusion_minimal_with_type.csv"

print("Membaca CSV:", fusion_path)
df = pd.read_csv(fusion_path)

print("Membaca HTML:", html_path)
text = html_path.read_text(encoding="utf-8", errors="ignore")
text = html.unescape(text)

pattern = re.compile(
    r"Type:\s*</b>\s*([^<]+).*?"
    r"MMSI:\s*</b>\s*([^<]+).*?"
    r"SOG:\s*</b>\s*([^<]+).*?"
    r"COG:\s*</b>\s*([^<]+).*?"
    r"SAR scene:\s*</b>\s*([^<]+).*?"
    r"patch_cal:\s*</b>\s*([^<]+)",
    re.IGNORECASE | re.DOTALL
)

records = []

for match in pattern.finditer(text):
    ship_type, mmsi, sog, cog, scene, patch_cal = match.groups()

    records.append({
        "Ship_type": ship_type.strip(),
        "MMSI": str(mmsi).strip(),
        "Sog": str(sog).strip(),
        "Cog": str(cog).strip(),
        "scene": scene.strip(),
        "patch_cal": patch_cal.strip(),
    })

type_df = pd.DataFrame(records)

print("Jumlah data type dari HTML:", len(type_df))

if type_df.empty:
    raise ValueError(
        "Gagal mengambil Type dari HTML. "
        "Kemungkinan format popup HTML berbeda."
    )

for col in ["MMSI", "Sog", "Cog", "scene", "patch_cal"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    if col in type_df.columns:
        type_df[col] = type_df[col].astype(str).str.strip()

merge_cols = ["scene", "patch_cal", "MMSI"]

df_new = df.merge(
    type_df[merge_cols + ["Ship_type"]].drop_duplicates(),
    on=merge_cols,
    how="left"
)

missing = df_new["Ship_type"].isna().sum()

if missing > 0:
    print("Masih ada Ship_type kosong setelah merge utama:", missing)

    backup = type_df[["scene", "patch_cal", "Ship_type"]].drop_duplicates()

    df_backup = df.merge(
        backup,
        on=["scene", "patch_cal"],
        how="left"
    )

    df_new["Ship_type"] = df_new["Ship_type"].fillna(df_backup["Ship_type"])

df_new["Ship_type"] = df_new["Ship_type"].fillna("Other Type")

df_new.to_csv(out_path, index=False)

print("✅ File baru dibuat:", out_path)
print()
print("Jumlah per jenis kapal:")
print(df_new["Ship_type"].value_counts())
