import pandas as pd
import numpy as np
from pathlib import Path

# =============================
# PATH
# =============================
HERE = Path(__file__).resolve().parent

candidate_paths = [
    HERE / "fusion_minimal.csv",
    HERE / "dataset" / "fusion_minimal.csv"
]

csv_path = None
for p in candidate_paths:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError(
        "fusion_minimal.csv tidak ditemukan.\n"
        "Cek apakah file ada di folder script atau di folder dataset/"
    )

out_csv = HERE / "kalman_ais_result.csv"

print("Reading:", csv_path)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(csv_path)

# simpan urutan asli
df["_orig_order"] = np.arange(len(df))

# =============================
# CLEANING
# =============================
needed = ["MMSI", "AIS_Latitude", "AIS_Longitude", "Sog", "Cog"]
df = df.dropna(subset=needed).copy()

df = df[
    df["AIS_Latitude"].between(-90, 90) &
    df["AIS_Longitude"].between(-180, 180)
].copy()

df["Sog"] = pd.to_numeric(df["Sog"], errors="coerce")
df["Cog"] = pd.to_numeric(df["Cog"], errors="coerce")

if "True_Head" in df.columns:
    df["True_Head"] = pd.to_numeric(df["True_Head"], errors="coerce")
    df.loc[df["True_Head"] == 511, "True_Head"] = np.nan

# nilai khusus AIS
df.loc[df["Sog"] == 1023, "Sog"] = np.nan
df.loc[df["Cog"] == 3600, "Cog"] = np.nan

# scale AIS
df["Sog"] = df["Sog"] / 10.0   # knot
df["Cog"] = df["Cog"] / 10.0   # degree

df = df.dropna(subset=["Sog", "Cog"]).copy()
counts = df["MMSI"].value_counts()

print("\n=== RINGKASAN MMSI ===")
print("Total baris:", len(df))
print("Jumlah MMSI unik:", counts.shape[0])
print("MMSI yang muncul >= 2 kali:", (counts >= 2).sum())
print("MMSI yang muncul >= 3 kali:", (counts >= 3).sum())

print("\nTop 20 MMSI terbanyak:")
print(counts.head(20))

# tanpa timestamp, minimal pertahankan urutan asli per MMSI
df = df.sort_values(["MMSI", "_orig_order"]).reset_index(drop=True)

# =============================
# HELPER: lat/lon <-> meter
# =============================
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y

def xy_to_latlon(x, y, lat0, lon0):000
    R = 6371000.0
    lat = np.degrees(y / R) + lat0
    lon = np.degrees(x / (R * np.cos(np.radians(lat0)))) + lon0
    return lat, lon

# =============================
# KALMAN FILTER 2D
# state = [x, y, vx, vy]
# meas  = [x, y]
# =============================
def kalman_filter_track(group, dt=1.0):
    group = group.copy().reset_index(drop=True)

    lat0 = group["AIS_Latitude"].iloc[0]
    lon0 = group["AIS_Longitude"].iloc[0]

    obs_xy = group.apply(
        lambda r: latlon_to_xy(r["AIS_Latitude"], r["AIS_Longitude"], lat0, lon0),
        axis=1
    )
    obs_xy = np.array(list(obs_xy))
    zxs = obs_xy[:, 0]
    zys = obs_xy[:, 1]

    sog0 = group["Sog"].iloc[0] * 0.514444  # knot -> m/s
    cog0 = np.radians(group["Cog"].iloc[0])

    vx0 = sog0 * np.sin(cog0)
    vy0 = sog0 * np.cos(cog0)

    x = np.array([[zxs[0]],
                  [zys[0]],
                  [vx0],
                  [vy0]], dtype=float)

    P = np.diag([100.0, 100.0, 25.0, 25.0])

    F = np.array([
        [1, 0, dt, 0 ],
        [0, 1, 0,  dt],
        [0, 0, 1,  0 ],
        [0, 0, 0,  1 ]
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    q = 5.0
    Q = np.array([
        [q, 0, 0, 0],
        [0, q, 0, 0],
        [0, 0, q, 0],
        [0, 0, 0, q]
    ], dtype=float)

    r = 20.0
    Rm = np.array([
        [r, 0],
        [0, r]
    ], dtype=float)

    I = np.eye(4)

    filt_x, filt_y = [], []
    pred_x, pred_y = [], []

    for i in range(len(group)):
        x = F @ x
        P = F @ P @ F.T + Q

        pred_x.append(float(x[0, 0]))
        pred_y.append(float(x[1, 0]))

        z = np.array([[zxs[i]], [zys[i]]], dtype=float)
        y_res = z - (H @ x)
        S = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y_res
        P = (I - K @ H) @ P

        filt_x.append(float(x[0, 0]))
        filt_y.append(float(x[1, 0]))

    pred_latlon = [xy_to_latlon(px, py, lat0, lon0) for px, py in zip(pred_x, pred_y)]
    filt_latlon = [xy_to_latlon(fx, fy, lat0, lon0) for fx, fy in zip(filt_x, filt_y)]

    group["Pred_Lat"] = [p[0] for p in pred_latlon]
    group["Pred_Lon"] = [p[1] for p in pred_latlon]
    group["Kalman_Lat"] = [p[0] for p in filt_latlon]
    group["Kalman_Lon"] = [p[1] for p in filt_latlon]

    return group

# =============================
# APPLY PER MMSI
# =============================
results = []

for mmsi, g in df.groupby("MMSI"):
    if len(g) < 2:
        g = g.copy()
        g["Pred_Lat"] = g["AIS_Latitude"]
        g["Pred_Lon"] = g["AIS_Longitude"]
        g["Kalman_Lat"] = g["AIS_Latitude"]
        g["Kalman_Lon"] = g["AIS_Longitude"]
        results.append(g)
    else:
        results.append(kalman_filter_track(g, dt=1.0))

result_df = pd.concat(results, ignore_index=True)
# =============================
# HITUNG SHIFT KALMAN
# =============================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# bikin kolom shift dulu
if "Kalman_Lat" in result_df.columns and "Kalman_Lon" in result_df.columns:
    result_df["Kalman_Shift_m"] = haversine_m(
        result_df["AIS_Latitude"],
        result_df["AIS_Longitude"],
        result_df["Kalman_Lat"],
        result_df["Kalman_Lon"]
    )
else:
    print("Kolom Kalman_Lat / Kalman_Lon belum ada")

# simpan hasil
result_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)

print("\n=== HASIL SHIFT ===")
if "Kalman_Shift_m" in result_df.columns:
    print(result_df["Kalman_Shift_m"].describe())
else:
    print("Kolom Kalman_Shift_m belum terbentuk")

print("\nContoh hasil:")
print(result_df[[
    "MMSI", "AIS_Latitude", "AIS_Longitude",
    "Kalman_Lat", "Kalman_Lon", "Kalman_Shift_m"
]].head())

print("\nContoh data dengan shift terbesar:")
print(
    result_df.sort_values("Kalman_Shift_m", ascending=False)[
        ["MMSI", "AIS_Latitude", "AIS_Longitude", "Kalman_Lat", "Kalman_Lon", "Kalman_Shift_m"]
    ].head(10)
)

# =============================
# ERROR / SHIFT CHECK
# =============================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

result_df["Kalman_Shift_m"] = haversine_m(
    result_df["AIS_Latitude"],
    result_df["AIS_Longitude"],
    result_df["Kalman_Lat"],
    result_df["Kalman_Lon"]
)

# hapus helper kolom
result_df = result_df.drop(columns=["_orig_order"], errors="ignore")

# =============================
# ERROR / SHIFT CHECK
# =============================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

print("Kolom result_df sebelum shift:")
print(result_df.columns.tolist())

if "Kalman_Lat" in result_df.columns and "Kalman_Lon" in result_df.columns:
    result_df["Kalman_Shift_m"] = haversine_m(
        result_df["AIS_Latitude"],
        result_df["AIS_Longitude"],
        result_df["Kalman_Lat"],
        result_df["Kalman_Lon"]
    )
    print("\nStatistik Kalman_Shift_m:")
    print(result_df["Kalman_Shift_m"].describe())
else:
    print("Kolom Kalman_Lat / Kalman_Lon belum ada, jadi Kalman_Shift_m tidak bisa dibuat.")