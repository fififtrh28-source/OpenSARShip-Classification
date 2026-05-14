import pandas as pd
from pathlib import Path
from datetime import datetime

print("alert_engine.py mulai dijalankan...")

HERE = Path(__file__).resolve().parent

input_path = HERE / "kalman_ais_result.csv"
output_path = HERE / "dataset" / "dark_vessel_alerts.csv"

print("Input :", input_path)
print("Output:", output_path)

if not input_path.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

df = pd.read_csv(input_path)
print("Jumlah data input:", len(df))
print("Kolom:", list(df.columns))

alerts = []

for idx, row in df.iterrows():
    score = 0
    reasons = []

    mmsi = row.get("MMSI")
    sog = pd.to_numeric(row.get("Sog"), errors="coerce")
    ship_type = str(row.get("Ship_Type", "")).lower()
    nav_status = str(row.get("Nav_Status", "")).lower()
    kalman_shift = pd.to_numeric(row.get("Kalman_Shift_m"), errors="coerce")

    if pd.isna(mmsi) or str(mmsi).strip() == "":
        score += 70
        reasons.append("Objek terdeteksi tetapi tidak memiliki data MMSI/AIS.")

    if "fishing" in ship_type:
        score += 20
        reasons.append("Jenis kapal terindikasi fishing vessel.")

    if pd.notna(sog) and sog <= 3:
        score += 10
        reasons.append("Kecepatan kapal rendah sehingga perlu diverifikasi.")

    if nav_status.strip() == "" or "unknown" in nav_status or "not defined" in nav_status:
        score += 10
        reasons.append("Status navigasi tidak jelas atau tidak tersedia.")

    if pd.notna(kalman_shift) and kalman_shift >= 500:
        score += 20
        reasons.append(f"Terdapat anomali pergeseran posisi sebesar {kalman_shift:.2f} meter.")

    if score >= 70:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    elif score >= 20:
        level = "LOW"
    else:
        level = None

    if level is not None:
        alerts.append({
            "alert_id": f"ALERT-{idx+1:04d}",
            "alert_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alert_level": level,
            "risk_score": score,
            "mmsi": mmsi,
            "latitude": row.get("Kalman_Lat"),
            "longitude": row.get("Kalman_Lon"),
            "sog": row.get("Sog"),
            "cog": row.get("Cog"),
            "ship_type": row.get("Ship_Type"),
            "nav_status": row.get("Nav_Status"),
            "kalman_shift_m": kalman_shift,
            "reason": " | ".join(reasons),
            "recipient": "BAKAMLA, TNI AL, KKP, KEMENHUB",
            "status": "Perlu verifikasi operator"
        })

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv(output_path, index=False)

print("Jumlah alert:", len(alerts_df))
print("File alert berhasil dibuat:", output_path)

if len(alerts_df) > 0:
    print(alerts_df[[
        "alert_id",
        "alert_level",
        "risk_score",
        "mmsi",
        "latitude",
        "longitude",
        "kalman_shift_m",
        "recipient"
    ]].head(10))
else:
    print("Tidak ada alert yang memenuhi kriteria.")

print("alert_engine.py selesai dijalankan.")