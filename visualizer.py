import json
import pandas as pd
import matplotlib.pyplot as plt
from setup_db import get_connection

def get_recent_rows(limit=50):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, location_name, precipitation, created_at FROM rainfall_data ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def extract_series(precip_json):
    try:
        if isinstance(precip_json, str):
            js = json.loads(precip_json)
        else:
            js = precip_json
    except Exception:
        return [], []
    ts = js.get("ts") or js.get("timestamps") or []
    vals = js.get("precip-surface") or js.get("precip") or js.get("precipitation") or []
    return ts, vals

def build_dataframe(rows):
    records = []
    for r in rows:
        ts, vals = extract_series(r["precipitation"])
        for t, v in zip(ts, vals):
            # Windy ts typically epoch seconds
            try:
                t_dt = pd.to_datetime(t, unit="s")
            except:
                t_dt = pd.to_datetime(t)
            records.append({"location": r["location_name"], "time": t_dt, "rain_mm": float(v) if v is not None else 0.0})
    if not records:
        return pd.DataFrame(columns=["location", "time", "rain_mm"])
    df = pd.DataFrame(records)
    return df

def plot_timeseries(df):
    if df.empty:
        print("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(12,6))
    for name, g in df.groupby("location"):
        ax.plot(g["time"], g["rain_mm"], marker="o", label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Precip (mm)")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    rows = get_recent_rows(60)
    df = build_dataframe(rows)
    plot_timeseries(df)
