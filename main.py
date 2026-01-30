import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

# CONFIGURATION — documented for reproducibility
np.random.seed(42)

DATA_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv"
)

WINDOW_SIZE = 12            # moving-average window
CONTAMINATION = 0.02        # expected anomaly proportion
SAVE_FIGS = True            # set False if not needed
OUTPUT_DIR = "figures"      # figures saved here

# STAGE 1 — LOAD & CLEAN DATA
df = pd.read_csv(DATA_URL)
df = df.dropna(subset=['value'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# STAGE 2 — TREND + RESIDUAL (DSP)
df['trend'] = df['value'].rolling(
    window=WINDOW_SIZE,
    center=True
).mean()

df['residual'] = df['value'] - df['trend']
df = df.dropna(subset=['residual'])

scaler = StandardScaler()
df['residual_scaled'] = scaler.fit_transform(df[['residual']])

# STAGE 3 — DESCRIPTIVE STATISTICS
desc = df['value'].describe()
sk = skew(df['value'])
kt = kurtosis(df['value'], fisher=False)
corr = df['value'].corr(df['trend'])

print("DESCRIPTIVE STATISTICS")
print(desc)
print(f"Skewness: {sk:.3f}")
print(f"Kurtosis: {kt:.3f}")
print(f"Raw/Trend correlation: {corr:.3f}")

# FIGURE 1 — histogram
plt.style.use("default")
plt.figure(figsize=(12,4))
plt.hist(df['value'], bins=60)
plt.title("Distribution of CPU Utilisation", fontsize=13)
plt.xlabel("CPU Utilisation (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    import os; os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/figure1_histogram.png", dpi=300)

plt.show()

# STAGE 4 — ISOLATION FOREST (RAW vs RESIDUAL)
model_raw = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42
)
df['pred_raw'] = (model_raw.fit_predict(df[['value']]) == -1).astype(int)

model_res = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42
)
df['pred_res'] = (model_res.fit_predict(df[['residual_scaled']]) == -1).astype(int)

raw_count = df['pred_raw'].sum()
res_count = df['pred_res'].sum()

df['masked_anomaly'] = (df['pred_res'] == 1) & (df['pred_raw'] == 0)

print("ANOMALY COUNTS")
print(f"Raw signal anomalies: {raw_count}")
print(f"Residual anomalies: {res_count}")
print(f"Masked anomalies (only visible after detrending): {df['masked_anomaly'].sum()}")

# STAGE 4B — STATISTICAL BASELINE (ROLLING MEAN + 3σ)

df['roll_mean'] = df['value'].rolling(window=WINDOW_SIZE).mean()
df['roll_std'] = df['value'].rolling(window=WINDOW_SIZE).std()

upper = df['roll_mean'] + 3 * df['roll_std']
lower = df['roll_mean'] - 3 * df['roll_std']

df['pred_3sigma'] = ((df['value'] > upper) | (df['value'] < lower)).astype(int)

baseline_count = df['pred_3sigma'].sum()

print(f"3σ baseline anomalies: {baseline_count}")

# STAGE 5 — VISUALISATION PIPELINE
plt.figure(figsize=(15,4))
plt.fill_between(df['timestamp'], df['value'],
                 color='steelblue', alpha=0.35, label="Raw")
plt.plot(df['timestamp'], df['trend'],
         color='red', linewidth=1.2, label="Trend")
plt.title("Signal Decomposition — Raw vs Moving Average Trend")
plt.legend()
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    plt.savefig(f"{OUTPUT_DIR}/figure2_raw_vs_trend.png", dpi=300)

plt.figure(figsize=(15,4))
plt.plot(df['timestamp'], df['residual'], color='green')
plt.axhline(0, linestyle='--', color='black', linewidth=1)
plt.title("Residual Component After Detrending")
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    plt.savefig(f"{OUTPUT_DIR}/figure3_residual.png", dpi=300)

plt.figure(figsize=(15,4))
plt.plot(df['timestamp'], df['value'], color='grey', alpha=0.25)

masked = df[df['masked_anomaly']]
plt.scatter(masked['timestamp'], masked['value'],
            color='blue', s=12, label="Masked anomalies")

plt.title("Hybrid Detection — Anomalies Revealed Only After Detrending")
plt.legend()
plt.tight_layout()
plt.show()

if SAVE_FIGS:
    plt.savefig(f"{OUTPUT_DIR}/figure4_masked_anomalies.png", dpi=300)

# STAGE 6 — WINDOW SENSITIVITY ANALYSIS

def detect_with_window(w):
    tmp = df[['timestamp','value']].copy()
    tmp['trend'] = tmp['value'].rolling(window=w, center=True).mean()
    tmp['residual'] = tmp['value'] - tmp['trend']
    tmp = tmp.dropna()
    tmp['residual_scaled'] = scaler.fit_transform(tmp[['residual']])
    model = IsolationForest(contamination=CONTAMINATION, random_state=42)
    tmp['pred'] = (model.fit_predict(tmp[['residual_scaled']]) == -1).astype(int)
    return tmp['pred'].sum()

windows = [6, 12, 24, 48]
results = {w: detect_with_window(w) for w in windows}

print("WINDOW SENSITIVITY")
for w, c in results.items():
    print(f"Window {w}: {c} anomalies detected")

# STAGE 7 — SUMMARY
print(f"Residual mean: {df['residual'].mean():.5f}")

both = ((df['pred_raw']==1) & (df['pred_res']==1)).sum()
print(f"Anomalies detected in both raw and residual: {both}")