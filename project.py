import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm

# HTM core imports
from htm.bindings.sdr import SDR
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters, DateEncoder
from htm.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.algorithms import Predictor


# ============================
# 1. Config
# ============================

DATA_PATH = "data/series.csv"  # <-- CHANGE to your csv path
TIME_COL  = "timestamp"        # <-- CHANGE if different
VALUE_COL = "value"            # <-- CHANGE if different

TRAIN_RATIO = 0.8              # 80% train, 20% test

# SARIMAX hyperparameters (adjust for your data)
# Example: hourly data with daily seasonality -> s=24
SARIMA_ORDER = (2, 1, 2)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)


# ============================
# 2. Load & preprocess data
# ============================
df = pd.read_csv(DATA_PATH)

# parse and sort by time
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

df = df[[TIME_COL, VALUE_COL]].dropna()
df.rename(columns={TIME_COL: "timestamp", VALUE_COL: "value"}, inplace=True)

print("Total rows:", len(df))
print(df.head())

value_min = df["value"].min()
value_max = df["value"].max()
print("Value range:", value_min, "->", value_max)

# train/test split index
train_size = int(len(df) * TRAIN_RATIO)
print(f"Train size: {train_size}, Test size: {len(df) - train_size}")


# ============================
# 3. Build HTM components
# ============================

# --- Encoders ---
val_params = ScalarEncoderParameters()
val_params.minimum = float(value_min)
val_params.maximum = float(value_max)
val_params.size = 400            # total bits
val_params.activeBits = 21       # active bits
val_params.periodic = False

value_encoder = ScalarEncoder(val_params)

# time encoder: time-of-day + weekend
time_of_day = (21, 9.5)          # (bits, radius in hours)
weekend_bits = 21

time_encoder = DateEncoder(
    timeOfDay=time_of_day,
    weekend=weekend_bits
)

value_sdr_size = value_encoder.getWidth()
time_sdr_size = time_encoder.getWidth()
input_sdr_size = value_sdr_size + time_sdr_size

print("Value SDR size:", value_sdr_size)
print("Time SDR size:", time_sdr_size)
print("Total input SDR size:", input_sdr_size)

# --- Spatial Pooler ---
sp = SpatialPooler(
    inputDimensions=(input_sdr_size,),
    columnDimensions=(2048,),
    potentialPct=0.85,
    globalInhibition=True,
    localAreaDensity=-1.0,
    numActiveColumnsPerInhArea=40,
    synPermInactiveDec=0.008,
    synPermActiveInc=0.05,
    synPermConnected=0.1,
    boostStrength=3.0
)

# --- Temporal Memory ---
tm = TemporalMemory(
    columnDimensions=(2048,),
    cellsPerColumn=32,
    activationThreshold=12,
    initialPermanence=0.21,
    connectedPermanence=0.1,
    minThreshold=9,
    maxNewSynapseCount=20,
    permanenceIncrement=0.10,
    permanenceDecrement=0.10,
    predictedSegmentDecrement=0.0
)

# --- Predictor (1 step ahead) ---
predictor = Predictor(steps=[1], alpha=0.1)

# Reusable SDRs
input_sdr = SDR(input_sdr_size)
sp_output = SDR(sp.getColumnDimensions())
active_cells = SDR(tm.numberOfCells())


# ============================
# 4. HTM training + inference
# ============================

actual_values = []
htm_predictions = []
htm_anomaly_score = []

# mapping value -> bucket index for predictor
bucket_idx = {}
bucket_counter = 0

def get_bucket_index(v):
    global bucket_counter
    key = float(round(v, 2))
    if key not in bucket_idx:
        bucket_idx[key] = bucket_counter
        bucket_counter += 1
    return bucket_idx[key]

print("\n=== Running HTM over sequence ===")

for i, row in df.iterrows():
    v = float(row["value"])
    ts = row["timestamp"]

    learn_phase = i < train_size   # learn only on train portion

    # -------- Encode --------
    value_sdr = value_encoder.encode(v)
    time_sdr = time_encoder.encode(ts)

    input_sdr.zero()
    input_sdr.dense[0:value_sdr.size] = value_sdr.dense
    input_sdr.dense[value_sdr.size:] = time_sdr.dense

    # -------- Spatial Pooler --------
    sp_output.zero()
    sp.compute(input_sdr, learn_phase, sp_output)

    # -------- Temporal Memory --------
    tm.compute(sp_output, learn_phase)
    tm.activateDendrites(learn_phase)

    active_cells.sparse = tm.getActiveCells()

    # -------- Predictor --------
    if learn_phase:
        b_idx = get_bucket_index(v)
        predictor.learn(active_cells, b_idx)

    infer = predictor.infer(active_cells)
    pred_dist = infer[1]  # 1-step ahead

    if pred_dist:
        best_bucket = max(pred_dist, key=pred_dist.get)
        inv_map = {idx: val for val, idx in bucket_idx.items()}
        pred_value = inv_map.get(best_bucket, v)
    else:
        pred_value = v  # fallback early (before predictor learns)

    # -------- Simple anomaly score --------
    predictive_cells = set(tm.getPredictiveCells())
    active_set = set(active_cells.sparse)

    if len(active_set) > 0:
        overlap = len(active_set & predictive_cells) / float(len(active_set))
        anomaly = 1.0 - overlap
    else:
        anomaly = 0.0

    actual_values.append(v)
    htm_predictions.append(pred_value)
    htm_anomaly_score.append(anomaly)

df["htm_pred"] = htm_predictions
df["htm_anomaly"] = htm_anomaly_score


# ============================
# 5. SARIMAX benchmark
# ============================

print("\n=== Training SARIMAX benchmark ===")

df_bench = df.set_index("timestamp")

train_series = df_bench["value"].iloc[:train_size]
test_series = df_bench["value"].iloc[train_size:]

sarimax_model = sm.tsa.statespace.SARIMAX(
    train_series,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_res = sarimax_model.fit(disp=False)

# Forecast over test period
sarimax_forecast = sarimax_res.predict(
    start=train_series.index[0],
    end=df_bench.index[-1]
)
# Align with original df
df["sarimax_pred"] = sarimax_forecast.values


# ============================
# 6. Evaluation metrics
# ============================

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

test_df = df.iloc[train_size:].copy()

htm_rmse, htm_mae = regression_metrics(test_df["value"], test_df["htm_pred"])
sar_rmse, sar_mae = regression_metrics(test_df["value"], test_df["sarimax_pred"])

print("\n=== Test-set Forecast Accuracy ===")
print(f"HTM      -> RMSE: {htm_rmse:.4f}, MAE: {htm_mae:.4f}")
print(f"SARIMAX  -> RMSE: {sar_rmse:.4f}, MAE: {sar_mae:.4f}")


# ============================
# 7. Plots
# ============================

# 7.1 Forecast comparison (full series)
plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["value"], label="Actual", linewidth=1)
plt.plot(df["timestamp"], df["htm_pred"], label="HTM Pred", linewidth=1)
plt.plot(df["timestamp"], df["sarimax_pred"], label="SARIMAX Pred", linewidth=1)
plt.axvline(df["timestamp"].iloc[train_size], linestyle="--", label="Train/Test Split")
plt.title("Actual vs HTM & SARIMAX Predictions (Full Series)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# 7.2 Zoom on test region
plt.figure(figsize=(14, 6))
plt.plot(test_df["timestamp"], test_df["value"], label="Actual", linewidth=1)
plt.plot(test_df["timestamp"], test_df["htm_pred"], label="HTM Pred", linewidth=1)
plt.plot(test_df["timestamp"], test_df["sarimax_pred"], label="SARIMAX Pred", linewidth=1)
plt.title("Test Set: Actual vs Predictions")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# 7.3 HTM anomaly score
plt.figure(figsize=(14, 3))
plt.plot(df["timestamp"], df["htm_anomaly"])
plt.axvline(df["timestamp"].iloc[train_size], linestyle="--")
plt.title("HTM Anomaly Score (0–1)")
plt.xlabel("Time")
plt.ylabel("Anomaly")
plt.tight_layout()
plt.show()
