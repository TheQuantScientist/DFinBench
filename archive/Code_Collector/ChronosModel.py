import os
import gc
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline

# ================= CONFIG =================
DATA_ROOT_STOCK = r"D:\DFBENCH\dataset\stock"

SEQ_LENS = [30, 60, 90]

PRED_LEN = 30
MAX_CONTEXT = 256
NUM_SAMPLES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD FILES =================
def load_stock_files(folder):
    files = os.listdir(folder)
    stock_files = {}

    for f in files:
        if f.endswith(".csv"):
            ticker = f.split("_")[0]
            stock_files[ticker] = os.path.join(folder, f)

    return stock_files

# ================= LOAD DATA =================
def load_and_combine_multi_asset_data(file_paths, assets):
    data = {}

    for asset, path in file_paths.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[asset] = df

    df_all = None

    for asset in assets:
        renamed = data[asset].rename(columns={
            'open':   f'{asset}_open',
            'high':   f'{asset}_high',
            'low':    f'{asset}_low',
            'close':  f'{asset}_close',
            'volume': f'{asset}_volume'
        })

        df_all = renamed if df_all is None else df_all.join(renamed, how='outer')

    return df_all.ffill().bfill().sort_index()

# ================= NORMALIZATION =================
def normalize(df):
    mean = df.mean()
    std = df.std() + 1e-8
    return (df - mean) / std, mean, std

def denormalize(df, mean, std):
    return df * std + mean

# ================= CHRONOS =================
def load_chronos():
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
    pipeline.model = pipeline.model.to(DEVICE)
    return pipeline

def forecast_chronos(pipeline, context, pred_len):
    context_tensor = torch.tensor(context.T, dtype=torch.float32).to(DEVICE)

    forecast = pipeline.predict(
        context_tensor,
        prediction_length=pred_len,
        num_samples=NUM_SAMPLES
    )

    forecast = forecast.cpu().numpy()

    if forecast.ndim == 2:
        forecast = forecast.reshape(context_tensor.shape[0], 1, -1)

    forecast_mean = forecast.mean(axis=1)

    return forecast_mean.T

# ================= MAIN =================
def run_chronos_scenario():

    print("\n" + "="*80)
    print("RUNNING CHRONOS (80:10:10 SPLIT)")
    print("="*80)

    STOCK_FILES = load_stock_files(DATA_ROOT_STOCK)
    ALL_STOCKS = list(STOCK_FILES.keys())

    print(f"Total stocks: {len(ALL_STOCKS)}")

    df_all = load_and_combine_multi_asset_data(STOCK_FILES, ALL_STOCKS)

    close_cols = [f"{a}_close" for a in ALL_STOCKS]
    df_close = df_all[close_cols]

    df_norm, mean, std = normalize(df_close)

    # ===== SPLIT 80:10:10 =====
    n = len(df_norm)

    train_size = int(n * 0.8)
    val_size   = int(n * 0.1)
    test_size  = n - train_size - val_size

    train_df = df_norm.iloc[:train_size]
    val_df   = df_norm.iloc[train_size:train_size + val_size]
    test_df  = df_norm.iloc[train_size + val_size:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    pipeline = load_chronos()

    for SEQ_LEN in SEQ_LENS:

        print("\n" + "-"*60)
        print(f"SEQ_LEN = {SEQ_LEN}")
        print("-"*60)

        preds_all = []
        trues_all = []

        context_df = pd.concat([train_df, val_df])
        context_values = context_df.values
        test_values = test_df.values

        i = 0

        while i < len(test_values):

            context_window = context_values[-min(SEQ_LEN, MAX_CONTEXT):]

            forecast = forecast_chronos(pipeline, context_window, PRED_LEN)

            steps = min(PRED_LEN, len(test_values) - i)

            preds_chunk = forecast[:steps]
            true_chunk = test_values[i:i+steps]

            preds_all.append(preds_chunk)
            trues_all.append(true_chunk)

            # teacher forcing
            context_values = np.vstack([context_values, true_chunk])

            i += steps

        preds_all = np.concatenate(preds_all, axis=0)
        trues_all = np.concatenate(trues_all, axis=0)

        if preds_all.ndim == 3:
            preds_all = preds_all.mean(axis=-1)

        preds_denorm = denormalize(
            pd.DataFrame(preds_all, columns=close_cols),
            mean, std
        ).values

        trues_denorm = denormalize(
            pd.DataFrame(trues_all, columns=close_cols),
            mean, std
        ).values

        err = preds_denorm - trues_denorm

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))

        print(f"RMSE: {rmse:.6f}")
        print(f"MAE : {mae:.6f}")

        forecast_index = test_df.index[:len(preds_denorm)]

        final_df = pd.DataFrame(
            preds_denorm,
            index=forecast_index,
            columns=[f"{c}_pred" for c in close_cols]
        )

        filename = f"chronos_40stocks_seq{SEQ_LEN}.csv"
        final_df.to_csv(filename)

        print(f"→ Saved: {filename}")

        gc.collect()

# ================= RUN =================
if __name__ == "__main__":
    run_chronos_scenario()