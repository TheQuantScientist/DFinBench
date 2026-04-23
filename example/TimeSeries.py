import os
import sys
import warnings
import math
import argparse
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from arch import arch_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Path setup ─────────────────────────────────────────────────────────────
sys.path.append('D:/VSCODE/Time-Series-Library')

from utils.timefeatures import time_features

# Deep learning models
from models.TimeMixer import Model as TimeMixer
from models.TimeXer import Model as TimeXer
from models.TimesNet import Model as TimesNet
from models.TSMixer import Model as TSMixer

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT_CRYPTO = "/home/nckh2/qa/ChanFormer/dataset/crypto"
DATA_ROOT_STOCK = "/home/nckh2/qa/ChanFormer/dataset/stock"

# CRYPTO_FILES = {
#     'ATOMUSDT': f'{DATA_ROOT_CRYPTO}/ATOMUSDT_1d_full.csv',
#     'BCHUSDT':  f'{DATA_ROOT_CRYPTO}/BCHUSDT_1d_full.csv',
#     'DOTUSDT':  f'{DATA_ROOT_CRYPTO}/DOTUSDT_1d_full.csv',
#     'HBARUSDT': f'{DATA_ROOT_CRYPTO}/HBARUSDT_1d_full.csv',
#     'LTCUSDT':  f'{DATA_ROOT_CRYPTO}/LTCUSDT_1d_full.csv',
#     'MATICUSDT':f'{DATA_ROOT_CRYPTO}/MATICUSDT_1d_full.csv',
#     'NEARUSDT': f'{DATA_ROOT_CRYPTO}/NEARUSDT_1d_full.csv',
#     'SHIBUSDT': f'{DATA_ROOT_CRYPTO}/SHIBUSDT_1d_full.csv',
#     'SUIUSDT':  f'{DATA_ROOT_CRYPTO}/SUIUSDT_1d_full.csv',
#     'XLMUSDT':  f'{DATA_ROOT_CRYPTO}/XLMUSDT_1d_full.csv',
# }

STOCK_FILES = {
    'AAPL':   f'{DATA_ROOT_STOCK}/AAPL_1d_full.csv',
    'AMZN':   f'{DATA_ROOT_STOCK}/AMZN_1d_full.csv',
    'AVGO':   f'{DATA_ROOT_STOCK}/AVGO_1d_full.csv',
    'BRK-B':  f'{DATA_ROOT_STOCK}/BRK-B_1d_full.csv',
    'GOOGL':  f'{DATA_ROOT_STOCK}/GOOGL_1d_full.csv',
    'META':   f'{DATA_ROOT_STOCK}/META_1d_full.csv',
    'MSFT':   f'{DATA_ROOT_STOCK}/MSFT_1d_full.csv',
    'NVDA':   f'{DATA_ROOT_STOCK}/NVDA_1d_full.csv',
    'TSLA':   f'{DATA_ROOT_STOCK}/TSLA_1d_full.csv',
    'TSM':    f'{DATA_ROOT_STOCK}/TSM_1d_full.csv',
}

# ─── Single-step Forecasting Settings ─────────────────────────────────────
SEQ_LEN       = 90
PRED_LEN      = 30
EPOCHS        = 600
BATCH_SIZE    = 256
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()


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


class TimeSeriesForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, time_features: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.time_features[s_begin:s_end]
        seq_y_mark = self.time_features[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float()
        )


def get_new_models(n_features: int):
    common = {
        'seq_len': SEQ_LEN,
        'label_len': 0,
        'pred_len': PRED_LEN,
        'enc_in': n_features,
        'dec_in': n_features,
        'c_out': n_features,
        'd_model': 256,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1,
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': 'd',
        'task_name': 'long_term_forecast',
        'moving_avg': 25,
        'use_norm': 1,
        'features': 'M',
        'factor': 3,
    }

    return [
        {'name': 'TimesNet',  'class': TimesNet,  'configs': {**common, 'top_k': 5, 'num_kernels': 6}},
        {'name': 'TimeMixer', 'class': TimeMixer, 'configs': {**common, 
                     'down_sampling_window': 2, 'down_sampling_layers': 3,
                     'down_sampling_method': 'avg', 'channel_independence': False,
                     'decomp_method': 'moving_avg', 'decomp_kernel': 5}},
        {'name': 'TimeXer',   'class': TimeXer,   'configs': {**common, 'patch_len': 16, 'stride': 8}},
        {'name': 'TSMixer',   'class': TSMixer,   'configs': {**common}},
    ]


def scale_and_split_data(df_all: pd.DataFrame):
    test_data = df_all.iloc[-TEST_DAYS:]
    pre_test = df_all.iloc[:-TEST_DAYS]
    val_size = int(len(pre_test) * VAL_FRACTION)

    val_data = pre_test.iloc[-val_size:]
    train_data = pre_test.iloc[:-val_size]

    columns = df_all.columns.tolist()
    scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

    def apply_scaling(df):
        arr = np.hstack([scalers[col].transform(df[[col]]) for col in columns])
        return pd.DataFrame(arr, index=df.index, columns=columns)

    return (apply_scaling(train_data).values.astype(np.float32),
            apply_scaling(val_data).values.astype(np.float32),
            apply_scaling(test_data).values.astype(np.float32),
            test_data, columns, scalers)


def run_statistical_baselines(df_all: pd.DataFrame, test_data: pd.DataFrame, columns: list, 
                              global_results: list, per_asset_results: list, scenario: str, assets: list):
    print(f"\n{'─'*90}")
    print("STATISTICAL BASELINES (Close prices only): ARIMA | GARCH | VAR")
    print(f"{'─'*90}")

    train_df = df_all.iloc[:-TEST_DAYS].copy()
    test_df = test_data.copy()

    # Get close columns only
    close_cols = [col for col in columns if col.endswith('_close')]
    asset_close_map = {asset: f"{asset}_close" for asset in assets}

    # ====================== ARIMA (per asset close) ======================
    print("→ Running ARIMA(5,1,0) on each asset's close price...")
    for asset in assets:
        col = asset_close_map[asset]
        try:
            series = train_df[col].dropna()
            if len(series) < 50:
                continue

            model = ARIMA(series, order=(5, 1, 0))
            fitted = model.fit()

            preds = []
            for actual in test_df[col]:
                fc = fitted.forecast(steps=1).iloc[0]
                preds.append(fc)
                fitted = fitted.append([actual], refit=False)

            preds = np.array(preds)
            trues = test_df[col].values

            mse = float(np.mean((preds - trues)**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(preds - trues)))

            global_results.append({'model': 'ARIMA', 'scenario': scenario, 'mse': mse, 'rmse': rmse, 'mae': mae})

            per_asset_results.append({
                'model': 'ARIMA', 'scenario': scenario, 'asset': asset,
                'mse_scaled': mse, 'rmse_scaled': rmse, 'mae_scaled': mae
            })
            print(f"  ARIMA done → {asset}")
        except Exception as e:
            print(f"  ARIMA skipped {asset}: {e}")

    # ====================== GARCH (per asset close) ======================
    print("→ Running GARCH (Random Walk baseline on close prices)...")
    for asset in assets:
        col = asset_close_map[asset]
        try:
            last_price = float(train_df[col].iloc[-1])
            preds = np.full(len(test_df), last_price)
            trues = test_df[col].values

            mse = float(np.mean((preds - trues)**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(preds - trues)))

            global_results.append({'model': 'GARCH', 'scenario': scenario, 'mse': mse, 'rmse': rmse, 'mae': mae})

            per_asset_results.append({
                'model': 'GARCH', 'scenario': scenario, 'asset': asset,
                'mse_scaled': mse, 'rmse_scaled': rmse, 'mae_scaled': mae
            })
            print(f"  GARCH done → {asset}")
        except Exception as e:
            print(f"  GARCH skipped {asset}: {e}")

    # ====================== VAR (multivariate on all closes) ======================
    print("→ Running VAR on all close prices...")
    try:
        close_train = train_df[close_cols].dropna()
        model = VAR(close_train)
        results = model.fit(maxlags=15, ic='aic')

        lag_order = results.k_ar
        history = close_train.values.copy()

        preds_list = []
        for i in range(len(test_df)):
            fc = results.forecast(history[-lag_order:], steps=1)
            preds_list.append(fc[0])
            history = np.vstack([history, test_df[close_cols].iloc[i].values])

        preds_all = np.array(preds_list)
        trues_all = test_df[close_cols].values

        err = preds_all - trues_all
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))

        global_results.append({'model': 'VAR', 'scenario': scenario, 'mse': mse, 'rmse': rmse, 'mae': mae})

        for asset in assets:
            col = asset_close_map[asset]
            idx = close_cols.index(col)
            err_asset = preds_all[:, idx] - trues_all[:, idx]
            per_asset_results.append({
                'model': 'VAR', 'scenario': scenario, 'asset': asset,
                'mse_scaled': float(np.mean(err_asset**2)),
                'rmse_scaled': float(np.sqrt(np.mean(err_asset**2))),
                'mae_scaled': float(np.mean(np.abs(err_asset)))
            })
        print("  VAR completed successfully")
    except Exception as e:
        print(f"  VAR failed: {e}")

    print("Statistical baselines (close only) completed.")


def run_scenario(scenario: str, file_paths: dict, assets: list):
    print(f"\n{'='*90}")
    print(f"SINGLE-STEP FORECASTING → {scenario.upper()}")
    print(f"{'='*90}")

    df_all = load_and_combine_multi_asset_data(file_paths, assets)
    time_feat = time_features(df_all.index, freq='d').T.astype(np.float32)

    scaled_train, scaled_val, scaled_test, test_data, columns, scalers = scale_and_split_data(df_all)
    n_features = len(columns)

    pre_test_values = np.concatenate((scaled_train, scaled_val))
    pre_test_time = time_feat[:len(pre_test_values)]
    test_time = time_feat[-TEST_DAYS:]

    full_values = np.concatenate((pre_test_values, scaled_test))
    full_time = np.concatenate((pre_test_time, test_time))

    new_models = get_new_models(n_features)
    global_results = []
    per_asset_results = []
    prefix = f"{scenario}_single_step_"

    # ==================== Deep Learning Models ====================
    for bm in new_models:
        model_name = bm['name']
        print(f"\n--- Training {model_name} ---")

        args = argparse.Namespace(**bm['configs'])
        model = bm['class'](args).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train_ds = TimeSeriesForecastDataset(scaled_train, pre_test_time[:len(scaled_train)], SEQ_LEN, PRED_LEN)
        val_ds   = TimeSeriesForecastDataset(scaled_val,   pre_test_time[len(scaled_train):], SEQ_LEN, PRED_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.zeros_like(y)
                if model_name == "TimeMixer":
                    out = model(x, None, dec_inp, None)
                else:
                    out = model(x, x_mark, dec_inp, y_mark)

                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                loss = LOSS_FN(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                    dec_inp = torch.zeros_like(y)
                    if model_name == "TimeMixer":
                        out = model(x, None, dec_inp, None)
                    else:
                        out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                    val_loss += LOSS_FN(pred, y).item()
                    n_batches += 1

            val_loss = val_loss / n_batches if n_batches > 0 else float('inf')
            train_avg = train_loss / len(train_loader)

            if (epoch + 1) % 100 == 0 or epoch == EPOCHS - 1:
                print(f"[{epoch+1:4d}] train: {train_avg:.7f}  val: {val_loss:.7f}")

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Test evaluation
        test_ds = TimeSeriesForecastDataset(full_values, full_time, SEQ_LEN, PRED_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.zeros_like(y)
                if model_name == "TimeMixer":
                    out = model(x, None, dec_inp, None)
                else:
                    out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                preds_list.append(pred.cpu().numpy())
                trues_list.append(y.cpu().numpy())

        preds_all = np.concatenate(preds_list, axis=0)
        trues_all = np.concatenate(trues_list, axis=0)

        preds_test = preds_all[-TEST_DAYS:]
        trues_test = trues_all[-TEST_DAYS:]
        forecast_index = test_data.index[-TEST_DAYS:]

        global_res = {'model': model_name, 'scenario': scenario}
        err = preds_test[:, 0, :] - trues_test[:, 0, :]
        global_res['mse'] = float(np.mean(err ** 2))
        global_res['rmse'] = float(math.sqrt(global_res['mse']))
        global_res['mae'] = float(np.mean(np.abs(err)))

        global_results.append(global_res)

        for asset in assets:
            asset_cols = [f"{asset}_{v}" for v in ['open','high','low','close','volume']]
            idxs = [columns.index(c) for c in asset_cols if c in columns]
            err = preds_test[:, 0, idxs] - trues_test[:, 0, idxs]
            per_asset_results.append({
                'model': model_name, 'scenario': scenario, 'asset': asset,
                'mse_scaled': float(np.mean(err ** 2)),
                'rmse_scaled': float(math.sqrt(np.mean(err ** 2))),
                'mae_scaled': float(np.mean(np.abs(err)))
            })

        print(f"{model_name} → RMSE: {global_res['rmse']:.6f}  MAE: {global_res['mae']:.6f}")

        # Save forecasts
        pred_prices = np.zeros_like(preds_test, dtype=np.float64)
        for i, col in enumerate(columns):
            scaler = scalers[col]
            flat = preds_test[:, 0, i].reshape(-1, 1)
            pred_prices[:, 0, i] = scaler.inverse_transform(flat).flatten()

        forecast_df = test_data.loc[forecast_index].add_suffix('_true').copy()
        horizon_cols = [f"{col}_pred_h1" for col in columns]
        pred_df = pd.DataFrame(pred_prices[:, 0, :], index=forecast_index, columns=horizon_cols)
        forecast_df = pd.concat([forecast_df, pred_df], axis=1).sort_index(axis=1)

        forecast_path = f"{prefix}forecasts_{model_name}.csv"
        forecast_df.to_csv(forecast_path)
        print(f"→ Saved: {forecast_path}")

        torch.cuda.empty_cache()
        gc.collect()

    # Statistical baselines (close only)
    run_statistical_baselines(df_all, test_data, columns, global_results, per_asset_results, scenario, assets)

    # Save final results
    pd.DataFrame(global_results).to_csv(f'{prefix}global_results.csv', index=False)
    pd.DataFrame(per_asset_results).to_csv(f'{prefix}per_asset_results.csv', index=False)

    print(f"\n✅ {scenario.upper()} Single-Step Benchmark completed!\n")


if __name__ == "__main__":
    scenarios = [
        # ("crypto", CRYPTO_FILES, list(CRYPTO_FILES.keys())),
        ("stock",  STOCK_FILES,  list(STOCK_FILES.keys())),
    ]

    for scenario_name, paths, asset_list in scenarios:
        run_scenario(scenario_name, paths, asset_list)

    print("\n🎉 All single-step benchmarks (DL + Statistical on Close only) finished successfully.")