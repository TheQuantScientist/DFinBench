import os
import sys
import argparse
import warnings
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)

# Add Time-Series-Library path
sys.path.append('/home/nckh2/qa/Time-Series-Library')
from utils.timefeatures import time_features

from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.Informer import Model as Informer

try:
    from models.iTransformer import Model as iTransformer
    HAS_iTRANSFORMER = True
except ImportError:
    HAS_iTRANSFORMER = False
    print("iTransformer not found – skipping.")

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT_CRYPTO = "/home/nckh2/qa/ChanFormer/dataset/crypto"
DATA_ROOT_STOCK  = "/home/nckh2/qa/ChanFormer/dataset/stock"

CRYPTO_FILES = {
    'ATOMUSDT': f'{DATA_ROOT_CRYPTO}/ATOMUSDT_1d_full.csv',
    'BCHUSDT':  f'{DATA_ROOT_CRYPTO}/BCHUSDT_1d_full.csv',
    'DOTUSDT':  f'{DATA_ROOT_CRYPTO}/DOTUSDT_1d_full.csv',
    'HBARUSDT': f'{DATA_ROOT_CRYPTO}/HBARUSDT_1d_full.csv',
    'LTCUSDT':  f'{DATA_ROOT_CRYPTO}/LTCUSDT_1d_full.csv',
    'MATICUSDT':f'{DATA_ROOT_CRYPTO}/MATICUSDT_1d_full.csv',
    'NEARUSDT': f'{DATA_ROOT_CRYPTO}/NEARUSDT_1d_full.csv',
    'SHIBUSDT': f'{DATA_ROOT_CRYPTO}/SHIBUSDT_1d_full.csv',
    'SUIUSDT':  f'{DATA_ROOT_CRYPTO}/SUIUSDT_1d_full.csv',
    'XLMUSDT':  f'{DATA_ROOT_CRYPTO}/XLMUSDT_1d_full.csv',
}

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


SEQ_LEN       = 90
LABEL_LEN     = 45
PRED_LEN      = 30
EPOCHS        = 1000
BATCH_SIZE    = 256
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20

EVAL_HORIZONS = [1, 7, 14, 21, 30]

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
    def __init__(self, data: np.ndarray, time_features: np.ndarray,
                 seq_len: int, label_len: int, pred_len: int):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
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


def get_benchmark_models(n_features: int):
    common = {
        'seq_len': SEQ_LEN, 'label_len': LABEL_LEN, 'pred_len': PRED_LEN,
        'enc_in': n_features, 'dec_in': n_features, 'c_out': n_features,
        'd_model': 384, 'n_heads': 6, 'e_layers': 4, 'd_layers': 1,
        'd_ff': 256, 'dropout': 0.1, 'activation': 'gelu',
        'embed': 'timeF', 'freq': 'd',
    }

    models = [
        {'name': 'Autoformer', 'class': Autoformer, 'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 3, 'moving_avg': 25}},
        {'name': 'FEDformer',  'class': FEDformer,  'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 3, 'moving_avg': 25, 'version': 'Fourier', 'mode_select': 'random', 'modes': 32}},
        {'name': 'Informer',   'class': Informer,   'configs': {**common, 'task_name': 'long_term_forecast', 'factor': 5, 'distil': True, 'output_attention': False}},
    ]
    if HAS_iTRANSFORMER:
        models.append({'name': 'iTransformer', 'class': iTransformer, 'configs': {**common, 'task_name': 'long_term_forecast', 'e_layers': 3, 'factor': 3}})
    return models


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


def run_scenario(scenario: str, file_paths: dict, assets: list):
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-HORIZON BENCHMARK FOR {scenario.upper()}")
    print('='*80)

    df_all = load_and_combine_multi_asset_data(file_paths, assets)
    time_feat = time_features(df_all.index, freq='d').T.astype(np.float32)

    scaled_train, scaled_val, scaled_test, test_data, columns, scalers = scale_and_split_data(df_all)
    n_features = len(columns)

    pre_test_values = np.concatenate((scaled_train, scaled_val))
    pre_test_time = time_feat[:len(pre_test_values)]
    test_time = time_feat[-TEST_DAYS:]

    benchmark_models = get_benchmark_models(n_features)
    global_results = []
    per_asset_results = []
    prefix = f"{scenario}_"

    for bm in benchmark_models:
        model_name = bm['name']
        print(f"\n--- Training {model_name} (PRED_LEN={PRED_LEN}) ---")

        args = argparse.Namespace(**bm['configs'])
        model = bm['class'](args).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train_ds = TimeSeriesForecastDataset(scaled_train, pre_test_time[:len(scaled_train)], SEQ_LEN, LABEL_LEN, PRED_LEN)
        val_ds   = TimeSeriesForecastDataset(scaled_val,   pre_test_time[len(scaled_train):],   SEQ_LEN, LABEL_LEN, PRED_LEN)

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
                dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)
                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])
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
                    dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)
                    out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                    val_loss += LOSS_FN(pred, y[:, -PRED_LEN:, :]).item()
                    n_batches += 1

            val_loss /= n_batches if n_batches > 0 else float('inf')
            train_avg = train_loss / len(train_loader)

            if (epoch + 1) % 50 == 0:
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

        # ─── Correct Test Evaluation ─────────────────────────────────────
        full_values = np.concatenate((pre_test_values, scaled_test))
        full_time = np.concatenate((pre_test_time, test_time))

        test_ds = TimeSeriesForecastDataset(full_values, full_time, SEQ_LEN, LABEL_LEN, PRED_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.cat([y[:, :LABEL_LEN, :], torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)
                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                preds_list.append(pred.cpu().numpy())
                trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

        preds_all = np.concatenate(preds_list, axis=0)   # (many_samples, PRED_LEN, C)
        trues_all = np.concatenate(trues_list, axis=0)

        # === CRITICAL FIX: Take only the last TEST_DAYS predictions ===
        n_test = TEST_DAYS
        preds_test = preds_all[-n_test:]      # Last 365 predictions
        trues_test = trues_all[-n_test:]

        forecast_index = test_data.index[-n_test:]   # Align with last 365 days

        # Metrics
        global_res = {'model': model_name, 'scenario': scenario}
        for h in EVAL_HORIZONS:
            h_idx = h - 1
            err = preds_test[:, h_idx, :] - trues_test[:, h_idx, :]
            global_res[f'h{h}_mse']  = float(np.mean(err ** 2))
            global_res[f'h{h}_rmse'] = float(math.sqrt(global_res[f'h{h}_mse']))
            global_res[f'h{h}_mae']  = float(np.mean(np.abs(err)))

        global_results.append(global_res)

        # Per-asset results
        for asset in assets:
            asset_cols = [f"{asset}_{v}" for v in ['open','high','low','close','volume']]
            idxs = [columns.index(c) for c in asset_cols]
            for h in EVAL_HORIZONS:
                h_idx = h - 1
                err = preds_test[:, h_idx, idxs] - trues_test[:, h_idx, idxs]
                per_asset_results.append({
                    'model': model_name, 'scenario': scenario, 'asset': asset, 'horizon': h,
                    'mse_scaled': float(np.mean(err ** 2)),
                    'rmse_scaled': float(math.sqrt(np.mean(err ** 2))),
                    'mae_scaled': float(np.mean(np.abs(err)))
                })

        print(f"{model_name} → H30 RMSE: {global_res['h30_rmse']:.6f}  MAE: {global_res['h30_mae']:.6f}")

        # ─── Inverse Transform & Save Forecasts ─────────────────────────────
        print(f"Inverse transforming predictions for {model_name}...")
        pred_prices = np.zeros_like(preds_test, dtype=np.float64)

        for i, col in enumerate(columns):
            scaler = scalers[col]
            flat = preds_test[:, :, i].reshape(-1, 1)
            pred_prices[:, :, i] = scaler.inverse_transform(flat).reshape(preds_test.shape[:2])

        # Build forecast DataFrame
        forecast_df = test_data.loc[forecast_index].add_suffix('_true').copy()

        pred_dfs = []
        for h in range(PRED_LEN):
            horizon_cols = [f"{col}_pred_h{h+1}" for col in columns]
            pred_dfs.append(pd.DataFrame(pred_prices[:, h, :],
                                       index=forecast_index,
                                       columns=horizon_cols))

        forecast_df = pd.concat([forecast_df] + pred_dfs, axis=1)
        forecast_df = forecast_df.sort_index(axis=1)

        forecast_path = f"{prefix}forecasts_{model_name}_h{PRED_LEN}.csv"
        forecast_df.to_csv(forecast_path)
        print(f"→ Saved forecasts: {forecast_path} ({len(forecast_df)} rows)")

        torch.cuda.empty_cache()
        gc.collect()

    # Save summary
    pd.DataFrame(global_results).to_csv(f'{prefix}multi_step_transformer_global.csv', index=False)
    pd.DataFrame(per_asset_results).to_csv(f'{prefix}multi_step_transformer_per_asset.csv', index=False)

    print(f"\n{scenario.upper()} multi-horizon benchmark completed.\n")


# ─── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scenarios = [
        ("crypto", CRYPTO_FILES, list(CRYPTO_FILES.keys())),
        ("stock",  STOCK_FILES,  list(STOCK_FILES.keys())),
    ]

    for scenario_name, paths, asset_list in scenarios:
        run_scenario(scenario_name, paths, asset_list)

    print("\nAll multi-horizon benchmarks completed successfully.")