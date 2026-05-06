import os
import sys
import argparse
import warnings
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# Add Time-Series-Library path
sys.path.append('/home/nckh2/qa/Time-Series-Library')
from utils.timefeatures import time_features

from models.DLinear import Model as DLinear

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT_STOCK = r"D:\DFBENCH\dataset\stock"

STOCK_FILES = {
    # Tech
    'AAPL':  f'{DATA_ROOT_STOCK}/AAPL_1d_full.csv',
    'MSFT':  f'{DATA_ROOT_STOCK}/MSFT_1d_full.csv',
    'GOOGL': f'{DATA_ROOT_STOCK}/GOOGL_1d_full.csv',
    'AMZN':  f'{DATA_ROOT_STOCK}/AMZN_1d_full.csv',
    'NVDA':  f'{DATA_ROOT_STOCK}/NVDA_1d_full.csv',
    'META':  f'{DATA_ROOT_STOCK}/META_1d_full.csv',
    'TSM':   f'{DATA_ROOT_STOCK}/TSM_1d_full.csv',
    'AVGO':  f'{DATA_ROOT_STOCK}/AVGO_1d_full.csv',
    'ORCL':  f'{DATA_ROOT_STOCK}/ORCL_1d_full.csv',
    'CSCO':  f'{DATA_ROOT_STOCK}/CSCO_1d_full.csv',

    # Healthcare
    'JNJ':   f'{DATA_ROOT_STOCK}/JNJ_1d_full.csv',
    'UNH':   f'{DATA_ROOT_STOCK}/UNH_1d_full.csv',
    'LLY':   f'{DATA_ROOT_STOCK}/LLY_1d_full.csv',
    'NVO':   f'{DATA_ROOT_STOCK}/NVO_1d_full.csv',
    'MRK':   f'{DATA_ROOT_STOCK}/MRK_1d_full.csv',
    'ABBV':  f'{DATA_ROOT_STOCK}/ABBV_1d_full.csv',
    'TMO':   f'{DATA_ROOT_STOCK}/TMO_1d_full.csv',
    'DHR':   f'{DATA_ROOT_STOCK}/DHR_1d_full.csv',
    'PFE':   f'{DATA_ROOT_STOCK}/PFE_1d_full.csv',
    'AZN':   f'{DATA_ROOT_STOCK}/AZN_1d_full.csv',

    # Finance
    'JPM':   f'{DATA_ROOT_STOCK}/JPM_1d_full.csv',
    'BAC':   f'{DATA_ROOT_STOCK}/BAC_1d_full.csv',
    'V':     f'{DATA_ROOT_STOCK}/V_1d_full.csv',
    'MA':    f'{DATA_ROOT_STOCK}/MA_1d_full.csv',
    'WFC':   f'{DATA_ROOT_STOCK}/WFC_1d_full.csv',
    'MS':    f'{DATA_ROOT_STOCK}/MS_1d_full.csv',
    'GS':    f'{DATA_ROOT_STOCK}/GS_1d_full.csv',
    'C':     f'{DATA_ROOT_STOCK}/C_1d_full.csv',
    'AXP':   f'{DATA_ROOT_STOCK}/AXP_1d_full.csv',
    'SCHW':  f'{DATA_ROOT_STOCK}/SCHW_1d_full.csv',

    # Food & Beverage
    'KO':    f'{DATA_ROOT_STOCK}/KO_1d_full.csv',
    'PEP':   f'{DATA_ROOT_STOCK}/PEP_1d_full.csv',
    'MCD':   f'{DATA_ROOT_STOCK}/MCD_1d_full.csv',
    'SBUX':  f'{DATA_ROOT_STOCK}/SBUX_1d_full.csv',
    'KDP':   f'{DATA_ROOT_STOCK}/KDP_1d_full.csv',
    'MDLZ':  f'{DATA_ROOT_STOCK}/MDLZ_1d_full.csv',
    'HSY':   f'{DATA_ROOT_STOCK}/HSY_1d_full.csv',
    'GIS':   f'{DATA_ROOT_STOCK}/GIS_1d_full.csv',
    'NSRGY': f'{DATA_ROOT_STOCK}/NSRGY_1d_full.csv',
    'TSN':   f'{DATA_ROOT_STOCK}/TSN_1d_full.csv',
}
SECTORS = {
    "tech": [
        'AAPL','MSFT','GOOGL','AMZN','NVDA',
        'META','TSM','AVGO','ORCL','CSCO'
    ],

    "healthcare": [
        'JNJ','UNH','LLY','NVO','MRK',
        'ABBV','TMO','DHR','PFE','AZN'
    ],

    "finance": [
        'JPM','BAC','V','MA','WFC',
        'MS','GS','C','AXP','SCHW'
    ],

    "food and beverage": [
        'KO','PEP','MCD','SBUX','KDP',
        'MDLZ','HSY','GIS','NSRGY','TSN'
    ]
}

SEQ_LENS      = [30, 60, 90]
LABEL_LEN     = 30
PRED_LEN      = 30
EPOCHS        = 1
BATCH_SIZE    = 64
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 30
MIN_DELTA     = 2e-9
TEST_DAYS     = 365

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
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

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


def get_benchmark_models(n_features: int, seq_len: int):
    models = [
        {
            'name': 'DLinear',
            'class': DLinear,
            'configs': {
                'seq_len': seq_len,   
                'label_len': LABEL_LEN,
                'pred_len': PRED_LEN,
                'enc_in': n_features,
                'dec_in': n_features,
                'c_out': n_features,
                'individual': False,
                'moving_avg': 25,
                'task_name': 'long_term_forecast'
            }
        }
    ]
    return models
def scale_and_split_data(df_all: pd.DataFrame):
    n = len(df_all)

    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    train_data = df_all.iloc[:train_end]
    val_data   = df_all.iloc[train_end:val_end]
    test_data  = df_all.iloc[val_end:]

    columns = df_all.columns.tolist()

    # Fit scaler ONLY trên train (đúng chuẩn ML)
    scalers = {
        col: MinMaxScaler().fit(train_data[[col]])
        for col in columns
    }

    def apply_scaling(df):
        arr = np.hstack([
            scalers[col].transform(df[[col]])
            for col in columns
        ])
        return pd.DataFrame(arr, index=df.index, columns=columns)

    return (
        apply_scaling(train_data).values.astype(np.float32),
        apply_scaling(val_data).values.astype(np.float32),
        apply_scaling(test_data).values.astype(np.float32),
        test_data,
        columns,
        scalers
    )

def run_scenario(scenario: str, file_paths: dict, assets: list, seq_len: int):
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-HORIZON BENCHMARK FOR {scenario.upper()}")
    print('='*80)

    df_all = load_and_combine_multi_asset_data(file_paths, assets)
    time_feat = time_features(df_all.index, freq='d').T.astype(np.float32)

    scaled_train, scaled_val, scaled_test, test_data, columns, scalers = scale_and_split_data(df_all)
    n_features = len(columns)

    # ===== FULL DATA =====
    full_values = np.concatenate((scaled_train, scaled_val, scaled_test))
    full_time = time_feat[:len(full_values)]

    # ===== TRAIN DATA =====
    train_ds = TimeSeriesForecastDataset(
        scaled_train,
        full_time[:len(scaled_train)],
        seq_len,   
        LABEL_LEN,
        PRED_LEN
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # ===== FULL DATA (for inference) =====
    full_ds = TimeSeriesForecastDataset(
        full_values,
        full_time,
        seq_len,  
        LABEL_LEN,
        PRED_LEN
    )

    full_loader = DataLoader(
        full_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    benchmark_models = get_benchmark_models(n_features, seq_len)
    global_results = []
    per_asset_results = []
    prefix = f"{scenario}_seq{seq_len}_"

    for bm in benchmark_models:
        model_name = bm['name']
        print(f"\n--- Training {model_name} (PRED_LEN={PRED_LEN}) ---")

        args = argparse.Namespace(**bm['configs'])
        model = bm['class'](args).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        # ===== TRAIN LOOP =====
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]

                dec_inp = torch.cat(
                    [y[:, :LABEL_LEN, :],
                     torch.zeros_like(y[:, -PRED_LEN:, :])],
                    dim=1
                )

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :]
                loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_avg = train_loss / len(train_loader)

            if (epoch + 1) % 50 == 0:
                print(f"[{epoch+1:4d}] train: {train_avg:.7f}")

            # Early stopping dùng train loss
            if train_avg < best_loss - MIN_DELTA:
                best_loss = train_avg
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ===== INFERENCE (FULL DATA) =====
        model.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for batch in full_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]

                dec_inp = torch.cat(
                    [y[:, :LABEL_LEN, :],
                     torch.zeros_like(y[:, -PRED_LEN:, :])],
                    dim=1
                )

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :]

                preds_list.append(pred.cpu().numpy())
                trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

        preds_all = np.concatenate(preds_list, axis=0)
        trues_all = np.concatenate(trues_list, axis=0)

        # ===== CHỈ LẤY TEST =====
        n_test = min(TEST_DAYS, len(test_data))

        preds_test = preds_all[-n_test:]
        trues_test = trues_all[-n_test:]
        forecast_index = test_data.index[-n_test:]

        # ===== METRICS =====
        global_res = {'model': model_name, 'scenario': scenario}

        for h in EVAL_HORIZONS:
            h_idx = h - 1
            err = preds_test[:, h_idx, :] - trues_test[:, h_idx, :]
            global_res[f'h{h}_mse']  = float(np.mean(err ** 2))
            global_res[f'h{h}_rmse'] = float(math.sqrt(global_res[f'h{h}_mse']))
            global_res[f'h{h}_mae']  = float(np.mean(np.abs(err)))

        global_results.append(global_res)

        # ===== PER-ASSET =====
        for asset in assets:
            asset_cols = [f"{asset}_{v}" for v in ['open','high','low','close','volume']]
            idxs = [columns.index(c) for c in asset_cols]

            for h in EVAL_HORIZONS:
                h_idx = h - 1
                err = preds_test[:, h_idx, idxs] - trues_test[:, h_idx, idxs]

                per_asset_results.append({
                    'model': model_name,
                    'scenario': scenario,
                    'asset': asset,
                    'horizon': h,
                    'mse_scaled': float(np.mean(err ** 2)),
                    'rmse_scaled': float(math.sqrt(np.mean(err ** 2))),
                    'mae_scaled': float(np.mean(np.abs(err)))
                })

        print(f"{model_name} → H30 RMSE: {global_res['h30_rmse']:.6f}  MAE: {global_res['h30_mae']:.6f}")

        # ===== INVERSE SCALE =====
        print(f"Inverse transforming predictions for {model_name}...")
        pred_prices = np.zeros_like(preds_test, dtype=np.float64)

        for i, col in enumerate(columns):
            scaler = scalers[col]
            flat = preds_test[:, :, i].reshape(-1, 1)
            pred_prices[:, :, i] = scaler.inverse_transform(flat).reshape(preds_test.shape[:2])

        # ===== SAVE =====
        forecast_df = test_data.copy().add_suffix('_true')

        pred_dfs = []
        for h in range(PRED_LEN):
            horizon_cols = [f"{col}_pred_h{h+1}" for col in columns]
            pred_dfs.append(pd.DataFrame(
                pred_prices[:, h, :],
                index=forecast_index,
                columns=horizon_cols
            ))

        forecast_df = pd.concat([forecast_df] + pred_dfs, axis=1)
        forecast_df = forecast_df.sort_index(axis=1)

        forecast_path = f"{prefix}forecasts_{model_name}_h{PRED_LEN}.csv"
        forecast_df.to_csv(forecast_path)

        print(f"→ Saved forecasts: {forecast_path} ({len(forecast_df)} rows)")


    pd.DataFrame(global_results).to_csv(f'{prefix}multi_step_transformer_global.csv', index=False)
    pd.DataFrame(per_asset_results).to_csv(f'{prefix}multi_step_transformer_per_asset.csv', index=False)

    print(f"\n{scenario.upper()} multi-horizon benchmark completed.\n")

#---Main
if __name__ == "__main__":
    ALL_STOCKS = []
    for sector_assets in SECTORS.values():
        ALL_STOCKS.extend(sector_assets)

    for seq in SEQ_LENS:
        print(f"\n========== SEQ_LEN = {seq} ==========")

        run_scenario(
            "all_stocks",
            STOCK_FILES,
            ALL_STOCKS,
            seq_len=seq   
        )