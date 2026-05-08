import os
import sys
import argparse
import math
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# =====================================================================
# Path block
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
ts_lib_path = os.path.join(current_dir, '../Time-Series-Library')


# Insert path
sys.path.insert(0, ts_lib_path)

# =====================================================================

from utils.timefeatures import time_features

# Import 2 Foundation Models
from models.TimesFM import Model as TimesFM
from models.TimeMoE import Model as TimeMoE

# --- 1. CONFIGURATION ---
DATA_DIR = "./dataset/stock"
PRED_LEN = 30
SEQ_LENS = [60, 90, 120]
BATCH_SIZE = 16  # Zero-shot
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATA PREPARATION ---
stock_files = {f.split('_')[0]: os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')}
stocks_list = list(stock_files.keys())

def load_and_combine(file_paths, assets):
    data = {}
    for asset, path in file_paths.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[asset] = df

    df_all = None
    for asset in assets:
        renamed = data[asset].rename(columns={
            'open': f'{asset}_open', 'high': f'{asset}_high', 'low': f'{asset}_low',
            'close': f'{asset}_close', 'volume': f'{asset}_volume'
        })
        df_all = renamed if df_all is None else df_all.join(renamed, how='outer')
    return df_all.ffill().bfill().sort_index()

# 80-10-10
def scale_and_split_data(df_all: pd.DataFrame):
    total_len = len(df_all)
    test_size = int(total_len * 0.10)
    val_size = int(total_len * 0.10)
    train_size = total_len - val_size - test_size

    train_data = df_all.iloc[:train_size]
    val_data = df_all.iloc[train_size:train_size+val_size]
    test_data = df_all.iloc[-test_size:]

    columns = df_all.columns.tolist()
    scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

    def apply_scaling(df):
        arr = np.hstack([scalers[col].transform(df[[col]]) for col in columns])
        return pd.DataFrame(arr, index=df.index, columns=columns)

    return (apply_scaling(train_data).values.astype(np.float32),
            apply_scaling(val_data).values.astype(np.float32),
            apply_scaling(test_data).values.astype(np.float32),
            test_size)

class TSDataset(Dataset):
    def __init__(self, data, time_features, seq_len, pred_len):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_end = idx + self.seq_len
        seq_x = self.data[idx : s_end]
        seq_y = self.data[s_end : s_end + self.pred_len]
        seq_x_mark = self.time_features[idx : s_end]
        seq_y_mark = self.time_features[s_end : s_end + self.pred_len]
        return (torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float(),
                torch.from_numpy(seq_x_mark).float(), torch.from_numpy(seq_y_mark).float())

# --- ZERO-SHOT BENCHMARK LOOP ---
if __name__ == "__main__":
    print("[*] Loading data...")
    df_all = load_and_combine(stock_files, stocks_list)
    time_feat = time_features(df_all.index, freq='d').T.astype(np.float32)
    
    scaled_train, scaled_val, scaled_test, test_size = scale_and_split_data(df_all)
    n_features = len(df_all.columns)

    # Nối val và test để đảm bảo mô hình có đủ "Lookback window" cho những ngày đầu tiên của tập Test
    pre_test_values = np.concatenate((scaled_train, scaled_val))
    pre_test_time = time_feat[:len(pre_test_values)]
    
    full_values = np.concatenate((pre_test_values, scaled_test))
    full_time = np.concatenate((pre_test_time, time_feat[-test_size:]))

    for current_seq_len in SEQ_LENS:
        print(f"\n{'='*60}")
        print(f"Start with SEQ_LEN = {current_seq_len} days")
        print(f"{'='*60}")

        # Config for Foundation Models
        foundation_cfg = {
            'seq_len': current_seq_len, 
            'label_len': current_seq_len // 2, 
            'pred_len': PRED_LEN,
            'enc_in': n_features, 'dec_in': n_features, 'c_out': n_features,
            'task_name': 'zero_shot_forecast' # Key for zero-shot
        }

        models_to_run = [
            # {'name': 'TimesFM', 'class': TimesFM},
            {'name': 'TimeMoE', 'class': TimeMoE}
        ]

        for bm in models_to_run:
            model_name = bm['name']
            print(f"\n>> Initializing model: {model_name}...")
            
            args = argparse.Namespace(**foundation_cfg)
            try:
                # Load pre-trained weights from Hugging Face
                model = bm['class'](args).to(DEVICE)
                model.eval()
            except Exception as e:
                print(f"[!] Error when loading {model_name}: {e}")
                continue

            print(f"[*] Running Zero-shot Inference on Test set...")
            test_ds = TSDataset(full_values, full_time, current_seq_len, PRED_LEN)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            preds_list, trues_list = [], []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Tiến độ {model_name}"):
                    x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                    
                    # Masking
                    dec_inp = torch.zeros([x.shape[0], PRED_LEN, x.shape[-1]]).to(DEVICE)
                    dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1)

                    # Call Model
                    out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :]
                    
                    preds_list.append(pred.cpu().numpy())
                    trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

            preds_all = np.concatenate(preds_list, axis=0)
            trues_all = np.concatenate(trues_list, axis=0)

            # Metric
            err = preds_all - trues_all
            mse = float(np.mean(err ** 2))
            rmse = float(math.sqrt(mse))
            mae = float(np.mean(np.abs(err)))

            print(f"Result {model_name} (Seq: {current_seq_len} -> Pred: {PRED_LEN}):")
            print(f"   - RMSE: {rmse:.5f}")
            print(f"   - MAE : {mae:.5f}")

            # Delete model for working with next model
            del model
            torch.cuda.empty_cache()
            gc.collect()

    print("\nEnding Benchmarking Zero-Shot!")