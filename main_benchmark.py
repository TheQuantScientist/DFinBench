"""
Unified Benchmarking Script for Time-Series Forecasting Models
"""
import os
import sys
import math
import gc
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ==============================================================================
#  PATH CONFIGURATION & IMPORTS
# ==============================================================================
os.makedirs("checkpoints", exist_ok=True)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TS_LIB_PATH = os.path.join(PROJECT_DIR, 'Time-Series-Library')
sys.path.insert(0, TS_LIB_PATH)

from utils.timefeatures import time_features

# ====TRAINABLE MODELS====
try:
    from models.TimesNet import Model as TimesNet  
    from models.TimeXer import Model as TimeXer
    from models.TimeMixer import Model as TimeMixer
    from models.ETSformer import Model as ETSformer
    from models.DLinear import Model as DLinear
    from models.Autoformer import Model as Autoformer
    from models.PatchTST import Model as PatchTST
    from models.iTransformer import Model as iTransformer
except ImportError as e:
    print(f"[STRUCTURAL FILE ERROR] Cannot find trainable model: {e}")

# ====FOUNDATION MODELS====
try:
    from models.Chronos import Model as Chronos
    from models.TimesFM import Model as TimesFM
    from models.TimeMoE import Model as TimeMoE
except ImportError as e:
    print(f"[STRUCTURAL FILE ERROR] Cannot find foundation model: {e}")

# ==============================================================================
#  GLOBAL CONFIGURATION (STRICT RULES)
# ==============================================================================
CONFIG = {
    "data_dir": os.path.join(PROJECT_DIR, "dataset", "stock"), 
    "pred_len": 30,                 
    "seq_lens": [60, 90, 120],      
    "train_ratio": 0.8,
    "val_ratio": 0.1,               
    "batch_size": 16,                
    "epochs": 100,  
    "num_workers": 4,                 
    "learning_rate": 1e-4,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
#  DATA PIPELINE (REAL DATASETS & SCALING)
# ==============================================================================
def load_and_combine(data_dir):
    import glob
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        logger.error(f"[!] Error: cannot find any .csv five in {data_dir}")
        sys.exit(1)
        
    stock_files = {os.path.basename(f).split('_')[0]: f for f in all_files}
    assets = list(stock_files.keys())
    
    logger.info(f"[*] Loading {len(assets)} data from dataset/stock...")
    
    data = {}
    for asset, path in stock_files.items():
        df = pd.read_csv(path)
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        
        cols_to_rename = {col: f'{asset}_{col}' for col in df.columns}
        data[asset] = df.rename(columns=cols_to_rename)

    df_all = None
    for asset in tqdm(assets, desc="Data Combining"):
        df_all = data[asset] if df_all is None else df_all.join(data[asset], how='outer')
        
    # Sắp xếp thời gian trước, sau đó chỉ dùng ffill, nếu vẫn còn NaN ở những ngày đầu tiên nhất, fill with zero -> fix look-ahead bias
    df_all = df_all.sort_index().ffill().fillna(0)
    logger.info(f"[*] Done combining! Size of total matrix: {df_all.shape} (R x C)")
    
    return df_all

def prepare_data(df_all):
    total_len = len(df_all)
    train_end = int(total_len * CONFIG["train_ratio"])
    val_end = train_end + int(total_len * CONFIG["val_ratio"])

    train_df = df_all.iloc[:train_end]
    val_df = df_all.iloc[train_end:val_end]
    test_df = df_all.iloc[val_end:]

    columns = df_all.columns.tolist()
    
    train_scaled = np.zeros_like(train_df.values, dtype=np.float32)
    val_scaled = np.zeros_like(val_df.values, dtype=np.float32)
    test_scaled = np.zeros_like(test_df.values, dtype=np.float32)

    logger.info("[*] Scaling data (MinMaxScaler)...")
    for i, col in enumerate(columns):
        scaler = MinMaxScaler()
        train_scaled[:, i] = scaler.fit_transform(train_df[[col]]).squeeze()
        val_scaled[:, i] = scaler.transform(val_df[[col]]).squeeze()
        test_scaled[:, i] = scaler.transform(test_df[[col]]).squeeze()

    logger.info(f"[*] Split - Train: {len(train_scaled)}, Val: {len(val_scaled)}, Test: {len(test_scaled)}")
    return train_scaled, val_scaled, test_scaled, columns


class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_mark, seq_len, pred_len):
        self.data = data
        self.time_mark = time_mark
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Tính toán độ dài đoạn gối đầu cho Decoder (Giống với args.label_len)
        self.label_len = seq_len // 2

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        
        # r_begin lùi lại một khoảng label_len để tạo đoạn gối đầu cho Transformer Decoder
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        
        seq_x = self.data[s_begin : s_end]
        seq_y = self.data[r_begin : r_end]
        seq_x_mark = self.time_mark[s_begin : s_end]
        seq_y_mark = self.time_mark[r_begin : r_end]
        
        return (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark))

# ==============================================================================
#  BENCHMARK ENGINE
# ==============================================================================
def run_benchmark():
    logger.info(f"Start running Benchmark on device: {CONFIG['device']}")
    
    # --- LOAD DATA ---
    df_all = load_and_combine(CONFIG["data_dir"])
    time_feat = time_features(df_all.index, freq='d').T.astype(np.float32)
    train_scaled, val_scaled, test_scaled, columns = prepare_data(df_all)
    n_features = len(columns)

    # Chia tách Time Features cho từng tập
    train_time = time_feat[:len(train_scaled)]
    val_time = time_feat[len(train_scaled) : len(train_scaled)+len(val_scaled)]
    test_time = time_feat[-len(test_scaled):]

    MODEL_REGISTRY = [
        #--- Group 1: Traditional & Transformer Models (needs_training = True) ---
        {'name': 'TimesNet',      'class': TimesNet,      'needs_training': True}, 
        {'name': 'TimeXer',       'class': TimeXer,       'needs_training': True},
        {'name': 'TimeMixer',     'class': TimeMixer,     'needs_training': True},
        {'name': 'ETSformer',     'class': ETSformer,     'needs_training': True},
        {'name': 'DLinear',       'class': DLinear,       'needs_training': True},
        {'name': 'Autoformer',    'class': Autoformer,    'needs_training': True},
        {'name': 'PatchTST',      'class': PatchTST,      'needs_training': True},
        {'name': 'iTransformer',  'class': iTransformer,  'needs_training': True},
        
        #--- Group 2: Foundation Models (needs_training = False) ---
        {'name': 'Chronos',       'class': Chronos,       'needs_training': False},
        {'name': 'TimesFM',       'class': TimesFM,       'needs_training': False},
        {'name': 'TimeMoE',       'class': TimeMoE,       'needs_training': False},
    ]

    #  Main Evaluation Loop
    for seq_len in CONFIG["seq_lens"]:
        logger.info(f"\n{'='*50}\nEVALUATING SEQUENCE LENGTH: {seq_len}\n{'='*50}")
        
        # Tập Train
        train_ds = TimeSeriesDataset(train_scaled, train_time, seq_len, CONFIG["pred_len"])
        train_loader = DataLoader(
            train_ds, 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            num_workers=CONFIG["num_workers"], # multithreading
            pin_memory=True                   
        )
        
        # Tập Validation: Chỉ lấy seq_len điểm dữ liệu cuối của Train để làm Lookback
        val_input = np.concatenate((train_scaled[-seq_len:], val_scaled), axis=0)
        val_input_time = np.concatenate((train_time[-seq_len:], val_time), axis=0)
        val_ds = TimeSeriesDataset(val_input, val_input_time, seq_len, CONFIG["pred_len"])
        val_loader = DataLoader(
            val_ds, 
            batch_size=CONFIG["batch_size"], 
            shuffle=False, 
            num_workers=CONFIG["num_workers"], 
            pin_memory=True
        )

        # Tập Test: Chỉ lấy seq_len điểm dữ liệu cuối của Validation để làm Lookback
        test_input = np.concatenate((val_scaled[-seq_len:], test_scaled), axis=0)
        test_input_time = np.concatenate((val_time[-seq_len:], test_time), axis=0)
        test_ds = TimeSeriesDataset(test_input, test_input_time, seq_len, CONFIG["pred_len"])
        test_loader = DataLoader(
            test_ds, 
            batch_size=CONFIG["batch_size"], 
            shuffle=False, 
            num_workers=CONFIG["num_workers"], 
            pin_memory=True
        )

        for model_info in MODEL_REGISTRY:
            model_name = model_info['name']
            needs_training = model_info['needs_training']
            logger.info(f"--> Initializing {model_name} (Trainable: {needs_training})")

            # ==========================================================
            # --- CONFIG ARGUMENTS PRODUCTION (FULL POWER FOR SERVER) ---
            # ==========================================================
            class Args: pass
            args = Args()
            
            #  Basic config
            args.seq_len = seq_len
            args.pred_len = CONFIG["pred_len"]
            args.label_len = seq_len // 2
            args.enc_in = args.dec_in = args.c_out = n_features
            args.task_name = 'long_term_forecast' if needs_training else 'zero_shot_forecast'
            args.features = 'M'  # Mixer: Multivariate
            
            #  Neuron network
            args.moving_avg = 25
            args.d_model = 256         
            args.n_heads = 8
            args.e_layers = 2        
            args.d_layers = 1
            args.d_ff = 512
            args.factor = 1
            args.dropout = 0.1
            args.fc_dropout = 0.1
            args.head_dropout = 0.1
            args.activation = 'gelu'
            args.output_attention = False
            args.embed = 'timeF'
            args.freq = 'd'
            
            # Specified config for TimesNet / TimeMixer
            args.top_k = 5                
            args.num_kernels = 6          
            args.down_sampling_layers = 2 
            args.down_sampling_window = 2 
            args.down_sampling_method = 'avg'
            
            # Specified config for PatchTST / iTransformer / ETSformer
            args.patch_len = 16
            args.stride = 8
            args.channel_independence = 1
            args.decomp_method = 'moving_avg'
            args.use_norm = 1
            args.class_strategy = 'projection'
            args.p_hidden_dims = [128, 128] 
            args.p_hidden_layers = 2
            args.use_future_temporal_feature = 0
            # ==========================================================
            
            if model_name == 'ETSformer':
                args.d_layers = args.e_layers  # Fix layers 

            try:
                model = model_info['class'](args).to(CONFIG['device'])
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue

            # --- TRAINING PHASE ---
            if needs_training:
                logger.info(f"Training {model_name}...")
                
                optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
                criterion = nn.MSELoss()
                
                best_loss = float('inf')

                patience = 5  
                early_stop_counter = 0

                for epoch in range(CONFIG["epochs"]):
                    model.train()
                    train_loss = 0
                    for x, y, x_mark, y_mark in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
                        x, y, x_mark, y_mark = x.to(CONFIG['device']), y.to(CONFIG['device']), x_mark.to(CONFIG['device']), y_mark.to(CONFIG['device'])
                        
                        dec_inp = torch.zeros([x.shape[0], CONFIG["pred_len"], x.shape[-1]]).to(CONFIG['device'])
                        dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1)
                        
                        out = model(x, x_mark, dec_inp, y_mark)
                        loss = criterion(out[:, -CONFIG["pred_len"]:, :], y[:, -CONFIG["pred_len"]:, :])
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # Add VAL PHASE: Evaluate model on Validation set to store best weight of model 
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for x, y, x_mark, y_mark in val_loader:
                            x, y, x_mark, y_mark = x.to(CONFIG['device']), y.to(CONFIG['device']), x_mark.to(CONFIG['device']), y_mark.to(CONFIG['device'])
                            dec_inp = torch.zeros([x.shape[0], CONFIG["pred_len"], x.shape[-1]]).to(CONFIG['device'])
                            dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1)
                            
                            out = model(x, x_mark, dec_inp, y_mark)
                            loss = criterion(out[:, -CONFIG["pred_len"]:, :], y[:, -CONFIG["pred_len"]:, :])
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), f"checkpoints/{model_name}_seq{seq_len}_best.pth")
                        logger.info(f"Epoch {epoch+1} -> Val Loss improved to {best_loss:.4f}. Saved checkpoint!")
                        early_stop_counter = 0 
                    else:
                        early_stop_counter += 1
                        logger.info(f"Epoch {epoch+1} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss:.4f} (No improvement: {early_stop_counter}/{patience})")
                        
                        if early_stop_counter >= patience:
                            logger.info(f"🛑 EARLY STOPPING TRIGGERED {epoch+1}!")
                            break

                    logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")


            # --- TESTING PHASE ---
            logger.info(f"Testing {model_name} on unseen Test set...")
            if needs_training:
                # Load the model with best Validation Loss
                model.load_state_dict(torch.load(f"checkpoints/{model_name}_seq{seq_len}_best.pth"))
            
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y, x_mark, y_mark in tqdm(test_loader, desc=f"Inference {model_name}"):
                    x, y, x_mark, y_mark = x.to(CONFIG['device']), y.to(CONFIG['device']), x_mark.to(CONFIG['device']), y_mark.to(CONFIG['device'])
                    
                    dec_inp = torch.zeros([x.shape[0], CONFIG["pred_len"], x.shape[-1]]).to(CONFIG['device'])
                    dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1)
                    
                    out = model(x, x_mark, dec_inp, y_mark)
                    preds.append(out[:, -CONFIG["pred_len"]:, :].cpu().numpy())
                    trues.append(y[:, -CONFIG["pred_len"]:, :].cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            
            # Calculate Metrics
            mse = np.mean((preds - trues) ** 2)
            mae = np.mean(np.abs(preds - trues))
            logger.info(f"*** {model_name} RESULTS (Seq: {seq_len}) | RMSE: {math.sqrt(mse):.4f} | MAE: {mae:.4f} ***\n")

            # Memory Cleanup
            del model
            del optimizer
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Benchmarking Completed Successfully!")

if __name__ == "__main__":
    run_benchmark()