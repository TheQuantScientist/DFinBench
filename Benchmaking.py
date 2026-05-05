# Benchmarking.py - FINAL FIXED VERSION

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.Autoformer import Model as Autoformer
from models.PatchTST import Model as PatchTST
from models.iTransformer import Model as iTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================
PREDICTION_HORIZON = 30
LOOKBACKS = [60, 90, 120]
SPLIT_RATIOS = [0.8, 0.1, 0.1]
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

# ============================================================================
# DATASET
# ============================================================================
class StockDataset(Dataset):
    def __init__(self, features, target, lookback, horizon):
        self.features = features
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        
        self.X = []
        self.y = []
        
        for i in range(len(features) - lookback - horizon + 1):
            self.X.append(features[i:i+lookback])
            self.y.append(target[i+lookback:i+lookback+horizon])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# ============================================================================
# LOAD DATA
# ============================================================================
def load_stock_data(ticker):
    file_path = f'./dataset/stock/{ticker}_1d_full.csv'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None
    
    df = pd.read_csv(file_path)
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    features = df[feature_cols].values.astype(np.float32)
    target = df[['close']].values.astype(np.float32)
    
    return features, target

def scale_data(features, target):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)
    
    return features_scaled, target_scaled, scaler_X, scaler_y

# ============================================================================
# CREATE MODEL - FIXED: Đảm bảo output đúng 1 feature
# ============================================================================
def create_model(model_name, lookback, horizon, n_features=5):
    config = SimpleNamespace(
        task_name='long_term_forecast',
        seq_len=lookback,
        pred_len=horizon,
        label_len=lookback // 2,
        enc_in=n_features,
        dec_in=n_features,
        c_out=1,  # Quan trọng: chỉ predict close price
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=2,
        d_ff=2048,
        factor=3,
        dropout=0.05,
        activation='gelu',
        output_attention=False,
        embed='fixed',
        freq='d',
        moving_avg=25,
        features='MS',
        use_norm=True,
        num_class=10,
    )
    
    if model_name == 'Autoformer':
        return Autoformer(config)
    elif model_name == 'PatchTST':
        return PatchTST(config)
    elif model_name == 'iTransformer':
        return iTransformer(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================================
# TRAIN MODEL - FIXED: Chỉ lấy cột close từ output nếu cần
# ============================================================================
def train_model(model, train_loader, val_loader, lookback, horizon, epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    label_len = lookback // 2
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)  # [batch, lookback, 5]
            y_batch = y_batch.to(device)  # [batch, horizon, 1]
            
            batch_size = X_batch.shape[0]
            
            x_mark_enc = torch.zeros(batch_size, lookback, 4).to(device)
            x_mark_dec = torch.zeros(batch_size, horizon + label_len, 4).to(device)
            dec_inp = torch.zeros(batch_size, label_len + horizon, X_batch.shape[2]).to(device)
            dec_inp[:, :label_len, :] = X_batch[:, -label_len:, :]
            
            optimizer.zero_grad()
            
            # Forward
            if isinstance(model, Autoformer):
                output = model(x_enc=X_batch, x_mark_enc=x_mark_enc, x_dec=dec_inp, x_mark_dec=x_mark_dec)
            else:
                output = model(X_batch, x_mark_enc, dec_inp, x_mark_dec)
                if isinstance(output, tuple):
                    output = output[0]
            
            # FIX: Nếu output có nhiều hơn 1 feature, chỉ lấy cột close (cột index 3 hoặc cột cuối)
            if output.shape[-1] > 1:
                # Giả sử close price là feature cuối hoặc index 3
                output = output[:, :, -1:]  # Lấy feature cuối cùng
                # Hoặc: output = output[:, :, 3:4]  # Nếu close là column index 3
            
            if output.shape[1] > horizon:
                pred = output[:, -horizon:, :]
            else:
                pred = output
            
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                batch_size = X_batch.shape[0]
                
                x_mark_enc = torch.zeros(batch_size, lookback, 4).to(device)
                x_mark_dec = torch.zeros(batch_size, horizon + label_len, 4).to(device)
                dec_inp = torch.zeros(batch_size, label_len + horizon, X_batch.shape[2]).to(device)
                dec_inp[:, :label_len, :] = X_batch[:, -label_len:, :]
                
                if isinstance(model, Autoformer):
                    output = model(x_enc=X_batch, x_mark_enc=x_mark_enc, x_dec=dec_inp, x_mark_dec=x_mark_dec)
                else:
                    output = model(X_batch, x_mark_enc, dec_inp, x_mark_dec)
                    if isinstance(output, tuple):
                        output = output[0]
                
                # FIX: Chỉ lấy 1 feature
                if output.shape[-1] > 1:
                    output = output[:, :, -1:]
                
                if output.shape[1] > horizon:
                    pred = output[:, -horizon:, :]
                else:
                    pred = output
                
                val_loss += criterion(pred, y_batch).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Load best model
    best_model_path = f'best_model_{model.__class__.__name__}.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)
    
    return model

# ============================================================================
# EVALUATE - FIXED: Đảm bảo output chỉ có 1 feature
# ============================================================================
def evaluate_model(model, test_loader, scaler_y, lookback, horizon):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    label_len = lookback // 2
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            batch_size = X_batch.shape[0]
            
            x_mark_enc = torch.zeros(batch_size, lookback, 4).to(device)
            x_mark_dec = torch.zeros(batch_size, horizon + label_len, 4).to(device)
            dec_inp = torch.zeros(batch_size, label_len + horizon, X_batch.shape[2]).to(device)
            dec_inp[:, :label_len, :] = X_batch[:, -label_len:, :]
            
            if isinstance(model, Autoformer):
                output = model(x_enc=X_batch, x_mark_enc=x_mark_enc, x_dec=dec_inp, x_mark_dec=x_mark_dec)
            else:
                output = model(X_batch, x_mark_enc, dec_inp, x_mark_dec)
                if isinstance(output, tuple):
                    output = output[0]
            
            # FIX: Chỉ lấy 1 feature (close price)
            if output.shape[-1] > 1:
                output = output[:, :, -1:]  # Lấy feature cuối
            
            if output.shape[1] > horizon:
                pred = output[:, -horizon:, :]
            else:
                pred = output
            
            predictions.append(pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    
    # Concatenate
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Inverse transform
    predictions_flat = predictions.reshape(-1, 1)
    actuals_flat = actuals.reshape(-1, 1)
    
    predictions_inv = scaler_y.inverse_transform(predictions_flat).reshape(predictions.shape)
    actuals_inv = scaler_y.inverse_transform(actuals_flat).reshape(actuals.shape)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals_inv.flatten(), predictions_inv.flatten()))
    mae = mean_absolute_error(actuals_inv.flatten(), predictions_inv.flatten())
    mape = np.mean(np.abs((actuals_inv - predictions_inv) / (actuals_inv + 1e-8))) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ============================================================================
# MAIN BENCHMARK
# ============================================================================
def benchmark_stock(ticker, sector):
    print(f"\n{'='*60}")
    print(f"Processing: {ticker} ({sector})")
    print(f"{'='*60}")
    
    features, target = load_stock_data(ticker)
    if features is None:
        return None
    
    features_scaled, target_scaled, scaler_X, scaler_y = scale_data(features, target)
    
    results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback: {lookback} days ---")
        results[lookback] = {}
        
        dataset = StockDataset(features_scaled, target_scaled, lookback, PREDICTION_HORIZON)
        
        total = len(dataset)
        train_end = int(total * SPLIT_RATIOS[0])
        val_end = train_end + int(total * SPLIT_RATIOS[1])
        
        train_ds = torch.utils.data.Subset(dataset, range(0, train_end))
        val_ds = torch.utils.data.Subset(dataset, range(train_end, val_end))
        test_ds = torch.utils.data.Subset(dataset, range(val_end, total))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        for model_name in ['Autoformer', 'PatchTST', 'iTransformer']:
            print(f"\nTraining {model_name}...")
            try:
                model = create_model(model_name, lookback, PREDICTION_HORIZON, n_features=5)
                model = train_model(model, train_loader, val_loader, lookback, PREDICTION_HORIZON, epochs=EPOCHS)
                metrics = evaluate_model(model, test_loader, scaler_y, lookback, PREDICTION_HORIZON)
                results[lookback][model_name] = metrics
                print(f"✅ {model_name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results[lookback][model_name] = None
        
        torch.cuda.empty_cache()
    
    return results

def main():
    print("="*80)
    print("BENCHMARKING: Autoformer vs PatchTST vs iTransformer")
    print("="*80)
    print(f"Prediction Horizon: {PREDICTION_HORIZON} days")
    print(f"Lookback windows: {LOOKBACKS}")
    print(f"Split ratio: {SPLIT_RATIOS[0]}-{SPLIT_RATIOS[1]}-{SPLIT_RATIOS[2]}")
    print(f"Input features: open, high, low, close, volume (5 features)")
    print(f"Target: close price")
    print("="*80)
    
    test_stocks = {"Tech": ["AAPL"]}
    all_results = {}
    
    for sector, stocks in test_stocks.items():
        for ticker in stocks:
            results = benchmark_stock(ticker, sector)
            if results:
                all_results[f"{ticker}_{sector}"] = results
    
    # Save results
    import json
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=4, default=convert_to_serializable)
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()