import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from chronos import BaseChronosPipeline

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cuda",  
            torch_dtype=torch.bfloat16,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # [DFinBench Integration Note]:
        # Chronos is natively univariate. While vectorizing/flattening across features 
        # (like we did in TimesFM) would speed up inference, we intentionally use a loop here. 
        # Passing [Batch * Features, Seq] into Chronos's massive LLM attention mechanism 
        # simultaneously would immediately cause a CUDA Out-of-Memory (OOM) error on the server. 
        # This loop trades off speed for strict memory safety, ensuring stable benchmarking.
        
        outputs = []
        # Safe loop for avoiding overflow of VRAM
        for i in range(x_enc.shape[-1]):
            # Sinh ra dự đoán cho feature thứ i
            # Generate prediction for feature ith
            # Default Output of chronos is [Batch, Num_Samples, Pred_len]
            samples = self.model.predict(x_enc[..., i], prediction_length=self.pred_len)
            
            # Lấy trung bình của các mẫu để ra point forecast [Batch, Pred_len]
            # Get mean of samples to get point forecast [Batch, Pred_len]
            point_forecast = samples.mean(dim=1)
            outputs.append(point_forecast)
            
        # Combinning to [Batch, Pred_len, Features]
        dec_out = torch.stack(outputs, dim=-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None