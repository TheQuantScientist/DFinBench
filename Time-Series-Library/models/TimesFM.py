import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timesfm
import inspect

# ====================================================================
# [DFinBench] SELF-HEALING PATCH V2.0:
# Quét chữ ký hàm (signature) và tự động chặn TẤT CẢ tham số rác 
# (proxies, resume_download, token...) từ HuggingFace
# ====================================================================
original_init = timesfm.TimesFM_2p5_200M_torch.__init__
valid_params = inspect.signature(original_init).parameters

def patched_init(self, *args, **kwargs):
    # Chỉ giữ lại những key mà hàm gốc của Google thực sự có khai báo
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    original_init(self, *args, **filtered_kwargs)

timesfm.TimesFM_2p5_200M_torch.__init__ = patched_init
# ====================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", 
            torch_compile=True, 
            device=current_device
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # [DFinBench Integration Note]: 
        # TimesFM is natively univariate. To benchmark on multivariate data efficiently, 
        # we flatten the [Batch, Seq, Features] tensor into [Batch * Features, Seq] 
        # to leverage GPU parallelization. This wrapper optimization preserves mathematical 
        # equivalence with looping over features, but prevents CPU bottlenecks.
        
        # x_enc shape: [Batch, Seq_len, Features]
        B, S, F = x_enc.shape
        
        # Optimize: Flatten Batch and Feature to run parallelly
        # Transform to [Batch, Feature, Seq_len] -> flatten to [Batch * Feature, Seq_len]
        x_flat = x_enc.transpose(1, 2).reshape(B * F, S)
        
        # Bring into model only one
        output, _ = self.model.forecast(
            horizon=self.pred_len,
            inputs=x_flat.cpu().numpy().tolist()
        )
        
        # output shape currently is [Batch * Feature, pred_len]
        dec_out = torch.Tensor(output).to(x_enc.device)
        
        # Reshape to [Batch, Feature, pred_len]
        dec_out = dec_out.view(B, F, self.pred_len)
        # Normal form to benchmark: [Batch, pred_len, Feature] 
        dec_out = dec_out.transpose(1, 2)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None