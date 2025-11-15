import torch
import torch.nn as nn

class FlowTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, max_len=64, proj_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over sequence dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, seq_mask=None):
        # x: (B, L, input_dim)
        B, L, _ = x.shape
        x = self.input_proj(x) + self.pos_emb[:, :L, :]
        z = self.transformer(x, src_key_padding_mask=seq_mask)  # (B, L, d_model)
        z = z.permute(0,2,1)  # (B, d_model, L)
        z = self.pool(z).squeeze(-1)  # (B, d_model)
        return self.proj(z)  # (B, proj_dim)
