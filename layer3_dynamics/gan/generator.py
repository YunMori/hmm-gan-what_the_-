import torch
import torch.nn as nn


class TimingGenerator(nn.Module):
    """LSTM-based generator: noise(16) + context(32 per step) -> timing sequences.

    Context is injected at every LSTM timestep so that HMM state, complexity,
    and fatigue signals are maintained throughout the entire sequence generation.
    """

    def __init__(self, noise_dim: int = 16, context_dim: int = 32,
                 hidden_size: int = 256, num_layers: int = 3, seq_len: int = 32,
                 step_noise_scale: float = 0.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.step_noise_scale = step_noise_scale

        self.noise_proj = nn.Linear(noise_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size + context_dim,  # context injected at every step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),   # [key_down_ms, key_up_ms, gap_ms]
            nn.Softplus(),       # ensure positive outputs
        )

    def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # noise: (B, noise_dim), context: (B, context_dim)
        x = self.noise_proj(noise)                                        # (B, hidden)
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1).clone()          # (B, seq_len, hidden)
        if self.step_noise_scale > 0.0:
            x = x + torch.randn_like(x) * self.step_noise_scale
        ctx = context.unsqueeze(1).expand(-1, self.seq_len, -1)          # (B, seq_len, context_dim)
        x = torch.cat([x, ctx], dim=-1)                                   # (B, seq_len, hidden+context_dim)
        out, _ = self.lstm(x)                                             # (B, seq_len, hidden)
        timing = self.output_proj(out)                                    # (B, seq_len, 3)
        return timing
