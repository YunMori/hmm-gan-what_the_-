import torch
import numpy as np
from typing import Optional, List
from pathlib import Path
from loguru import logger

from layer3_dynamics.gan.generator import TimingGenerator


class GANInference:
    def __init__(self, model_path: Optional[str] = None, config: dict = None):
        config = config or {}
        self.noise_dim = config.get("noise_dim", 16)
        self.context_dim = config.get("context_dim", 32)
        self.seq_len = config.get("seq_len", 32)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.G = TimingGenerator(
            noise_dim=self.noise_dim,
            context_dim=self.context_dim,
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 3),
            seq_len=self.seq_len,
            step_noise_scale=config.get("step_noise_scale", 0.0),
        )
        self.G.to(self.device)

        self.trained = False
        if model_path and Path(model_path).exists():
            self.G.load_state_dict(torch.load(model_path, map_location=self.device))
            self.trained = True
            logger.info(f"Loaded GAN generator from {model_path}")
        else:
            logger.info("No GAN model found, using untrained generator (fallback to HMM)")

        self.G.eval()

    def sample_timings(self, context_vector: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Returns (n_samples, seq_len, 3) array of [key_down_ms, key_up_ms, gap_ms].
        Values are in milliseconds.
        """
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim, device=self.device)
            ctx = torch.as_tensor(context_vector, dtype=torch.float32).unsqueeze(0).expand(n_samples, -1).to(self.device)
            timings = self.G(noise, ctx)  # (n, seq_len, 3)
            # Convert from normalized seconds back to ms
            timings_ms = timings.cpu().numpy() * 1000.0
        return timings_ms

    def build_context_vector(self, complexity: int, fatigue: float,
                              hmm_state: int, prev_key: str = " ",
                              is_bigram: bool = False,
                              source_location: float = 0.0,
                              curr_char: str = " ") -> np.ndarray:
        """Build 32-dim context vector.

        Layout (unified with scripts/convert_cs1_dataset.py _build_context):
            [0]    source_location (0-1)
            [1]    char_type (0=alpha, 0.5=digit, 1=special)
            [2]    curr_char ASCII / 128
            [3]    is_delete
            [4]    complexity (0-1)
            [5]    fatigue (0-1)
            [6:12] HMM state one-hot (6 states)
            [12]   prev_key ASCII / 128
            [13]   is_bigram
            [14-31] reserved (0)
        """
        ctx = np.zeros(self.context_dim, dtype=np.float32)
        ctx[0] = float(np.clip(source_location, 0.0, 1.0))
        if curr_char:
            c = curr_char[0]
            if c.isalpha():
                ctx[1] = 0.0
            elif c.isdigit():
                ctx[1] = 0.5
            else:
                ctx[1] = 1.0
            ctx[2] = ord(c) / 128.0
            ctx[3] = 1.0 if curr_char in ("\x08", "backspace", "BackSpace") else 0.0
        ctx[4] = complexity / 4.0
        ctx[5] = fatigue
        if 0 <= hmm_state < 6:
            ctx[6 + hmm_state] = 1.0
        if prev_key:
            ctx[12] = ord(prev_key[0]) / 128.0
        ctx[13] = float(is_bigram)
        return ctx
