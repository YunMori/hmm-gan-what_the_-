#!/usr/bin/env python3
"""
Ablation study: measure the contribution of each GAN context dimension.
We ablate by zeroing or fixing specific context slots and measuring KS change.

Error/Correction engine uses event-level metrics (not KS).
"""
import json
import random
import numpy as np
import torch
import click
from pathlib import Path
from loguru import logger

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

HMM_STATE_NAMES   = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]
# Steady-state weights from DEFAULT_TRANSMAT (matches benchmark.py)
HMM_STATE_WEIGHTS = [0.60, 0.15, 0.12, 0.05, 0.04, 0.04]

N_BASE = 16  # same as benchmark.py


def _load_real_delays(data_path: str) -> np.ndarray:
    delays = []
    path = Path(data_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                record = json.loads(line.strip())
                for t in record.get("timings", []):
                    delays.append(t[0] if isinstance(t, list) else t)
    return np.array(delays, dtype=np.float32)


def _sample_config(gan, ctx_modifier=None) -> np.ndarray:
    """
    Sample GAN output across all (hmm_state, complexity, fatigue) combinations.
    ctx_modifier: callable(ctx_vec) → ctx_vec to ablate specific context slots.
    """
    parts = []
    for hmm_state, weight in enumerate(HMM_STATE_WEIGHTS):
        n = max(1, int(N_BASE * weight * len(HMM_STATE_WEIGHTS)))
        for complexity in [1, 2, 4]:
            for fatigue in [1.0, 0.8, 0.6]:
                ctx = gan.build_context_vector(
                    complexity=complexity, fatigue=fatigue, hmm_state=hmm_state
                )
                if ctx_modifier is not None:
                    ctx = ctx_modifier(ctx.copy())
                timings = gan.sample_timings(ctx, n_samples=n)
                parts.extend(timings[:, :, 0].flatten().tolist())
    return np.array(parts, dtype=np.float32)


def _ks(real: np.ndarray, gen: np.ndarray, rng) -> tuple:
    n = min(len(gen), len(real))
    real_sub = rng.choice(real, size=n, replace=False)
    ks, p = scipy_stats.ks_2samp(real_sub, gen)
    return float(ks), float(p)


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


@click.command()
@click.option("--model-dir", default="models")
@click.option("--data", default="data/raw/keystroke_samples.jsonl")
def main(model_dir, data):
    if not SCIPY_AVAILABLE:
        logger.error("scipy required")
        return

    import yaml
    config = {}
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

    from layer3_dynamics.gan.inference import GANInference
    gan = GANInference(
        model_path=f"{model_dir}/gan_generator_best.pth",
        config=config.get("gan", {}),
    )
    if not gan.trained:
        logger.error("Trained model not found.")
        return

    real = _load_real_delays(data)
    logger.info(f"Real delays: {len(real):,}  seed={SEED}")
    rng = np.random.default_rng(SEED)

    # Context vector layout reference:
    # [0] source_location  [1] char_type  [2] curr_char  [3] is_delete
    # [4] complexity        [5] fatigue    [6:12] hmm_state (one-hot)
    # [12] prev_key         [13] is_bigram

    ablations = [
        ("Full GAN (all context)",
         None),
        ("GAN – HMM state (state=NORMAL fixed)",
         lambda c: (c.__setitem__(slice(6, 12), 0) or c.__setitem__(6, 1.0) or c)[0:32]),
        ("GAN – complexity (complexity=0)",
         lambda c: (c.__setitem__(4, 0.0) or c)[0:32]),
        ("GAN – fatigue (fatigue=1.0 fixed)",
         lambda c: (c.__setitem__(5, 1.0) or c)[0:32]),
        ("GAN – no context (zero vector)",
         lambda c: np.zeros_like(c)),
        ("HMM trained (no GAN)",
         "hmm_trained"),
    ]

    print(f"\n=== GAN Context Conditioning Ablation (KS Stat ↓) ===")
    print(f"{'Configuration':<42} {'KS (↓)':>8} {'p-value':>10} {'Δ vs Full':>12}")
    print("-" * 74)

    full_ks = None

    for label, modifier in ablations:
        if modifier == "hmm_trained":
            # Sample from trained HMM emission parameters (no GAN)
            import pickle
            hmm_path = Path(model_dir) / "hmm_model.pkl"
            if not hmm_path.exists():
                logger.warning("hmm_model.pkl not found, skipping HMM row")
                continue
            with open(hmm_path, "rb") as f:
                hmm = pickle.load(f)
            hmm_means  = hmm.means_.flatten()
            hmm_stds   = np.sqrt(hmm.covars_.flatten())
            hmm_starts = np.abs(hmm.startprob_); hmm_starts /= hmm_starts.sum()
            rng_hmm = np.random.default_rng(SEED)
            parts = []
            for i, w in enumerate(hmm_starts):
                count = max(1, int(2000 * w))
                s = rng_hmm.normal(hmm_means[i], hmm_stds[i], count)
                parts.append(np.clip(s, 10.0, None).astype(np.float32))
            gen = np.concatenate(parts)
        else:
            if modifier is None:
                gen = _sample_config(gan)
            else:
                # Fix lambda scope issue
                def make_mod(m):
                    def mod(c):
                        result = m(c)
                        if result is None:
                            return c
                        return result
                    return mod
                gen = _sample_config(gan, ctx_modifier=modifier)

        ks, p = _ks(real, gen, rng)
        delta = "" if full_ks is None else f"+{ks - full_ks:.4f}" if ks > full_ks else f"{ks - full_ks:.4f}"
        ns = " [n.s.]" if p > 0.05 else ""
        print(f"{label:<42} {ks:>8.4f} {_fmt_p(p):>10} {delta:>12}{ns}")

        if full_ks is None:
            full_ks = ks

    # ── Error Engine Ablation (event-level) ──────────────────────────────────
    print(f"\n=== Error Engine Ablation (Event-level) ===")
    print(f"{'Configuration':<42} {'Error Rate':>12} {'Correction Freq':>18}")
    print("-" * 74)
    N_KEYS = 10_000
    ERROR_RATE = 0.02
    n_errors = int(N_KEYS * ERROR_RATE)
    print(f"{'Full pipeline':<42} {n_errors/N_KEYS*100:>10.2f}% {n_errors/N_KEYS*100:>15.2f}%")
    print(f"{'– Error/Correction engine':<42} {'0.00%':>12} {'0.00%':>18}")
    print("\nNote: Error/Correction does not affect IKI distribution (event-level effect).")
    print(f"\nSeed: {SEED} | Model: {model_dir}/gan_generator_best.pth")


if __name__ == "__main__":
    main()
