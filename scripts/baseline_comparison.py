#!/usr/bin/env python3
"""
Baseline comparison: HumanType vs Fixed WPM, Lognormal, HMM-only (default),
HMM-only (trained), and HumanType (ours).
"""
import json
import pickle
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
# Weights match benchmark.py (steady-state distribution from DEFAULT_TRANSMAT)
HMM_STATE_WEIGHTS = [0.60, 0.15, 0.12, 0.05, 0.04, 0.04]

# Default HMM Gaussian parameters (untrained prior)
HMM_DEFAULT_MEANS = {
    "NORMAL":     120.0,
    "SLOW":       250.0,
    "FAST":        60.0,
    "ERROR":      180.0,
    "CORRECTION": 100.0,
    "PAUSE":     2000.0,
}
HMM_DEFAULT_STDS = {k: v * 0.3 for k, v in HMM_DEFAULT_MEANS.items()}

N_SAMPLES = 20_000


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


def _ks_result(real: np.ndarray, gen: np.ndarray, rng) -> tuple:
    n = min(len(gen), len(real))
    real_sub = rng.choice(real, size=n, replace=False)
    ks, p = scipy_stats.ks_2samp(real_sub, gen)
    return float(ks), float(p)


def _fmt_pval(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _sample_fixed_wpm(n: int) -> np.ndarray:
    """65 WPM → mean IKI ≈ 185ms; add small Gaussian noise to avoid zero variance."""
    rng = np.random.default_rng(SEED)
    return rng.normal(185.0, 5.0, n).astype(np.float32)


def _sample_lognormal(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.lognormal(np.log(120.0), 0.8, n).astype(np.float32)


def _sample_hmm_default(n: int) -> np.ndarray:
    """HMM Gaussian sampling with default (untrained) parameters."""
    rng = np.random.default_rng(SEED)
    parts = []
    for name, weight in zip(HMM_STATE_NAMES, HMM_STATE_WEIGHTS):
        count = max(1, int(n * weight))
        samples = rng.normal(HMM_DEFAULT_MEANS[name], HMM_DEFAULT_STDS[name], count)
        samples = np.clip(samples, 10.0, None)
        parts.append(samples.astype(np.float32))
    return np.concatenate(parts)


def _sample_hmm_trained(model_path: str, n: int) -> np.ndarray | None:
    """Gaussian sampling from trained HMM emission parameters."""
    path = Path(model_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    rng = np.random.default_rng(SEED)
    means = model.means_.flatten()           # (n_states,)
    stds  = np.sqrt(model.covars_.flatten()) # (n_states,)
    weights = np.abs(model.startprob_)
    weights /= weights.sum()
    parts = []
    for i, w in enumerate(weights):
        count = max(1, int(n * w))
        samples = rng.normal(means[i], stds[i], count)
        samples = np.clip(samples, 10.0, None)
        parts.append(samples.astype(np.float32))
    return np.concatenate(parts)


def _sample_humantype(gan, n: int) -> np.ndarray:
    """HumanType: GAN sampling across all HMM states, complexity, and fatigue levels.
    Matches benchmark.py approach — raw GAN output without additional post-corrections,
    since the GAN was trained on real data that already embeds Fitts/bigram/fatigue effects.
    """
    parts = []
    n_base = max(1, n // (len(HMM_STATE_NAMES) * 3 * 3))  # per (state, complexity, fatigue)
    for hmm_state, (weight, name) in enumerate(zip(HMM_STATE_WEIGHTS, HMM_STATE_NAMES)):
        count = max(1, int(n * weight))
        for complexity in [1, 2, 4]:
            for fatigue in [1.0, 0.8, 0.6]:
                ctx = gan.build_context_vector(
                    complexity=complexity, fatigue=fatigue, hmm_state=hmm_state
                )
                n_sub = max(1, count // 9)
                timings = gan.sample_timings(ctx, n_samples=n_sub)
                delays_ms = timings[:, :, 0].flatten()
                parts.append(delays_ms)
    return np.concatenate(parts)


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

    real = _load_real_delays(data)
    if len(real) == 0:
        logger.error("No real data found.")
        return

    logger.info(f"Real delays: {len(real):,}  N_SAMPLES={N_SAMPLES}  seed={SEED}")
    rng = np.random.default_rng(SEED)

    baselines = [
        ("Fixed WPM (65 WPM)",      _sample_fixed_wpm(N_SAMPLES)),
        ("Lognormal prior",         _sample_lognormal(N_SAMPLES)),
        ("HMM (default params)",    _sample_hmm_default(N_SAMPLES)),
    ]

    hmm_trained = _sample_hmm_trained(f"{model_dir}/hmm_model.pkl", N_SAMPLES)
    if hmm_trained is not None:
        baselines.append(("HMM (trained)",    hmm_trained))

    baselines.append(("HumanType (ours)", _sample_humantype(gan, N_SAMPLES)))

    print("\n=== Baseline Comparison ===")
    print(f"{'Method':<28} {'KS (↓)':>8} {'p-value':>10} {'Median (ms)':>12} {'p95 (ms)':>10}")
    print("-" * 72)

    for label, gen in baselines:
        ks, p = _ks_result(real, gen, rng)
        marker = " ◀ ours" if "HumanType" in label else ""
        print(f"{label:<28} {ks:>8.4f} {_fmt_pval(p):>10} "
              f"{np.median(gen):>12.1f} {np.percentile(gen,95):>10.1f}{marker}")

    print(f"\nReal                         —          —        "
          f"{np.median(real):>12.1f} {np.percentile(real,95):>10.1f}")
    print(f"\nSeed: {SEED}")


if __name__ == "__main__":
    main()
