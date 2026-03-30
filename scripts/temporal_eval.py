#!/usr/bin/env python3
"""
Temporal Dependency Evaluation
================================
Evaluates how well the GAN preserves temporal structure in keystroke sequences
beyond marginal IKI distribution matching (KS test).

Metrics:
  1. AUTOCORRELATION profile: ACF at lags 1-10 for real vs generated sequences.
     A model that merely matches the marginal distribution can fail here.

  2. FIRST-ORDER TRANSITION: P(IKI[t+1] > threshold | IKI[t] > threshold)
     Measures burst-to-burst transition probabilities (pause clustering).

  3. RUN-LENGTH DISTRIBUTION: how many consecutive "fast" (<100ms) keystrokes
     appear in a row. Captures bigram burst length.

  4. ENTROPY RATE: Shannon entropy of discretized IKI sequences.
     Higher entropy = less predictable temporal structure.

Usage:
  python scripts/temporal_eval.py --model-dir models --data data/raw/keystroke_samples.jsonl
  python scripts/temporal_eval.py --model-dir models --data ... --output results/temporal.json
"""
import json
import sys
import random
import numpy as np
from pathlib import Path
import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HMM_STATE_NAMES   = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]
HMM_STATE_WEIGHTS = [0.40, 0.17, 0.16, 0.17, 0.03, 0.07]

# Bins for IKI discretization (ms)
IKI_BINS = [0, 60, 120, 200, 350, 600, 1200, 5000]
FAST_THRESHOLD   = 100    # ms — "fast" keystroke
PAUSE_THRESHOLD  = 600    # ms — "pause" keystroke
ACF_MAX_LAG      = 10


def _load_real_sequences(data_path: str, max_seqs: int = 5000) -> list:
    """Load list of IKI arrays (ms) from JSONL."""
    seqs = []
    path = Path(data_path)
    if not path.exists():
        return seqs
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_seqs:
                break
            record = json.loads(line.strip())
            timings = record.get("timings", [])
            seq = [t[0] if isinstance(t, list) else t for t in timings]
            if len(seq) >= 8:
                seqs.append(np.array(seq, dtype=np.float32))
    return seqs


def _generate_sequences(gan, n_seqs: int, seq_len: int = 32) -> list:
    """Generate IKI sequences from GAN across diverse contexts."""
    seqs = []
    n_per_state = max(1, n_seqs // 6)
    for state_idx, weight in enumerate(HMM_STATE_WEIGHTS):
        n = max(1, int(n_seqs * weight))
        for complexity in [1, 3]:
            for fatigue in [1.0, 0.7]:
                ctx = gan.build_context_vector(
                    complexity=complexity, fatigue=fatigue, hmm_state=state_idx
                )
                timings = gan.sample_timings(ctx, n_samples=max(1, n // 4))
                for i in range(timings.shape[0]):
                    seqs.append(timings[i, :, 0])   # keydown times in ms
    return seqs


# ── Metric 1: Autocorrelation ─────────────────────────────────────────────────

def _autocorrelation(seqs: list, max_lag: int = ACF_MAX_LAG) -> np.ndarray:
    """
    Compute mean autocorrelation at lags 1..max_lag across all sequences.
    Uses log-transform to stabilize variance (IKIs are lognormal).
    Returns array of shape (max_lag,).
    """
    acf_sum  = np.zeros(max_lag)
    acf_cnt  = np.zeros(max_lag)
    for seq in seqs:
        if len(seq) < max_lag + 2:
            continue
        # log-transform for stationarity
        x = np.log1p(np.clip(seq, 1.0, None))
        x = (x - x.mean()) / (x.std() + 1e-8)
        for lag in range(1, max_lag + 1):
            if len(x) > lag:
                c = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
                if not np.isnan(c):
                    acf_sum[lag - 1]  += c
                    acf_cnt[lag - 1]  += 1
    return np.where(acf_cnt > 0, acf_sum / acf_cnt, 0.0)


# ── Metric 2: Pause transition ────────────────────────────────────────────────

def _pause_transition_prob(seqs: list, threshold: float = PAUSE_THRESHOLD) -> float:
    """P(IKI[t+1] > threshold | IKI[t] > threshold) — pause clustering."""
    n_cond, n_event = 0, 0
    for seq in seqs:
        for t in range(len(seq) - 1):
            if seq[t] > threshold:
                n_event += 1
                if seq[t + 1] > threshold:
                    n_cond += 1
    return n_cond / n_event if n_event > 0 else 0.0


# ── Metric 3: Run-length distribution ─────────────────────────────────────────

def _run_length_distribution(seqs: list, threshold: float = FAST_THRESHOLD) -> dict:
    """Distribution of consecutive fast-keystroke run lengths."""
    run_counts = {}
    for seq in seqs:
        run = 0
        for iki in seq:
            if iki < threshold:
                run += 1
            else:
                if run > 0:
                    run_counts[run] = run_counts.get(run, 0) + 1
                run = 0
        if run > 0:
            run_counts[run] = run_counts.get(run, 0) + 1
    total = sum(run_counts.values())
    return {k: v / total for k, v in sorted(run_counts.items())} if total > 0 else {}


def _mean_run_length(run_dist: dict) -> float:
    if not run_dist:
        return 0.0
    return sum(k * v for k, v in run_dist.items())


# ── Metric 4: Entropy rate ────────────────────────────────────────────────────

def _entropy_rate(seqs: list, bins: list = IKI_BINS) -> float:
    """Shannon entropy of discretized IKI values."""
    all_vals = np.concatenate(seqs) if seqs else np.array([])
    if len(all_vals) == 0:
        return 0.0
    digitized = np.digitize(all_vals, bins)
    counts     = np.bincount(digitized, minlength=len(bins) + 1)
    probs      = counts / counts.sum()
    probs      = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ── Comparison helpers ────────────────────────────────────────────────────────

def _acf_mae(real_acf: np.ndarray, gen_acf: np.ndarray) -> float:
    return float(np.mean(np.abs(real_acf - gen_acf)))


def _run_dist_tv(real_dist: dict, gen_dist: dict) -> float:
    """Total variation distance between run-length distributions."""
    all_keys = set(real_dist) | set(gen_dist)
    return 0.5 * sum(abs(real_dist.get(k, 0) - gen_dist.get(k, 0)) for k in all_keys)


@click.command()
@click.option("--model-dir", default="models",                           show_default=True)
@click.option("--data",      default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--n-seqs",    default=2000, show_default=True,
              help="Number of sequences to evaluate (real and generated).")
@click.option("--output",    default="results/temporal_eval.json",       show_default=True)
@click.option("--noise-dim", default=None, type=int, show_default=True,
              help="Override noise_dim for model architecture.")
@click.option("--hidden-size", default=None, type=int, show_default=True,
              help="Override hidden_size for model architecture.")
@click.option("--checkpoint", default=None, show_default=True,
              help="Override checkpoint path (default: model_dir/gan_generator_best.pth).")
@click.option("--step-noise-scale", default=0.0, type=float, show_default=True,
              help="Per-step noise scale for TimingGenerator (Config D).")
def main(model_dir, data, n_seqs, output, noise_dim, hidden_size, checkpoint, step_noise_scale):
    import yaml
    config = {}
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

    gan_config = config.get("gan", {})
    if noise_dim is not None:
        gan_config["noise_dim"] = noise_dim
    if hidden_size is not None:
        gan_config["hidden_size"] = hidden_size
    gan_config["step_noise_scale"] = step_noise_scale

    ckpt_path = checkpoint if checkpoint else f"{model_dir}/gan_generator_best.pth"

    from layer3_dynamics.gan.inference import GANInference
    gan = GANInference(
        model_path=ckpt_path,
        config=gan_config,
    )
    if not gan.trained:
        logger.error("Trained GAN model not found.")
        return

    logger.info(f"Loading real sequences from {data}...")
    real_seqs = _load_real_sequences(data, max_seqs=n_seqs)
    logger.info(f"  {len(real_seqs)} real sequences loaded.")

    logger.info("Generating GAN sequences...")
    gen_seqs = _generate_sequences(gan, n_seqs=n_seqs)
    logger.info(f"  {len(gen_seqs)} generated sequences produced.")

    # ── Compute metrics ────────────────────────────────────────────────────────
    logger.info("Computing autocorrelation...")
    real_acf = _autocorrelation(real_seqs)
    gen_acf  = _autocorrelation(gen_seqs)
    acf_mae  = _acf_mae(real_acf, gen_acf)

    logger.info("Computing pause transition probabilities...")
    real_pause_tp = _pause_transition_prob(real_seqs)
    gen_pause_tp  = _pause_transition_prob(gen_seqs)

    logger.info("Computing run-length distributions...")
    real_runs = _run_length_distribution(real_seqs)
    gen_runs  = _run_length_distribution(gen_seqs)
    run_tv    = _run_dist_tv(real_runs, gen_runs)
    real_mean_run = _mean_run_length(real_runs)
    gen_mean_run  = _mean_run_length(gen_runs)

    logger.info("Computing entropy rates...")
    real_entropy = _entropy_rate(real_seqs)
    gen_entropy  = _entropy_rate(gen_seqs)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEMPORAL DEPENDENCY EVALUATION")
    print(f"{'='*60}")

    print(f"\n1. Autocorrelation (log-IKI) at lags 1–{ACF_MAX_LAG}:")
    print(f"   {'Lag':<6} {'Real ACF':>10} {'Gen ACF':>10} {'Δ':>8}")
    print(f"   {'-'*36}")
    for lag in range(ACF_MAX_LAG):
        delta = gen_acf[lag] - real_acf[lag]
        print(f"   {lag+1:<6} {real_acf[lag]:>10.4f} {gen_acf[lag]:>10.4f} {delta:>+8.4f}")
    print(f"   MAE(ACF): {acf_mae:.4f}  (lower = better temporal match)")

    print(f"\n2. Pause Clustering — P(pause | prev_pause)  [threshold={PAUSE_THRESHOLD}ms]:")
    print(f"   Real:      {real_pause_tp:.4f}")
    print(f"   Generated: {gen_pause_tp:.4f}")
    print(f"   Δ:         {gen_pause_tp - real_pause_tp:+.4f}")

    print(f"\n3. Fast-Keystroke Run Lengths  [threshold<{FAST_THRESHOLD}ms]:")
    print(f"   Mean run length — Real: {real_mean_run:.2f}  |  Gen: {gen_mean_run:.2f}")
    print(f"   Total variation distance: {run_tv:.4f}  (lower = better)")
    max_run = min(8, max((max(real_runs, default=0), max(gen_runs, default=0))) + 1)
    if max_run > 0:
        print(f"   {'Run':>4} {'Real':>8} {'Gen':>8}")
        for r in range(1, max_run + 1):
            print(f"   {r:>4} {real_runs.get(r, 0)*100:>7.1f}% {gen_runs.get(r, 0)*100:>7.1f}%")

    print(f"\n4. IKI Entropy Rate (bits):")
    print(f"   Real:      {real_entropy:.3f}")
    print(f"   Generated: {gen_entropy:.3f}")
    print(f"   Δ:         {gen_entropy - real_entropy:+.3f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ACF MAE:           {acf_mae:.4f}  (marginal dist. is KS={0.0716:.4f})")
    print(f"  Pause transition Δ: {abs(gen_pause_tp - real_pause_tp):.4f}")
    print(f"  Run-length TV:      {run_tv:.4f}")
    print(f"  Entropy Δ:          {abs(gen_entropy - real_entropy):.3f} bits")

    if acf_mae > 0.05:
        print("\n  [!] ACF MAE > 0.05: temporal dependencies are not well preserved.")
    else:
        print("\n  [OK] ACF MAE ≤ 0.05: temporal structure is well matched.")

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "seed": SEED,
        "n_real_seqs": len(real_seqs),
        "n_gen_seqs":  len(gen_seqs),
        "autocorrelation": {
            "lags": list(range(1, ACF_MAX_LAG + 1)),
            "real": real_acf.tolist(),
            "generated": gen_acf.tolist(),
            "mae": acf_mae,
        },
        "pause_transition": {
            "threshold_ms": PAUSE_THRESHOLD,
            "real": real_pause_tp,
            "generated": gen_pause_tp,
            "delta": gen_pause_tp - real_pause_tp,
        },
        "run_length": {
            "threshold_ms": FAST_THRESHOLD,
            "real_mean": real_mean_run,
            "gen_mean":  gen_mean_run,
            "tv_distance": run_tv,
            "real_dist": {str(k): v for k, v in real_runs.items()},
            "gen_dist":  {str(k): v for k, v in gen_runs.items()},
        },
        "entropy_rate": {
            "bins_ms": IKI_BINS,
            "real": real_entropy,
            "generated": gen_entropy,
            "delta": gen_entropy - real_entropy,
        },
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
