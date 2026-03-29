#!/usr/bin/env python3
"""
Per-State KS Evaluation
========================
Evaluates the KS statistic between generated and real IKI distributions
for each HMM behavioral state separately.

Two evaluation modes:

  1. CONDITIONING mode (default):
     For each state, fix the context vector to that state and generate N samples.
     Compare against the OVERALL real distribution.
     Measures: "does the model generate plausible IKIs when conditioned on state X?"

  2. TAGGING mode (--tag-real):
     Use the trained HMM to Viterbi-decode real sequences into states.
     For each state, compare state-tagged real IKIs vs GAN-generated IKIs.
     Measures: "does the model match the IKI distribution within each state?"

Usage:
  python scripts/per_state_ks.py --model-dir models --data data/raw/keystroke_samples.jsonl
  python scripts/per_state_ks.py --tag-real --model-dir models --data data/raw/...
"""
import json
import sys
import random
import pickle
import numpy as np
from pathlib import Path
import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

HMM_STATE_NAMES   = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]
# Default mean IKIs for each HMM state (from paper Table 2)
HMM_DEFAULT_MEANS = [120.0, 250.0, 60.0, 180.0, 100.0, 2000.0]

N_GEN_PER_STATE   = 300   # samples to generate per state
N_COMPLEXITY_VALS = [1, 2, 4]
N_FATIGUE_VALS    = [1.0, 0.8, 0.6]


def _load_real_delays(data_path: str) -> list:
    """Load all raw IKI values in ms from JSONL file."""
    delays = []
    path = Path(data_path)
    if not path.exists():
        return delays
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            for t in record.get("timings", []):
                val = t[0] if isinstance(t, list) else t
                delays.append(float(val))
    return delays


def _load_real_sequences(data_path: str) -> list:
    """Load raw IKI sequences (list of lists) for HMM tagging."""
    sequences = []
    path = Path(data_path)
    if not path.exists():
        return sequences
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            timings = record.get("timings", [])
            seq = [t[0] if isinstance(t, list) else t for t in timings]
            if len(seq) >= 4:
                sequences.append(np.array(seq, dtype=np.float32))
    return sequences


def _generate_for_state(gan, hmm_state: int, n: int) -> np.ndarray:
    """Generate IKI samples with a fixed HMM state, varying complexity & fatigue."""
    parts = []
    n_per_combo = max(1, n // (len(N_COMPLEXITY_VALS) * len(N_FATIGUE_VALS)))
    for complexity in N_COMPLEXITY_VALS:
        for fatigue in N_FATIGUE_VALS:
            ctx = gan.build_context_vector(
                complexity=complexity, fatigue=fatigue, hmm_state=hmm_state
            )
            timings = gan.sample_timings(ctx, n_samples=n_per_combo)
            parts.extend(timings[:, :, 0].flatten().tolist())
    return np.array(parts, dtype=np.float32)


def _ks(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    n = min(len(a), len(b))
    a_sub = rng.choice(a, size=n, replace=False)
    ks, p = scipy_stats.ks_2samp(a_sub, b)
    return float(ks), float(p)


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    filled = int(width * min(value / max_val, 1.0))
    return "█" * filled + "░" * (width - filled)


@click.command()
@click.option("--model-dir", default="models",                           show_default=True)
@click.option("--data",      default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--tag-real",  is_flag=True,
              help="Tag real data with HMM states and compare per-state distributions.")
@click.option("--n-samples", default=300, show_default=True,
              help="GAN samples to generate per state.")
@click.option("--output",    default="results/per_state_ks.json",        show_default=True)
@click.option("--noise-dim", default=None, type=int, show_default=True,
              help="Override noise_dim for model architecture.")
@click.option("--hidden-size", default=None, type=int, show_default=True,
              help="Override hidden_size for model architecture.")
@click.option("--checkpoint", default=None, show_default=True,
              help="Override checkpoint path (default: model_dir/gan_generator_best.pth).")
def main(model_dir, data, tag_real, n_samples, output, noise_dim, hidden_size, checkpoint):
    if not SCIPY_AVAILABLE:
        logger.error("scipy is required. Run: pip install scipy")
        return

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

    ckpt_path = checkpoint if checkpoint else f"{model_dir}/gan_generator_best.pth"

    from layer3_dynamics.gan.inference import GANInference
    gan = GANInference(
        model_path=ckpt_path,
        config=gan_config,
    )
    if not gan.trained:
        logger.error("Trained GAN model not found in model_dir.")
        return

    rng          = np.random.default_rng(SEED)
    real_all     = np.array(_load_real_delays(data), dtype=np.float32)
    logger.info(f"Loaded {len(real_all):,} real IKI samples from {data}")

    # ── Mode 1: CONDITIONING evaluation ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("MODE 1: CONDITIONING EVALUATION")
    print("  Context fixed to each HMM state; compare vs overall real distribution.")
    print(f"{'='*70}")
    print(f"\n{'State':<13} {'N_gen':>7} {'Median (ms)':>12} {'KS (↓)':>9} "
          f"{'p-value':>10}  Conditioning quality")
    print("-" * 70)

    conditioning_results = []
    for state_idx, state_name in enumerate(HMM_STATE_NAMES):
        gen = _generate_for_state(gan, state_idx, n_samples)
        ks, p = _ks(real_all, gen, rng)
        median_ms = float(np.median(gen))
        bar = _bar(1.0 - ks, max_val=1.0)   # higher bar = better KS
        print(f"{state_name:<13} {len(gen):>7,} {median_ms:>12.1f} {ks:>9.4f} "
              f"{_fmt_p(p):>10}  {bar}")
        conditioning_results.append({
            "state": state_name, "state_idx": state_idx,
            "n_generated": len(gen), "median_ms": median_ms, "ks": ks, "p": p,
        })

    medians = [r["median_ms"] for r in conditioning_results]
    spread  = float(np.std(medians))
    spread_range = float(max(medians) - min(medians))
    print(f"\n  State median spread (std):  {spread:.1f} ms")
    print(f"  State median spread (range): {spread_range:.1f} ms")
    print(f"  Real data median: {np.median(real_all):.1f} ms")

    if spread < 10.0:
        print("\n  [!] Low spread — context conditioning has weak effect on output.")
        print("      Consider increasing noise_dim-to-proj ratio (see ablation_noise_dim.py)")
    else:
        print(f"\n  [OK] State conditioning is differentiating outputs by {spread:.0f}ms std.")

    # ── Mode 2: TAGGING evaluation (optional) ─────────────────────────────────
    tagging_results = []
    if tag_real:
        print(f"\n{'='*70}")
        print("MODE 2: TAGGED REAL DATA vs GAN PER-STATE")
        print("  HMM Viterbi-decodes real sequences into states.")
        print(f"{'='*70}")

        hmm_path = Path(model_dir) / "hmm_model.pkl"
        if not hmm_path.exists():
            logger.warning("hmm_model.pkl not found — skipping tagging mode.")
        else:
            with open(hmm_path, "rb") as f:
                hmm = pickle.load(f)

            sequences = _load_real_sequences(data)
            logger.info(f"Loaded {len(sequences)} sequences for HMM tagging")

            state_pools = {i: [] for i in range(6)}
            for seq in sequences:
                try:
                    # hmmlearn Viterbi decode
                    seq_2d = seq.reshape(-1, 1)
                    _, state_seq = hmm.decode(seq_2d)
                    for iki, state in zip(seq, state_seq):
                        state_pools[state].append(float(iki))
                except Exception:
                    continue

            print(f"\n{'State':<13} {'Real N':>8} {'Gen N':>7} {'Real med':>10} "
                  f"{'Gen med':>9} {'KS (↓)':>9} {'p-value':>10}")
            print("-" * 70)

            for state_idx, state_name in enumerate(HMM_STATE_NAMES):
                real_state = np.array(state_pools[state_idx], dtype=np.float32)
                gen_state  = _generate_for_state(gan, state_idx, n_samples)

                if len(real_state) < 50:
                    print(f"{state_name:<13} {'<50 samples, skip':>50}")
                    continue

                ks, p = _ks(real_state, gen_state, rng)
                tagging_results.append({
                    "state": state_name, "state_idx": state_idx,
                    "n_real": len(real_state), "n_generated": len(gen_state),
                    "real_median_ms": float(np.median(real_state)),
                    "gen_median_ms":  float(np.median(gen_state)),
                    "ks": ks, "p": p,
                })
                print(f"{state_name:<13} {len(real_state):>8,} {len(gen_state):>7,} "
                      f"{np.median(real_state):>10.1f} {np.median(gen_state):>9.1f} "
                      f"{ks:>9.4f} {_fmt_p(p):>10}")

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "seed": SEED,
        "model_dir": model_dir,
        "data": data,
        "n_real_total": len(real_all),
        "n_samples_per_state": n_samples,
        "conditioning_results": conditioning_results,
        "state_spread_std_ms": spread,
        "state_spread_range_ms": spread_range,
        "tagging_results": tagging_results,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
