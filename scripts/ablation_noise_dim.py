#!/usr/bin/env python3
"""
noise_dim Ablation Study
========================
Trains multiple GAN configurations with different noise_dim values and
noise projection sizes to measure their effect on:
  1. Final KS statistic (distribution matching quality)
  2. Per-state IKI median spread (conditioning effectiveness)
  3. Training convergence speed

Configurations tested:
  A) noise_dim=8,  proj=256  (current ratio, tiny noise)
  B) noise_dim=16, proj=256  (current baseline)
  C) noise_dim=32, proj=128  (balanced ratio)
  D) noise_dim=64, proj=64   (proposed: context-dominant)
  E) noise_dim=64, proj=256  (high noise)

Noise-to-Context ratio in LSTM input = proj_size : context_dim(32)
  A: 256:32 = 8.0x noise dominant
  B: 256:32 = 8.0x noise dominant  (current)
  C: 128:32 = 4.0x noise dominant
  D:  64:32 = 2.0x noise dominant  (proposed fix)
  E: 256:32 = 8.0x noise dominant, more noise capacity

Usage:
  python scripts/ablation_noise_dim.py --data data/raw/keystroke_samples.jsonl
  python scripts/ablation_noise_dim.py --quick --epochs 50   # fast sanity check
  python scripts/ablation_noise_dim.py --full  --epochs 300  # convergence study
"""
import json
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from loguru import logger
import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
HMM_STATE_WEIGHTS = [0.40, 0.17, 0.16, 0.17, 0.03, 0.07]

# ── Ablation configurations ────────────────────────────────────────────────────
CONFIGS = [
    {"name": "A: noise=8,  proj=256", "noise_dim": 8,  "hidden_size": 256, "ratio": "8.0x"},
    {"name": "B: noise=16, proj=256", "noise_dim": 16, "hidden_size": 256, "ratio": "8.0x (baseline)"},
    {"name": "C: noise=32, proj=128", "noise_dim": 32, "hidden_size": 128, "ratio": "4.0x"},
    {"name": "D: noise=64, proj=64, nl=2+sn", "noise_dim": 64, "hidden_size": 64, "ratio": "2.0x+step_noise", "num_layers": 2, "step_noise_scale": 0.3},
    {"name": "E: noise=64, proj=256", "noise_dim": 64, "hidden_size": 256, "ratio": "8.0x (more noise)"},
    {"name": "F: noise=64, proj=64, nl=2",    "noise_dim": 64, "hidden_size": 64, "ratio": "2.0x",           "num_layers": 2, "step_noise_scale": 0.0},
]


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_ks(G, noise_dim: int, device, real_delays: np.ndarray,
                n_samples: int = 512) -> float:
    if not SCIPY_AVAILABLE or len(real_delays) == 0:
        return 1.0
    context_dim = 32
    G.eval()
    parts = []
    with torch.no_grad():
        for hmm_state, weight in enumerate(HMM_STATE_WEIGHTS):
            n = max(1, int(n_samples * weight))
            ctx = np.zeros(context_dim, dtype=np.float32)
            ctx[4] = 0.604
            ctx[5] = 0.523
            ctx[6 + hmm_state] = 1.0
            ctx_t = torch.as_tensor(ctx).unsqueeze(0).expand(n, -1).to(device)
            noise = torch.randn(n, noise_dim, device=device)
            timings = G(noise, ctx_t).cpu().numpy() * 1000.0
            parts.append(timings[:, :, 0].flatten())
    G.train()
    gen_arr = np.concatenate(parts)
    rng = np.random.default_rng(SEED)
    real_sub = rng.choice(real_delays, size=min(len(gen_arr), len(real_delays)), replace=False)
    ks_stat, _ = scipy_stats.ks_2samp(real_sub, gen_arr)
    return float(ks_stat)


def _compute_acf_mae(G, noise_dim: int, device, real_seqs: list,
                     max_lag: int = 10, n_gen: int = 300) -> float:
    """
    ACF MAE: log-IKI autocorrelation at lags 1..max_lag, real vs generated.
    real_seqs: list of np.ndarray (IKI in seconds from val dataset).
    """
    context_dim = 32

    def _acf(seqs):
        acf_sum = np.zeros(max_lag)
        acf_cnt = np.zeros(max_lag)
        for seq in seqs:
            if len(seq) < max_lag + 2:
                continue
            x = np.log1p(np.clip(seq * 1000.0, 1.0, None))  # seconds → ms → log
            x = (x - x.mean()) / (x.std() + 1e-8)
            for lag in range(1, max_lag + 1):
                c = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
                if not np.isnan(c):
                    acf_sum[lag - 1] += c
                    acf_cnt[lag - 1] += 1
        return np.where(acf_cnt > 0, acf_sum / acf_cnt, 0.0)

    # Real ACF
    real_acf = _acf(real_seqs[:500])

    # Generated ACF
    G.eval()
    gen_seqs = []
    with torch.no_grad():
        n_per_state = max(1, n_gen // 6)
        for hmm_state in range(6):
            ctx = np.zeros(context_dim, dtype=np.float32)
            ctx[4] = 0.5; ctx[5] = 0.8
            ctx[6 + hmm_state] = 1.0
            ctx_t = torch.as_tensor(ctx).unsqueeze(0).expand(n_per_state, -1).to(device)
            noise = torch.randn(n_per_state, noise_dim, device=device)
            timings = G(noise, ctx_t).cpu().numpy()  # seconds
            for i in range(timings.shape[0]):
                gen_seqs.append(timings[i, :, 0])
    G.train()

    gen_acf = _acf(gen_seqs)
    return float(np.mean(np.abs(real_acf - gen_acf)))


def _compute_per_state_spread(G, noise_dim: int, device, n_per_state: int = 200) -> float:
    """
    Returns std of per-state median IKIs.
    Higher std = better state differentiation (context is influencing output).
    """
    context_dim = 32
    G.eval()
    medians = []
    with torch.no_grad():
        for hmm_state in range(6):
            ctx = np.zeros(context_dim, dtype=np.float32)
            ctx[4] = 0.5   # medium complexity
            ctx[5] = 0.8   # slight fatigue
            ctx[6 + hmm_state] = 1.0
            ctx_t = torch.as_tensor(ctx).unsqueeze(0).expand(n_per_state, -1).to(device)
            noise = torch.randn(n_per_state, noise_dim, device=device)
            timings = G(noise, ctx_t).cpu().numpy() * 1000.0
            state_delays = timings[:, :, 0].flatten()
            medians.append(float(np.median(state_delays)))
    G.train()
    spread = float(np.std(medians))
    return spread, medians


def _build_generator(noise_dim: int, hidden_size: int, context_dim: int = 32,
                     num_layers: int = 3, seq_len: int = 32,
                     step_noise_scale: float = 0.0):
    """Build a TimingGenerator with custom noise_dim and hidden_size."""
    from layer3_dynamics.gan.generator import TimingGenerator
    return TimingGenerator(
        noise_dim=noise_dim,
        context_dim=context_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=seq_len,
        step_noise_scale=step_noise_scale,
    )


def train_config(cfg: dict, dataset_path: str, epochs: int,
                 batch_size: int, device_str: str,
                 eval_every: int = 10, target_ks: float = 0.10,
                 patience: int = 15,
                 checkpoint_dir: str = "models",
                 log_file: str = None) -> dict:
    """Train one GAN configuration and return results dict."""
    from layer3_dynamics.gan.discriminator import TimingDiscriminator
    from layer3_dynamics.gan.dataset import KeystrokeDataset

    noise_dim   = cfg["noise_dim"]
    hidden_size = cfg["hidden_size"]
    device      = torch.device(device_str)

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    safe_name   = cfg["name"].replace(":", "").replace(" ", "_").replace(",", "")
    ckpt_dir    = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    g_path      = ckpt_dir / f"gan_generator_{safe_name}_best.pth"
    d_path      = ckpt_dir / f"gan_discriminator_{safe_name}_best.pth"

    # ── Log file (JSONL, one entry per eval) ─────────────────────────────────
    log_path = Path(log_file) if log_file else \
               Path("logs") / f"train_{safe_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    G = _build_generator(
        noise_dim, hidden_size,
        num_layers=cfg.get("num_layers", 3),
        step_noise_scale=cfg.get("step_noise_scale", 0.0),
    ).to(device)
    D = TimingDiscriminator(context_dim=32, hidden_size=256).to(device)

    best_ks          = float("inf")
    best_score       = float("inf")

    # Resume from checkpoint if it exists
    start_epoch = 0
    if g_path.exists() and d_path.exists():
        G.load_state_dict(torch.load(g_path, map_location=device))
        D.load_state_dict(torch.load(d_path, map_location=device))
        # Read log and recompute best_ks / best_score from all entries
        if log_path.exists():
            with open(log_path) as f:
                lines = [l for l in f if l.strip()]
            if lines:
                last = json.loads(lines[-1])
                start_epoch = last.get("epoch", 0)
                # Recompute best_ks and best_score across all log entries
                # so that entries written before combined score was added are included
                SPREAD_LAMBDA = 0.001
                for line in lines:
                    entry = json.loads(line)
                    ks_val = entry.get("ks", float("inf"))
                    sp_val = entry.get("state_spread_ms", 0.0)
                    sc_val = ks_val - SPREAD_LAMBDA * sp_val
                    if ks_val < best_ks:
                        best_ks = ks_val
                    if sc_val < best_score:
                        best_score = sc_val
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}: {g_path} "
                    f"(best_ks={best_ks:.4f}, best_score={best_score:.4f})")

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    full_dataset = KeystrokeDataset(dataset_path, seq_len=32)
    val_size     = max(1000, int(len(full_dataset) * 0.10))
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    real_delays = np.array([
        full_dataset.sequences[idx]["timings"][t, 0] * 1000.0
        for idx in val_ds.indices
        for t in range(full_dataset.sequences[idx]["timings"].shape[0])
    ])

    # Real sequences (in seconds) for ACF MAE computation
    real_seqs = [
        full_dataset.sequences[idx]["timings"][:, 0]   # (seq_len,) in seconds
        for idx in val_ds.indices[:500]
    ]

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0)

    no_improve       = 0
    below_target_cnt = 0   # 연속으로 KS < target_ks 달성 횟수
    ks_history       = []
    stop_epoch       = epochs

    logger.info(f"\n[{cfg['name']}] Starting training ({epochs} epochs, "
                f"checkpoint→{g_path}, log→{log_path})...")

    for epoch in range(start_epoch, epochs):
        G.train(); D.train()
        g_losses, d_losses = [], []
        for batch in loader:
            real = batch["timings"].to(device)
            ctx  = batch["context"][:, 0, :].to(device)
            B    = real.size(0)

            # D step ×2
            noise_d = torch.randn(B, noise_dim, device=device)
            fake_d  = G(noise_d, ctx).detach()
            for _ in range(2):
                d_real = D(real, ctx)
                d_fake = D(fake_d, ctx)
                d_loss = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                d_losses.append(d_loss.item())

            # G step ×1
            noise_g = torch.randn(B, noise_dim, device=device)
            fake_g  = G(noise_g, ctx)
            g_loss  = -D(fake_g, ctx).mean()
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

        if SCIPY_AVAILABLE and (epoch + 1) % eval_every == 0:
            ks        = _compute_ks(G, noise_dim, device, real_delays)
            acf_mae   = _compute_acf_mae(G, noise_dim, device, real_seqs)
            spread, _ = _compute_per_state_spread(G, noise_dim, device, n_per_state=100)
            g_mean    = sum(g_losses) / len(g_losses)
            d_mean    = sum(d_losses) / len(d_losses)

            # 연속 target 달성 카운트
            if ks < target_ks:
                below_target_cnt += 1
            else:
                below_target_cnt = 0

            # Combined score: lower is better
            # spread bonus: 100ms spread ≈ 0.1 KS improvement
            SPREAD_LAMBDA = 0.001
            score = ks - SPREAD_LAMBDA * spread

            entry   = {
                "epoch": epoch + 1, "ks": round(ks, 6),
                "best_ks": round(min(best_ks, ks), 6),
                "score": round(score, 6),
                "best_score": round(min(best_score, score), 6),
                "acf_mae": round(acf_mae, 4),
                "state_spread_ms": round(spread, 2),
                "g_loss": round(g_mean, 4), "d_loss": round(d_mean, 4),
                "no_improve": no_improve,
                "below_target_cnt": below_target_cnt,
            }
            ks_history.append(entry)

            # Append to persistent log file
            with open(log_path, "a") as lf:
                lf.write(json.dumps(entry) + "\n")

            logger.info(f"  epoch {epoch+1:3d} | KS={ks:.4f} | score={score:.4f} | best_score={best_score:.4f} "
                        f"| ACF_MAE={acf_mae:.4f} | spread={spread:.1f}ms "
                        f"| G={g_mean:.3f} D={d_mean:.3f} "
                        f"| below_target={below_target_cnt}/5")

            if score < best_score:
                best_ks    = ks
                best_score = score
                no_improve = 0
                # Save best checkpoint
                torch.save(G.state_dict(), g_path)
                torch.save(D.state_dict(), d_path)
                logger.info(f"  ✓ Best checkpoint saved (KS={ks:.4f}, spread={spread:.1f}ms, score={score:.4f})")
            else:
                no_improve += 1

            if below_target_cnt >= 5:
                stop_epoch = epoch + 1
                logger.success(f"  KS < {target_ks} for 5 consecutive evals — stopping at epoch {stop_epoch}")
                break
            if no_improve >= patience:
                stop_epoch = epoch + 1
                logger.info(f"  Early stop (patience) at epoch {stop_epoch}, best KS={best_ks:.4f}")
                break

    # Final per-state spread with best model (load best checkpoint)
    if g_path.exists():
        G.load_state_dict(torch.load(g_path, map_location=device))
    G.eval()
    spread, state_medians = _compute_per_state_spread(G, noise_dim, device)

    return {
        "name":          cfg["name"],
        "noise_dim":     noise_dim,
        "hidden_size":   hidden_size,
        "ratio":         cfg["ratio"],
        "best_ks":       best_ks,
        "stop_epoch":    stop_epoch,
        "state_spread_ms": spread,
        "state_medians_ms": {
            name: round(med, 1)
            for name, med in zip(HMM_STATE_NAMES, state_medians)
        },
        "ks_history":    ks_history,
    }


@click.command()
@click.option("--data",       default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--epochs",     default=200,  show_default=True, help="Max training epochs per config")
@click.option("--batch-size", default=64,   show_default=True)
@click.option("--eval-every", default=10,   show_default=True)
@click.option("--patience",   default=15,   show_default=True)
@click.option("--quick",      is_flag=True, help="Quick mode: 50 epochs, skip slow configs")
@click.option("--full",       is_flag=True, help="Full mode: 300 epochs, all configs")
@click.option("--output",     default="results/ablation_noise_dim.json", show_default=True)
@click.option("--configs",    default="ABCDE", show_default=True,
              help="Which configs to run (e.g. 'BCD' for configs B, C, D only)")
def main(data, epochs, batch_size, eval_every, patience, quick, full, output, configs):
    if quick:
        epochs     = 50
        eval_every = 5
        patience   = 10
        logger.info("Quick mode: 50 epochs per config")
    elif full:
        epochs     = 300
        patience   = 20
        logger.info("Full mode: 300 epochs per config")

    device_str = _best_device()
    logger.info(f"Device: {device_str}")
    logger.info(f"Dataset: {data}")
    logger.info(f"Epochs: {epochs} | batch={batch_size} | eval_every={eval_every}")

    # Filter configs
    config_letters = [c for c in "ABCDEF" if c in configs.upper()]
    selected = [cfg for cfg, letter in zip(CONFIGS, "ABCDEF") if letter in config_letters]
    logger.info(f"Running configs: {[c['name'] for c in selected]}")

    Path("logs").mkdir(exist_ok=True)
    results = []
    for cfg in selected:
        safe_name = cfg["name"].replace(":", "").replace(" ", "_").replace(",", "")
        try:
            result = train_config(
                cfg, data, epochs, batch_size, device_str,
                eval_every=eval_every, patience=patience,
                checkpoint_dir="models",
                log_file=f"logs/train_{safe_name}.jsonl",
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Config {cfg['name']} failed: {e}")
            results.append({"name": cfg["name"], "error": str(e)})

    # ── Print results table ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("NOISE_DIM ABLATION RESULTS")
    print("=" * 90)
    print(f"{'Config':<32} {'noise_dim':>10} {'hidden':>8} {'ratio':>18} "
          f"{'Best KS (↓)':>12} {'Stop epoch':>11} {'State spread (↑)':>17}")
    print("-" * 90)

    baseline_ks = None
    for r in results:
        if "error" in r:
            print(f"{r['name']:<32}  ERROR: {r['error']}")
            continue
        ks      = r["best_ks"]
        spread  = r["state_spread_ms"]
        delta   = f"({ks - baseline_ks:+.4f})" if baseline_ks is not None else "(baseline)"
        if baseline_ks is None:
            baseline_ks = ks
        print(f"{r['name']:<32} {r['noise_dim']:>10} {r['hidden_size']:>8} "
              f"{r['ratio']:>18} {ks:>10.4f} {delta:>12} {r['stop_epoch']:>10} "
              f"{spread:>14.1f}ms")

    print("\nState-conditional median IKIs (ms) per config:")
    print(f"{'Config':<32}" + "".join(f" {s:>11}" for s in HMM_STATE_NAMES))
    print("-" * (32 + 11 * 6))
    for r in results:
        if "error" in r or "state_medians_ms" not in r:
            continue
        row = f"{r['name']:<32}"
        for s in HMM_STATE_NAMES:
            row += f" {r['state_medians_ms'].get(s, 0):>11.1f}"
        print(row)

    print("\nNote: 'State spread' = std(per-state median IKIs). Higher = better conditioning.")
    print("      Baseline = Config B (noise_dim=16, current paper setting).\n")

    # ── Save JSON results ──────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
