#!/usr/bin/env python3
"""
HMM State Re-alignment
======================
After GaussianHMM EM training, state indices are assigned arbitrarily by
likelihood optimization — not by semantic meaning.
This script:
  1. Loads the trained HMM model (models/hmm_model.pkl)
  2. Sorts states by emission mean (ascending IKI)
  3. Re-assigns state indices to match semantic labels:
       FAST(2) < CORRECTION(4) < NORMAL(0) < ERROR(3) < SLOW(1) < PAUSE(5)
  4. Permutes all HMM parameters (means, covars, transmat, startprob)
  5. Saves the re-aligned HMM (backs up the original)
  6. Re-decodes the JSONL dataset with the corrected HMM and updates
     context vector slots [6:12] (HMM state one-hot)

Expected state mean order after alignment:
  Index  Name        Target mean  Notes
    0    NORMAL      ~120 ms      Steady comfortable typing
    1    SLOW        ~250 ms      Before complex expressions
    2    FAST        ~60 ms       Muscle-memory bigram bursts
    3    ERROR       ~180 ms      Pre-error irregular state
    4    CORRECTION  ~100 ms      Backspace + retype
    5    PAUSE       ~2000 ms     Deliberate stop

Usage:
  python scripts/realign_hmm_states.py
  python scripts/realign_hmm_states.py --model-dir models --data data/raw/keystroke_samples.jsonl
  python scripts/realign_hmm_states.py --dry-run   # inspect only, no file changes
"""
import json
import pickle
import sys
import shutil
import numpy as np
from pathlib import Path
import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

# Semantic labels in state-index order
STATE_NAMES        = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]
# Target ascending-mean semantic order (smallest mean → largest mean)
# i.e., when we sort states by mean, position 0 should become FAST(2),
# position 1 should become CORRECTION(4), etc.
SEMANTIC_ORDER     = [2, 4, 0, 3, 1, 5]   # new_index for rank 0,1,2,3,4,5
# Reference target means (ms) for sanity check only
TARGET_MEANS_MS    = {0: 120, 1: 250, 2: 60, 3: 180, 4: 100, 5: 2000}


def _permute_hmm(model, perm: np.ndarray):
    """
    Return a copy of model with state parameters permuted by perm.
    perm[old_idx] = new_idx
    """
    import copy
    m = copy.deepcopy(model)
    n = len(perm)

    # Inverse permutation: new_idx → old_idx
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)

    # means_: (n_states, n_features)
    m.means_ = model.means_[inv]

    # covars_: use private _covars_ to bypass hmmlearn property setter validation
    m._covars_ = model._covars_[inv]

    # startprob_: (n_states,)
    m.startprob_ = model.startprob_[inv]

    # transmat_: (n_states, n_states)
    m.transmat_ = model.transmat_[np.ix_(inv, inv)]

    return m


def _build_perm(model) -> np.ndarray:
    """
    Build the permutation array perm where perm[old_idx] = new_idx.

    Steps:
      1. Sort old states by ascending emission mean → rank order
      2. Assign new_idx = SEMANTIC_ORDER[rank]
    """
    means = model.means_.flatten()          # (n_states,)
    sort_order = np.argsort(means)          # sort_order[rank] = old_idx with rank-th smallest mean
    perm = np.empty(len(means), dtype=int)
    for rank, old_idx in enumerate(sort_order):
        perm[old_idx] = SEMANTIC_ORDER[rank]
    return perm


def _decode_and_update_context(jsonl_path: Path, model, out_path: Path,
                                batch_size: int = 5000):
    """
    Re-Viterbi-decode every sequence in the JSONL with the re-aligned HMM
    and update context slots [6:12] in-place.
    """
    logger.info(f"Re-decoding {jsonl_path} with re-aligned HMM...")

    lines_written = 0
    lines_total   = 0

    with open(jsonl_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            lines_total += 1
            try:
                record = json.loads(line.strip())
                timings = record.get("timings", [])
                if not timings:
                    fout.write(line)
                    continue

                # IKI values (delay_ms) — first element of each timing triple
                delays_ms = np.array(
                    [t[0] if isinstance(t, list) else t for t in timings],
                    dtype=np.float32
                ).reshape(-1, 1)

                # Viterbi decode with re-aligned HMM
                state_seq = model.predict(delays_ms)   # (seq_len,)

                # Update or create context vectors
                ctx = record.get("context", None)
                seq_len = len(timings)

                if ctx and len(ctx) >= seq_len:
                    ctx_arr = [list(c) for c in ctx[:seq_len]]
                else:
                    ctx_arr = [[0.0] * 32 for _ in range(seq_len)]

                for t, state in enumerate(state_seq):
                    # Clear old HMM one-hot (slots 6-11)
                    for s in range(6):
                        ctx_arr[t][6 + s] = 0.0
                    # Set new one-hot
                    if 0 <= state < 6:
                        ctx_arr[t][6 + int(state)] = 1.0

                record["context"] = ctx_arr
                fout.write(json.dumps(record) + "\n")
                lines_written += 1

            except Exception as e:
                # Keep original line on error
                fout.write(line)

            if lines_total % 10000 == 0:
                logger.info(f"  Processed {lines_total:,} sequences...")

    logger.info(f"Done. {lines_written:,}/{lines_total:,} sequences updated.")


@click.command()
@click.option("--model-dir",  default="models",                           show_default=True)
@click.option("--data",       default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--dry-run",    is_flag=True, help="Print alignment only; do not write files.")
@click.option("--skip-recode",is_flag=True, help="Re-align HMM only; skip JSONL re-coding.")
def main(model_dir, data, dry_run, skip_recode):
    model_path = Path(model_dir) / "hmm_model.pkl"
    if not model_path.exists():
        logger.error(f"HMM model not found at {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    means = model.means_.flatten()
    logger.info("Current HMM emission means (ms):")
    for i, m in enumerate(means):
        logger.info(f"  State {i} ({STATE_NAMES[i]}): {m:.1f} ms")

    # Build permutation
    perm = _build_perm(model)
    sort_order = np.argsort(means)

    print("\n" + "="*60)
    print("PROPOSED RE-ALIGNMENT")
    print("="*60)
    print(f"{'Rank':>5} {'Old idx':>8} {'Old name':>12} {'Mean (ms)':>10}"
          f"  →  {'New idx':>8} {'New name':>12}")
    print("-"*60)
    for rank, old_idx in enumerate(sort_order):
        new_idx  = SEMANTIC_ORDER[rank]
        old_name = STATE_NAMES[old_idx]
        new_name = STATE_NAMES[new_idx]
        target   = TARGET_MEANS_MS[new_idx]
        print(f"  {rank:>3}  {old_idx:>8}  {old_name:>12}  {means[old_idx]:>9.1f}"
              f"  →  {new_idx:>8}  {new_name:>12}  (target ~{target}ms)")

    print()
    if dry_run:
        logger.info("Dry-run: no files written.")
        return

    # ── Save re-aligned HMM ───────────────────────────────────────────────────
    backup_path = Path(model_dir) / "hmm_model_original.pkl"
    if not backup_path.exists():
        shutil.copy(model_path, backup_path)
        logger.info(f"Original HMM backed up to {backup_path}")

    aligned_model = _permute_hmm(model, perm)

    with open(model_path, "wb") as f:
        pickle.dump(aligned_model, f)
    logger.success(f"Re-aligned HMM saved to {model_path}")

    # Verify
    new_means = aligned_model.means_.flatten()
    logger.info("Verified re-aligned emission means (ms):")
    for i, m in enumerate(new_means):
        target = TARGET_MEANS_MS[i]
        ok = "OK" if abs(m - target) < max(target * 2, 200) else "CHECK"
        logger.info(f"  State {i} ({STATE_NAMES[i]}): {m:.1f} ms  (target ~{target}ms)  [{ok}]")

    # ── Re-decode JSONL dataset ────────────────────────────────────────────────
    if skip_recode:
        logger.info("--skip-recode: JSONL dataset not updated.")
        return

    data_path = Path(data)
    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}; skipping JSONL re-coding.")
        return

    # Back up original JSONL
    backup_jsonl = data_path.with_suffix(".jsonl.bak")
    if not backup_jsonl.exists():
        shutil.copy(data_path, backup_jsonl)
        logger.info(f"Original dataset backed up to {backup_jsonl}")

    tmp_path = data_path.with_suffix(".jsonl.tmp")
    _decode_and_update_context(data_path, aligned_model, tmp_path)
    tmp_path.replace(data_path)
    logger.success(f"Dataset context vectors updated: {data_path}")
    logger.info("Re-alignment complete. You can now retrain the GAN.")


if __name__ == "__main__":
    main()
