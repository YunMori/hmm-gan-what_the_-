# human_like_coding_typing_ai

> A 4-layer AI pipeline that simulates human-like code typing — complete with realistic timing, fatigue, errors, and auto-corrections.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## Overview

`human_like_coding_typing_ai` is the AI engine behind the **vvs** Swift app. It takes a code string as input and produces a sequence of keystroke events that mimic how a real developer would type — including natural pauses at complex blocks, occasional typos, self-corrections, speed variation from fatigue, and bigram-based acceleration.

**Core technique:** HMM (Hidden Markov Model) + GAN (Generative Adversarial Network) + KLM (Keystroke-Level Model) combined to synthesize timing that is statistically indistinguishable from real human input.

---

## Architecture

```
Input Code (string)
        ↓
┌───────────────────────────────────────┐
│  Layer 1 · Code Buffer & Analysis     │  Dependency extraction, language detection
├───────────────────────────────────────┤
│  Layer 2 · AST Scheduling & KLM       │  Tree-sitter AST → complexity classification
│                                       │  → KLM pause injection → nonlinear routing
├───────────────────────────────────────┤
│  Layer 3 · Dynamics Synthesis         │  HMM state sequence → GAN timing samples
│                                       │  → Fitts' Law + bigram speedup + fatigue
│                                       │  → error generation + correction events
├───────────────────────────────────────┤
│  Layer 4 · Injection                  │  Desktop (pyautogui) · Web (Playwright)
│                                       │  · JSON output (Swift subprocess)
└───────────────────────────────────────┘
        ↓
KeystrokeEvent[] + Stats JSON
```

---

## Features

- **Multi-language support** — Python, JavaScript, TypeScript, Java, Go, Rust (via Tree-sitter)
- **HMM typing states** — 6 states: NORMAL, SLOW, FAST, ERROR, CORRECTION, PAUSE
- **GAN-based timing** — trained generator produces realistic keydown/keyhold/gap sequences
- **Fitts' Law** — keys farther apart on the keyboard take longer to reach
- **Bigram acceleration** — common character pairs (e.g. `th`, `in`) typed faster
- **Fatigue modeling** — speed gradually decreases over long typing sessions
- **Error simulation** — neighbor key, swap, double, omit errors with auto-correction
- **Complexity-aware pauses** — longer pauses before complex AST nodes (classes, nested loops)
- **Dry-run mode** — simulate without any actual keystrokes
- **Swift subprocess integration** — communicate via stdin/stdout JSON

---

## Installation

```bash
git clone https://github.com/YunMori/human_like_coding_typing_ai.git
cd human_like_coding_typing_ai
pip install -r requirements.txt
```

**Optional dependencies** (install only what you need):

```bash
pip install pyautogui      # Desktop injection
pip install playwright     # Web injection
playwright install chromium

pip install pynput         # Keystroke data collection
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `typing.base_wpm` | 65 | Base typing speed |
| `typing.error_rate` | 0.02 | Probability of a typo per character |
| `typing.fatigue_decay_per_char` | 0.0005 | Speed reduction per character typed |
| `llm.provider` | `claude` | Code generation backend (`claude` / `openai`) |
| `hmm.n_states` | 6 | Number of HMM behavioral states |
| `gan.seq_len` | 32 | Keystroke sequence length for GAN |

---

## Usage

### CLI

```bash
# Type code into a desktop application
python main.py run --lang python --target desktop

# Output timing plan as JSON (for Swift integration)
python main.py type-plan --lang python

# Dry-run — simulate without typing
python main.py run --lang python --target desktop --dry-run

# Inject into a web browser
python main.py run --lang javascript --target web --url http://localhost:3000 --selector "#editor"

# Use custom trained models
python main.py run --lang python --model-dir models/
```

### Swift Subprocess Integration

Send a JSON payload via stdin, receive keystroke events via stdout:

```bash
echo '{"code": "def hello():\n    print(\"world\")", "language": "python"}' \
  | python typing_engine_server.py
```

**Input schema:**
```json
{
  "code": "string",
  "language": "python",
  "model_dir": "models",
  "config_path": "config.yaml",
  "seed": 42
}
```

**Output schema:**
```json
{
  "events": [
    { "key": "d", "delay_before_ms": 120, "key_hold_ms": 80, "is_error": false, "is_correction": false }
  ],
  "stats": {
    "total_keystrokes": 47,
    "error_rate": 0.021,
    "effective_wpm": 61.3,
    "total_duration_ms": 8420
  }
}
```

---

## Training (Optional)

Pre-trained weights are not included. To train on your own typing data:

```bash
# 1. Collect real keystroke data (records to data/raw/)
python scripts/collect_keystroke_data.py

# 2. Train the GAN timing model
python main.py train-gan

# 3. Train the HMM model
python main.py train-hmm

# 4. Evaluate model quality
python scripts/benchmark.py
```

---

## Project Structure

```
.
├── main.py                        # CLI entry point (click)
├── typing_engine_server.py        # Swift subprocess interface
├── config.yaml                    # Model & timing parameters
├── requirements.txt
├── core/
│   ├── pipeline.py                # 4-layer orchestrator
│   └── session.py                 # SessionConfig / SessionResult
├── layer1_codegen/
│   ├── code_buffer.py             # Code + metadata container
│   └── dependency_extractor.py   # Multi-language static analysis
├── layer2_scheduler/
│   ├── scheduler.py               # Layer 2 orchestrator
│   ├── ast_parser.py              # Tree-sitter AST parsing
│   ├── block_classifier.py        # Complexity classification
│   ├── klm_scheduler.py           # KLM pause calculation
│   ├── pause_injector.py          # Pause injection
│   ├── nonlinear_router.py        # Back-visit routing
│   └── typing_plan.py             # TypingPlan / TypingSegment
├── layer3_dynamics/
│   ├── timing_synthesizer.py      # Core synthesis engine
│   ├── hmm_engine.py              # HMM state machine
│   ├── error_generator.py         # Typo simulation
│   ├── correction_engine.py       # Auto-correction events
│   ├── fatigue_model.py           # Speed decay over time
│   ├── bigram_model.py            # Character-pair speedup
│   ├── keyboard_layout.py         # Fitts' Law distance
│   └── gan/                       # GAN model (generator, discriminator, trainer)
├── layer4_injection/
│   ├── desktop_injector.py        # pyautogui injection
│   ├── web_injector.py            # Playwright injection
│   ├── json_output_injector.py    # JSON output
│   └── injector_factory.py
├── data/
│   ├── bigram_frequencies.json    # Pre-computed bigram stats
│   ├── raw/                       # Collected keystroke data
│   └── processed/
├── models/                        # Trained model weights
├── scripts/
│   ├── collect_keystroke_data.py
│   ├── train_gan.py
│   ├── train_hmm.py
│   └── benchmark.py
└── tests/
```

---

## License

MIT
