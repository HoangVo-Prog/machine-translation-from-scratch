# machine-translation-from-scratch

---

## Suggested Project Structure

```bash
machine-translation-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml              # optional, useful later if you formalize the package
├── configs/
│   ├── default.yaml
│   ├── data.yaml
│   └── train.yaml
├── data/
│   ├── raw/                    # original downloaded datasets
│   ├── interim/                # cleaned or split intermediate files
│   └── processed/              # tokenized / numericalized outputs
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training_debug.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── tokenization.py
│   │   └── vocabulary.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── seq2seq.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── translate.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── logging.py
│   │   └── io.py
│   └── main.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_dataset.py
│   ├── test_tokenization.py
│   └── test_model_shapes.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
└── docs/
    ├── project_plan.md
    └── experiment_notes.md
```

---

## Environment Setup

```bash
git clone https://github.com/HoangVo-Prog/machine-translation-from-scratch.git
cd machine-translation-from-scratch
```

This project supports both:

- **Conda** for environment management
- **venv** for lightweight virtual environments

Dependency installation is done with **uv**.

---

## Option 1: Setup with Conda + uv
```bash
conda create -n mt-scratch python=3.12.12 -y
conda activate mt-scratch
pip install uv
uv pip install -r requirements.txt
```

## Option 2: Setup with venv + uv

```bash
python3.12 -m venv .venv
```


### Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### Install 

```bash
pip install uv
uv pip install -r requirements.txt
```
---

## Best Practices

- keep raw data untouched in `data/raw/`
- store processed outputs separately in `data/processed/`
- save checkpoints under `outputs/checkpoints/`
- separate reusable library code in `src/` from runnable scripts in `scripts/`
- use `tests/` for quick validation of dataset logic, tokenization, and model shapes

---

## Training Command (CLI)

The training logic lives in `src/training/` (`trainer.py`, `optimizer.py`, `losses.py`, `evaluate.py`).
CLI implementation lives in `src/cli/train.py`.
`scripts/train.py` is only a launcher wrapper.

```bash
python -m src.cli.train \
  --model_factory src.factories:build_model \
  --train_dataloader_factory src.factories:build_train_dataloader \
  --eval_dataloader_factory src.factories:build_eval_dataloader \
  --tokenizer_factory src.factories:build_tokenizer \
  --train_file data/processed/train.jsonl \
  --eval_file data/processed/valid.jsonl \
  --output_dir checkpoints \
  --num_epochs 5 \
  --learning_rate 5e-4 \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --label_smoothing 0.1 \
  --warmup_ratio 0.1 \
  --scheduler_type linear \
  --max_grad_norm 1.0 \
  --eval_steps 200 \
  --save_steps 200 \
  --metric_for_best_model eval/bleu \
  --wandb_enabled true \
  --wandb_project machine-translation
```

Replace `src.factories:*` with real callable paths in your project.

Equivalent command via launcher:

```bash
python scripts/train.py --help
```

Optional W&B environment setup:

```bash
# .env or shell
WANDB_API_KEY=your_key_here
WANDB_PROJECT=machine-translation
```
