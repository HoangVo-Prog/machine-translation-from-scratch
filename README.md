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

