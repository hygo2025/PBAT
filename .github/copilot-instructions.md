# Copilot instructions

## Build, test, lint
- Install dependencies: `pip install -r requirements.txt` (or `make install` to create the pyenv/venv setup).
- Run training/fit via LightningCLI: `python run.py fit --config src/configs/retail.yaml`
- Format code: `make beautify` (runs `black src`)

## High-level architecture
- `run.py` defines `PBATLightningCLI`, wiring `RecModel` and `RecDataModule`; YAML configs provide `model`, `data`, `trainer`, and `optimizer` settings.
- `src/model.py` implements the Lightning `RecModel`, which wraps the BERT-style backbone (`src/models/bert4rec.py`) and the Wasserstein prediction head (`src/models/heads.py`).
- `src/datamodule.py` builds datasets via `dataset_factory` and instantiates `RecDataloader` for train/val splits.
- `src/datasets/*.py` load raw tab-separated files from `data/<dataset>.txt` with columns `uid`, `sid`, `behavior`, `timestamp`; preprocessing materializes `data/preprocessed/<dataset>-.../dataset.pkl`.
- `src/dataloaders/rec_dataloader.py` constructs train/eval datasets; negative sampling lives under `src/dataloaders/negative_samplers`.

## Key conventions
- `dataset_code` in configs must match the keys in `src/datasets/__init__.py` (`retail`, `yelp`, `ijcai`, `grupozap`).
- `target_behavior` is the behavior evaluated in leave-one-out splitting; users are only evaluated if their last event matches the target.
- IDs are 1-indexed with `0` reserved for padding; the mask token is `num_items + 1` in `RecTrainDataset`.
- When `multi_behavior` is false, preprocessing filters to only `target_behavior`.
- Negative samples are cached per dataset under the preprocessed folder (e.g., `random-sample_size99.pkl`).
