# AI-EDOL: Privacy-Preserving Synthetic Building Energy Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20280210.svg)](https://doi.org/10.5281/zenodo.20280210)

AI-EDOL generates privacy-preserving synthetic building energy data using a
generative AI model. A decoder-only GPT model is trained on smart-meter
electricity and gas consumption data together with contextual building and
occupant characteristics, and then used to generate high-fidelity synthetic
data that can be shared openly without exposing the underlying records.

This repository is the open-source release of the pipeline. It contains the
model, the data pipeline, and the evaluation framework, and runs end to end on
generated **dummy data** so anyone can reproduce the workflow without access to
the original sensitive dataset.

The "EDOL" in the project name refers to UCL's Energy Demand Observatory and
Laboratory (EDOL) Programme Grant, which maintains the building energy data
resource this work is built around.

## What the pipeline does

The pipeline runs in six stages, orchestrated by `run_pipeline.py`:

| Stage | Description |
|-------|-------------|
| `dummy_data`  | Generate SERL-like dummy data (smart-meter time series + contextual data) |
| `tokeniser`   | Build tokenisers and convert continuous data into discrete tokens |
| `sharding`    | Filter households and create train/validation/test shards |
| `training`    | Train separate electricity and gas GPT models |
| `inference`   | Generate synthetic households from the trained models |
| `evaluation`  | Evaluate the fidelity of the synthetic data against the real data |

Electricity and gas are modelled by two separate GPT models that share the
same contextual inputs. Continuous consumption is discretised with uniform
(equal-width) binning so the models operate on token sequences.

## Requirements

- Python 3.11
- The packages in `requirements.txt` (PyTorch, pandas, numpy, scikit-learn,
  scipy, matplotlib, seaborn, statsmodels, pyyaml)

## Setup

```bash
git clone <repository-url>
cd ai-edol
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

Run the full pipeline end to end on dummy data:

```bash
python run_pipeline.py --config config.yaml
```

All outputs are written to the run directory (`output/` by default):

- `data/dummy_data/`   — generated dummy data
- `data/tokenised_data/` — tokenised data
- `data/shards/`       — train/validation/test shards
- `electricity_best_val_model.pth`, `gas_best_val_model.pth` — trained models
- `generated_data/`    — synthetic data
- `evaluation/`        — fidelity evaluation results and visualisations
- `*_log.jsonl`        — structured logs for each stage

**Runtime.** On a recent multi-core laptop (Apple Silicon, ~14 cores, running
on CPU/MPS) the full demo takes roughly 25 minutes. Synthetic data generation
is by far the slowest stage at ~20 minutes — generation is autoregressive
(~30 seconds per household), so its cost scales with `inference.num_samples` ×
`inference.batch_size`. The other five stages together take around 5 minutes.
A CUDA GPU substantially speeds up training and generation.

Run a subset of stages (each stage uses the outputs of the previous ones, so
earlier stages must have been run already):

```bash
python run_pipeline.py --config config.yaml --stages tokeniser sharding
```

## Configuration

All settings live in a single file, `config.yaml`. Each stage reads its own
section (`dummy_data`, `tokeniser`, `sharding`, `training`, `inference`,
`evaluation`). The `experiment` section controls the run directory and which
stages run. Edit `config.yaml` to change the number of households, model
architecture, training epochs, and so on. Data paths in the config are
resolved relative to the run directory.

## Repository layout

```
ai-edol/
  run_pipeline.py      Pipeline entry point
  config.yaml          Single configuration file
  src/
    models/            GPT model architecture
    preprocessing/     Data loading, datasets, sharding, stratification
    training/          Training loop
    inference/         Synthetic data generation
    evaluation/        Fidelity evaluation framework
    utils/             Dummy data generation, tokeniser, logging, device utils
```

## License

This project is released under the MIT License — see the [LICENSE](LICENSE)
file for details.

## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research
Council (EPSRC) under grant UKRI2708, "AI-EDOL: AI Generated Synthetic Smart
Meter Data".

The GPT model architecture follows the decoder-only transformer design
popularised by Andrej Karpathy's open educational materials, in particular the
"Neural Networks: Zero to Hero" lecture series and the accompanying `nanoGPT`
repository. The implementation here is independent — built on PyTorch's
standard transformer components and extended for the multi-feature, dual-fuel,
contextualised generation task — but those materials were the starting point
for the model design, and we gratefully acknowledge their value as a learning
resource.
