# EpiCoGE

EpiCoGE predicts antibody-specific epitopes from antibody–antigen sequences and
separately encoded monomer structures. Local scalar–vector geometry is followed
by self-attention, simultaneous bidirectional cross-attention, and contact-assisted
residue-pair information feeding a learnable antigen node head. A relative docking
pose is not required as input.

This repository provides the **primary AsEP Group and Ratio training/evaluation
experiments**. EpiCoGE is trained from random initialization. Frozen AntiBERTy and
ESM2 representations supply input features. No trained EpiCoGE weights are bundled.

## Installation

The paper training environment used Linux, Python 3.12.10, PyTorch 2.11.0 with
CUDA 12.8, NumPy 2.5.1, Biopython 1.88, and an NVIDIA A100-SXM4-80GB. The supplied
requirements pin these Python-library versions; Clustal Omega 1.2.4 is required
for the sequence-to-structure mapping. Use a CUDA GPU supporting BF16 for the
paper precision path.

From the repository root, with Conda installed:

```bash
conda env create -f environment.yml
conda activate epicoge
```

Alternatively, in a Python 3.12 environment with Clustal Omega 1.2.4 on PATH:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
clustalo --version
```

The `pyproject.toml` dependency ranges permit compatibility checks with other
environments; `requirements.txt` records the paper versions. CPU/FP32 execution is
available explicitly for inspection, and is different from the paper BF16 setting.

## Data and feature preparation

Download the fixed [AsEP v1.2.0 dataset](https://doi.org/10.5281/zenodo.11495514)
and extract it. Use the folder containing `split_dict.pt` and
`asepv1-AbDb-IDs.txt` as `--raw-root`. See [data/README.md](data/README.md) for
the archive checksum, expected structure, candidate axes and generated file schema.

```bash
python scripts/prepare_data.py --raw-root /path/to/asepv1.2.0 --output data
python scripts/extract_features.py --data-root data --device cuda:0
python scripts/build_geometry.py --data-root data --split group --roles train validation
python scripts/build_geometry.py --data-root data --split ratio --roles train validation
```

The antibody side reuses the official AsEP 512-dimensional AntiBERTy representations
under the `igfold` cache key. Antigen features are generated with the public
`esm2_t33_650M_UR50D` model, **layer 33, 1280 dimensions, full antigen sequence,
FP32, no truncation**, followed by official candidate selection. The smaller
antigen embeddings already present in AsEP are not used. ESM2 pretrained
parameters are downloaded from the official ESM host into the local Torch cache,
checked against the file hashes used for the paper, and loaded with fair-esm.
They are not trained as part of EpiCoGE. Set `TORCH_HOME` to relocate that cache if needed.

Feature extraction reuses valid sequence caches. Geometry preparation records
hashes, checks candidates/labels, and refuses to replace a completed prepared
role. Pass `--clustalo /path/to/clustalo` when it is not on PATH. The primary
experiment uses official bound structures with the opposite partner removed,
independent coordinate frames, and unchanged official candidate masks and labels.

## Train the primary experiments

```bash
python scripts/train.py --config configs/group.yaml --data-root data --output runs --device cuda:0
python scripts/train.py --config configs/ratio.yaml --data-root data --output runs --device cuda:0
```

Each command sequentially trains seeds **20260908, 20260919 and 20260920**.
To run one seed, add `--seed 20260908`. Existing run directories are not overwritten.
The two protocols use separate models with the same architecture and training recipe.

The configuration fixes hidden width 256, 16 vector channels, two local layers,
four attention heads, dropout 0.1 and **5,019,682 trainable parameters**, excluding
the frozen language models. Node positive weight is 1; the contact loss coefficient
is 0.1. Contact positive weight is computed once from the full Train negative/positive
pair count and checked against the recorded split-specific value.

Training uses AdamW, learning rate 0.0003, weight decay 0.01 excluding biases and
LayerNorm parameters, complex-equal effective batches of 32 with accumulation,
gradient clipping at norm 1, a maximum of 100 epochs and patience 10. Each epoch
uses a seeded random permutation of all Train complexes. The final partial batch
is normalized by its actual size. There is no learning-rate schedule. Geometry,
vector norms, pair/node heads and loss arithmetic remain FP32 under BF16 autocast.
TF32 is disabled and deterministic algorithms are requested.

Every epoch selects a single threshold on **Val** using macro MCC on the 0–1 grid
with step 0.001, preferring the higher threshold on ties. Checkpoint ties retain
the earlier epoch. `runs/<split>/seed_<seed>/best.pt` is generated locally with
the selected threshold; `history.json` records the training/Val trace. Training
never loads the Test samples. 

## Evaluate at the saved Val threshold

Prepare Test inputs separately, then evaluate the selected checkpoints:

```bash
python scripts/build_geometry.py --data-root data --split group --roles test
python scripts/build_geometry.py --data-root data --split ratio --roles test

for split in group ratio; do
  for seed in 20260908 20260919 20260920; do
    python scripts/evaluate.py \
      --checkpoint runs/$split/seed_$seed/best.pt \
      --data-root data --role test --device cuda:0 \
      --output runs/$split/seed_$seed/test
  done
  python scripts/evaluate.py --summaries \
    runs/$split/seed_20260908/test/metrics.json \
    runs/$split/seed_20260919/test/metrics.json \
    runs/$split/seed_20260920/test/metrics.json \
    --output runs/$split/summary
done
```

The loop is Bash syntax. Individual Python commands can also be executed directly.
The evaluator reads the checkpoint's Val threshold; there is no Test threshold
optimization. It writes `metrics.json`, `per_complex.csv` and residue-level
`predictions.json`. Summaries contain the mean and sample SD of the three seeds.

AUROC, MCC, precision, recall and F1 are calculated per complex and macro averaged
over 170 Test complexes. No predicted positives gives precision 0; a zero MCC
denominator gives MCC 0. See [results/README.md](results/README.md) and the two
reference CSVs for the reported results and original per-seed thresholds.

## Repository contents

```text
src/epicoge/    model, geometry, features, data assembly, metrics and runtime
configs/       Group and Ratio primary experiment configurations
scripts/       data registration, feature extraction, geometry, training, evaluation
data/          input documentation and paper Train/Val membership manifests
results/       reference main-experiment metrics
```

Raw data, generated features, run outputs and locally trained checkpoints are
excluded by `.gitignore`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream resources.

## License

EpiCoGE code is released under the [MIT License](LICENSE). Dataset and pretrained
model terms remain with their respective providers.
