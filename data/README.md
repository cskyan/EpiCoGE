# AsEP inputs

Download **asep-dataset.zip** from the fixed AsEP **v1.2.0** record:
https://doi.org/10.5281/zenodo.11495514

The record lists this archive as approximately 2.6 GB, with MD5
`ad0055009ed45bdca7453f1fe58d953b`. Use this version rather than the "newer version"
link. Extract the archive and find the directory containing:

```text
asepv1.2.0/
  asepv1-AbDb-IDs.txt
  split_dict.pt
  interim_graphs/
    .../<complex_id>.pt
  structures/
    .../<complex_id>.pdb
```

Pass that directory to `scripts/prepare_data.py --raw-root`. The scripts read
official Torch graph dictionaries directly; installing the AsEP interface or
PyTorch Geometric is unnecessary for these files.

`manifests/group_train_validation.csv` and `ratio_train_validation.csv` record the
paper's training/validation order using `role,index,complex_id`. Indices refer to
the zero-based positions in `asepv1-AbDb-IDs.txt`. They are checked against the
downloaded official split. Test membership is reconstructed from the same official
split file and sorted by its original index. Each split has 1,383 Train, 170 Val
and 170 Test complexes.

The generated files, excluded from Git, are:

| Path | Contents |
|---|---|
| `dataset.json` | Raw-file paths, hashes, sequence identities and dataset version |
| `manifests/group.csv`, `ratio.csv` | Complete official split memberships |
| `sequences.json` | Full antigen sequences keyed by SHA256 |
| `embeddings/<sequence_sha256>.pt` | Full-sequence ESM2 layer-33 FP32 embedding |
| `embedding_metadata.json` | Model, dimensions, versions and pretrained-file hash |
| `geometry/<split>/<role>/<complex_id>.npz` | Coordinates, backbone vectors/masks, SASA and edges |
| `processed/<split>/<role>/<complex_id>.pt` | Training-ready inputs and separate targets |
| `prepared/<split>_<role>.json` | Per-sample hashes and preprocessing metadata |

Each processed sample has three keys: `id`, `inputs` and `targets`.

| Field | Definition |
|---|---|
| `inputs.ab` | Official AntiBERTy/IgFold-cache values at `seqres2cdr`, shape n×512 |
| `inputs.ag` | Complete antigen-sequence ESM2-650M layer-33 values at `seqres2surf`, shape m×1280 |
| `inputs.ag_candidate_indices` | Zero-based original antigen-sequence indices |
| `inputs.ab_chain_type`, `ab_cdr_type` | H/L identity and sequence-position CDR-type annotation |
| `inputs.{ab,ag}_scalar_features` | 20 amino-acid indicators, five properties, SASA/200 and four atom-missingness flags |
| `inputs.{ab,ag}_coordinates` | Independently centered/rotated C-alpha coordinates in angstroms |
| `inputs.{ab,ag}_vectors`, `vector_mask` | Three N/CA/C/O backbone vectors and validity masks |
| `inputs.{ab,ag}_spatial_edges` | Directed edges at distance <10 angstroms, at most 16 neighbors per source |
| `inputs.{ab,ag}_sequence_edges` | Bidirectional consecutive original residue edges within each chain |
| `targets.contact` | Official n×m binary contact matrix |
| `targets.epitope` | Official candidate node labels, checked against columnwise contact maxima |

H CDR-type intervals are sequence positions ≤42, 43–76 and >76; L intervals are
≤42, 43–72 and >72. Positions are 1-based within the chain and apply only to the
official CDR candidates. They are not a replacement for an antibody numbering scheme.

Both antibody chains form one partner for SASA; antigen chains form the other.
All opposite-partner chains are removed before SASA. Biopython's Shrake–Rupley
implementation uses a 1.4-angstrom probe and 100 points. Graph mapping preserves
official candidate order. Spatial and sequence edge types remain distinct even
when they connect the same nodes. Missing backbone vectors are zero, with masks.

The official AsEP contact/epitope labels are preserved, including their original
4.5-angstrom contact definition. Geometric preprocessing does not regenerate them.
Labels are never included in the forward input dictionary. Registration and
preprocessing can read the fixed Test records; the training program loads only
Train/Val. Test evaluation is a separate command with the saved Val threshold.
