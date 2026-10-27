"""Partner-separated geometry with official sequence-to-structure mapping."""
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import subprocess
import tempfile
import numpy as np

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def parse_pdb(path: Path) -> dict[str, list[dict]]:
    """Preserve ATOM residue order, including insertion codes preceding their base number."""
    chains = defaultdict(dict)
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[16:17] not in (" ", "A"):
                continue
            chain = line[21:22].strip()
            key = (line[22:26].strip(), line[26:27].strip())
            residue = chains[chain].setdefault(
                key,
                {"name": AA3.get(line[17:20].strip(), "X"), "atoms": {}},
            )
            atom = line[12:16].strip()
            residue["atoms"].setdefault(
                atom,
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64),
            )
    return {chain: list(residues.values()) for chain, residues in chains.items()}


def _aligned_fasta(text: str) -> list[str]:
    sequences = []
    current = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    if len(sequences) != 2:
        raise ValueError(f"Expected two aligned FASTA sequences, got {len(sequences)}")
    return sequences


def _clustalo_mapping(seqres: str, observed: str, executable: Path) -> dict[int, int]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        input_path = root / "input.fasta"
        output_path = root / "aligned.fasta"
        input_path.write_text(f">seqres\n{seqres}\n>observed\n{observed}\n", encoding="ascii")
        subprocess.run(
            [str(executable), "-i", str(input_path), "-o", str(output_path), "--outfmt=fasta", "--force", "--auto"],
            check=True,
            capture_output=True,
            text=True,
        )
        aligned_seqres, aligned_observed = _aligned_fasta(output_path.read_text(encoding="ascii"))
    if "-" in aligned_seqres:
        raise ValueError("Clustal Omega inserted a gap into SEQRES; official mapping assumption violated")
    mapping = {}
    seq_index = observed_index = 0
    for seq_char, observed_char in zip(aligned_seqres, aligned_observed):
        if seq_char != "-":
            if observed_char != "-":
                mapping[seq_index] = observed_index
            seq_index += 1
        if observed_char != "-":
            observed_index += 1
    return mapping


def map_sequence(seqres: str, residues: list[dict], aligner: Path | None = None) -> dict[int, int]:
    observed = "".join(residue["name"] for residue in residues)
    if aligner is not None:
        return _clustalo_mapping(seqres, observed, aligner)
    matcher = SequenceMatcher(None, seqres, observed, autojunk=False)
    mapping = {
        seq_index + offset: observed_index + offset
        for seq_index, observed_index, size in matcher.get_matching_blocks()
        for offset in range(size)
    }
    if len(mapping) != len(observed):
        raise ValueError(f"Observed residues are not an exact subsequence of SEQRES: mapped={len(mapping)}/{len(observed)}")
    return mapping


def random_rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def _mask(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=bool)


def _side(
    chains: dict[str, list[dict]],
    sequences: dict[str, str],
    mask,
    rotation: np.ndarray,
    aligner: Path | None,
    sasa: dict[str, list[float]] | None = None,
    center_override: np.ndarray | None = None,
) -> dict:
    mask = _mask(mask)
    offsets = {}
    cursor = 0
    for chain, sequence in sequences.items():
        offsets[chain] = cursor
        cursor += len(sequence)
    if cursor != len(mask):
        raise ValueError(f"Mask length {len(mask)} does not match sequence length {cursor}")
    all_ca = []
    selected_residues = []
    selected_chains = []
    selected_sequence_indices = []
    selected_observed_indices = []
    selected_sasa = []
    for chain, sequence in sequences.items():
        residues = chains[chain]
        mapping = map_sequence(sequence, residues, aligner)
        for residue in residues:
            if "CA" in residue["atoms"]:
                all_ca.append(residue["atoms"]["CA"])
        inverse = {seq_index: observed_index for seq_index, observed_index in mapping.items()}
        local_mask = mask[offsets[chain] : offsets[chain] + len(sequence)]
        for seq_index in np.flatnonzero(local_mask):
            if int(seq_index) not in inverse:
                raise ValueError(f"Selected SEQRES position is missing from structure: {chain}:{seq_index}")
            residue = residues[inverse[int(seq_index)]]
            if "CA" not in residue["atoms"]:
                raise ValueError(f"Selected residue lacks CA: {chain}:{seq_index}")
            selected_residues.append(residue)
            selected_chains.append(chain)
            selected_sequence_indices.append(offsets[chain] + int(seq_index))
            selected_observed_indices.append(inverse[int(seq_index)])
            selected_sasa.append(sasa[chain][inverse[int(seq_index)]] if sasa else 0.0)
    all_ca = np.asarray(all_ca, dtype=np.float64)
    original = np.asarray([residue["atoms"]["CA"] for residue in selected_residues], dtype=np.float64)
    center = all_ca.mean(axis=0) if center_override is None else np.asarray(center_override, dtype=np.float64)
    transformed_all = (all_ca - center) @ rotation.T
    transformed = (original - center) @ rotation.T
    atom_names = ("N", "CA", "C", "O")
    backbone = np.zeros((len(selected_residues), len(atom_names), 3), dtype=np.float64)
    backbone_mask = np.zeros((len(selected_residues), len(atom_names)), dtype=bool)
    for residue_index, residue in enumerate(selected_residues):
        for atom_index, atom_name in enumerate(atom_names):
            if atom_name in residue["atoms"]:
                backbone[residue_index, atom_index] = (residue["atoms"][atom_name] - center) @ rotation.T
                backbone_mask[residue_index, atom_index] = True
    vectors = np.zeros((len(selected_residues), 3, 3), dtype=np.float64)
    vector_mask = np.zeros((len(selected_residues), 3), dtype=bool)
    for vector_index, atom_index in enumerate((0, 2, 3)):
        valid = backbone_mask[:, 1] & backbone_mask[:, atom_index]
        vectors[valid, vector_index] = backbone[valid, atom_index] - backbone[valid, 1]
        vector_mask[:, vector_index] = valid
    sequence_pairs = []
    for left in range(len(selected_residues) - 1):
        right = left + 1
        if (
            selected_chains[left] == selected_chains[right]
            and selected_sequence_indices[right] == selected_sequence_indices[left] + 1
            and selected_observed_indices[right] == selected_observed_indices[left] + 1
        ):
            sequence_pairs.extend(((left, right), (right, left)))
    sequence_edges = np.asarray(sequence_pairs, dtype=np.int64).T if sequence_pairs else np.empty((2, 0), dtype=np.int64)
    return {
        "original": original,
        "coordinates": transformed,
        "center": center,
        "rotation": rotation,
        "transformed_all_ca_mean": transformed_all.mean(axis=0),
        "chains": selected_chains,
        "sequence_indices": np.asarray(selected_sequence_indices, dtype=np.int64),
        "backbone": backbone,
        "backbone_mask": backbone_mask,
        "vectors": vectors,
        "vector_mask": vector_mask,
        "sequence_edges": sequence_edges,
        "sasa": np.asarray(selected_sasa, dtype=np.float64),
    }


def monomer_sasa(path: Path, chain_ids) -> dict[str, list[float]]:
    """Shrake-Rupley residue SASA after removing every non-partner chain."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    model = next(PDBParser(QUIET=True).get_structure(path.stem, path).get_models())
    keep = set(chain_ids)
    for chain in list(model):
        if chain.id not in keep:
            model.detach_child(chain.id)
    ShrakeRupley().compute(model, level="R")
    return {
        chain: [float(residue.sasa) for residue in model[chain]
                if residue.id[0] == " "]
        for chain in keep
    }


def build_partner_separated(graph: dict, pdb_path: Path | dict[str, Path], seed: int, aligner: Path | None = None,
                            partners: tuple[str, ...] = ("antibody", "antigen"),
                            coordinate_frame: str = "separate") -> dict:
    if coordinate_frame not in ("separate", "shared"):
        raise ValueError("coordinate_frame must be separate or shared")
    paths = ({"antibody": pdb_path, "antigen": pdb_path}
             if isinstance(pdb_path, Path) else pdb_path)
    ab_sequences = graph["seqres"]["ab"]
    ag_sequences = graph["seqres"]["ag"]
    if set(ab_sequences) != {"H", "L"} or not ag_sequences:
        raise ValueError("Expected H+L antibody and at least one antigen chain")
    parsed = {name: parse_pdb(paths[name]) for name in partners}
    shared_center = None
    shared_rotation = None
    if coordinate_frame == "shared":
        all_ca = [residue["atoms"]["CA"] for chains in parsed.values()
                  for chain in chains.values() for residue in chain if "CA" in residue["atoms"]]
        if not all_ca:
            raise ValueError("shared frame requires at least one CA atom")
        shared_center = np.asarray(all_ca, dtype=np.float64).mean(axis=0)
        shared_rotation = random_rotation(seed)
    result = {}
    if "antibody" in partners:
        result["antibody"] = _side(parsed["antibody"], ab_sequences,
                                   graph["mapping"]["ab"]["seqres2cdr"],
                                   shared_rotation if shared_rotation is not None else random_rotation(seed), aligner,
                                   monomer_sasa(paths["antibody"], ab_sequences), shared_center)
    if "antigen" in partners:
        result["antigen"] = _side(parsed["antigen"], ag_sequences,
                                  graph["mapping"]["ag"]["seqres2surf"],
                                  shared_rotation if shared_rotation is not None else random_rotation(seed + 1), aligner,
                                  monomer_sasa(paths["antigen"], ag_sequences), shared_center)
    if not result or any(name not in ("antibody", "antigen") for name in partners):
        raise ValueError("partners must contain antibody and/or antigen")
    return result


def spatial_edges(coordinates: np.ndarray, radius: float = 10.0, max_neighbors: int = 16) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0 or max_neighbors <= 0:
        raise ValueError("radius and max_neighbors must be positive")
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    pairs = []
    values = []
    for source, row in enumerate(distances):
        candidates = np.flatnonzero((row < radius) & (np.arange(len(row)) != source))
        order = np.lexsort((candidates, np.round(row[candidates], decimals=8)))[:max_neighbors]
        for target in candidates[order]:
            pairs.append((source, int(target)))
            values.append(float(row[target]))
    edges = np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    return edges, np.asarray(values, dtype=np.float64)


def add_graph_geometry(side: dict, radius: float = 10.0, max_neighbors: int = 16) -> dict:
    edges, distances = spatial_edges(side["coordinates"], radius, max_neighbors)
    side["spatial_edges"] = edges
    side["spatial_distances"] = distances
    return side
