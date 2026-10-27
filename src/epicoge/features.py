"""Residue scalar features and official-candidate chain annotations."""
import torch
from torch.nn import functional as F

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


AA_PROPERTIES = {
    "A": (0.40, 0, 0, 0, 1), "C": (0.56, 0, 0, 0, 1), "D": (-0.78, -1, 1, 0, 1),
    "E": (-0.78, -1, 1, 0, 0), "F": (0.62, 0, 0, 1, 0), "G": (-0.09, 0, 0, 0, 1),
    "H": (-0.71, 0.1, 1, 1, 0), "I": (1.00, 0, 0, 0, 0), "K": (-0.86, 1, 1, 0, 0),
    "L": (0.84, 0, 0, 0, 0), "M": (0.42, 0, 0, 0, 0), "N": (-0.78, 0, 1, 0, 1),
    "P": (-0.36, 0, 0, 0, 1), "Q": (-0.78, 0, 1, 0, 0), "R": (-1.00, 1, 1, 0, 0),
    "S": (-0.18, 0, 1, 0, 1), "T": (-0.16, 0, 1, 0, 1), "V": (0.93, 0, 0, 0, 1),
    "W": (-0.20, 0, 0, 1, 0), "Y": (-0.29, 0, 1, 1, 0),
}


def residue_features(sequence: str, indices: torch.Tensor, sasa: torch.Tensor,
                     backbone_mask: torch.Tensor) -> torch.Tensor:
    selected = [sequence[int(index)] for index in indices]
    one_hot = F.one_hot(torch.tensor([AA_ORDER.index(amino) for amino in selected]), len(AA_ORDER)).float()
    properties = torch.tensor([AA_PROPERTIES[amino] for amino in selected], dtype=torch.float32)
    return torch.cat((one_hot, properties, (sasa / 200.0)[:, None], (~backbone_mask).float()), dim=-1)


def antibody_annotations(sequences: dict[str, str], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    chain_types, cdr_types, cursor = [], [], 0
    for chain_index, chain in enumerate(("H", "L")):
        selected = mask[cursor: cursor + len(sequences[chain])].nonzero(as_tuple=False).squeeze(-1).tolist()
        second_boundary = 76 if chain == "H" else 72
        for index in selected:
            cdr = 0 if index + 1 <= 42 else 1 if index + 1 <= second_boundary else 2
            chain_types.append(chain_index)
            cdr_types.append(cdr)
        cursor += len(sequences[chain])
    return torch.tensor(chain_types), torch.tensor(cdr_types)
