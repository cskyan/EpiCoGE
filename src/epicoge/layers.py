"""Within-monomer geometric message primitives."""
import torch
from torch import nn
from torch.nn import functional as F

class _GVPMessage(nn.Module):
    """One rotation-equivariant scalar-vector message layer within a monomer."""

    def __init__(self, hidden: int, vector_hidden: int):
        super().__init__()
        self.vector_message = nn.Parameter(torch.empty(2 * vector_hidden, vector_hidden + 1))
        self.vector_out = nn.Parameter(torch.empty(vector_hidden, 2 * vector_hidden))
        nn.init.xavier_uniform_(self.vector_message)
        nn.init.xavier_uniform_(self.vector_out)
        self.message = nn.Sequential(nn.Linear(2 * hidden + 2 * vector_hidden + 2, hidden),
                                     nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Linear(hidden, vector_hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, scalar: torch.Tensor, vectors: torch.Tensor, coordinates: torch.Tensor,
                edges: torch.Tensor, distances: torch.Tensor,
                edge_type: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edges
        direction = F.normalize(coordinates[source] - coordinates[target], dim=-1)
        vector_input = torch.cat((vectors[source], direction[:, None]), dim=1)
        vector_hidden = torch.einsum("evc,ov->eoc", vector_input, self.vector_message)
        scalar_message = self.message(torch.cat((scalar[source], scalar[target],
                                                  vector_hidden.norm(dim=-1),
                                                  distances[:, None] / 10.0,
                                                  (torch.zeros_like(distances) if edge_type is None else edge_type)[:, None]), dim=-1))
        vector_message = torch.einsum("evc,ov->eoc", vector_hidden, self.vector_out)
        vector_message = vector_message * self.gate(scalar_message).sigmoid()[..., None]
        count = torch.zeros(len(scalar), 1, device=scalar.device, dtype=scalar.dtype).index_add_(
            0, target, torch.ones(len(target), 1, device=scalar.device, dtype=scalar.dtype)).clamp_min_(1)
        scalar_update = torch.zeros_like(scalar).index_add_(0, target, scalar_message.to(scalar.dtype))
        vector_update = torch.zeros_like(vectors).index_add_(0, target, vector_message.to(vectors.dtype))
        scalar = self.norm(scalar + scalar_update / count)
        vectors = vectors + vector_update / count[..., None]
        return scalar, vectors


def _monomer_edges(inputs: dict[str, torch.Tensor], prefix: str,
                    use_sequence_edges: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = inputs[f"{prefix}_spatial_edges"]
    distances = inputs[f"{prefix}_spatial_distances"]
    edge_type = torch.zeros_like(distances)
    if use_sequence_edges:
        sequence_edges = inputs[f"{prefix}_sequence_edges"]
        sequence_distances = torch.linalg.vector_norm(
            inputs[f"{prefix}_coordinates"][sequence_edges[0]]
            - inputs[f"{prefix}_coordinates"][sequence_edges[1]], dim=-1)
        edges = torch.cat((edges, sequence_edges), dim=1)
        distances = torch.cat((distances, sequence_distances))
        edge_type = torch.cat((edge_type, torch.ones_like(sequence_distances)))
    return edges, distances, edge_type
