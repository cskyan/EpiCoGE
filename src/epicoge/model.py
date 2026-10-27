"""EpiCoGE antibody-specific epitope prediction."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import _GVPMessage, _monomer_edges


class GeometryMessage(_GVPMessage):
    """Reuse audited GVP parameterization; dropout only on scalar residual updates."""
    def __init__(self, hidden, vector_hidden, dropout):
        super().__init__(hidden, vector_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scalar, vectors, coordinates, edges, distances, edge_type):
        source, target = edges
        direction = F.normalize(coordinates[source] - coordinates[target], dim=-1)
        vector_input = torch.cat((vectors[source], direction[:, None]), dim=1)
        mixed = torch.einsum('evc,ov->eoc', vector_input, self.vector_message)
        message = self.message(torch.cat((scalar[source], scalar[target], mixed.norm(dim=-1),
                                          distances[:, None] / 10, edge_type[:, None]), dim=-1))
        update = torch.einsum('evc,ov->eoc', mixed, self.vector_out)
        update = update * self.gate(message).sigmoid()[..., None]
        count = scalar.new_zeros((len(scalar), 1)).index_add_(
            0, target, scalar.new_ones((len(target), 1))).clamp_min_(1)
        scalar_update = torch.zeros_like(scalar).index_add_(0, target, message.to(scalar.dtype)) / count
        vector_update = torch.zeros_like(vectors).index_add_(0, target, update.to(vectors.dtype))
        return self.norm(scalar + self.dropout(scalar_update)), vectors + vector_update / count[..., None]


class LocalEncoder(nn.Module):
    def __init__(self, input_dim, hidden, vector_hidden, layers, dropout, antibody=False):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.vector_in = nn.Parameter(torch.empty(vector_hidden, 3))
        nn.init.xavier_uniform_(self.vector_in)
        self.messages = nn.ModuleList(GeometryMessage(hidden, vector_hidden, dropout) for _ in range(layers))
        self.bridge = nn.Linear(hidden + vector_hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        if antibody:
            self.chain = nn.Embedding(2, hidden)
            self.cdr = nn.Embedding(3, hidden)

    def forward(self, inputs, prefix):
        h0 = self.input(torch.cat((inputs[prefix], inputs[prefix + '_scalar_features']), dim=-1))
        if prefix == 'ab':
            h0 = h0 + self.chain(inputs['ab_chain_type']) + self.cdr(inputs['ab_cdr_type'])
        # Geometry and invariant norms stay FP32, even when attention/adapters use BF16.
        with torch.autocast(h0.device.type, enabled=False):
            scalar = h0.float()
            vectors = inputs[prefix + '_vectors'].float() * inputs[prefix + '_vector_mask'][..., None]
            vectors = torch.einsum('nvc,ov->noc', vectors, self.vector_in)
            edges, distances, types = _monomer_edges(inputs, prefix, True)
            for message in self.messages:
                scalar, vectors = message(scalar, vectors, inputs[prefix + '_coordinates'], edges, distances, types)
            return self.norm(h0.float() + self.bridge(torch.cat((scalar, vectors.norm(dim=-1)), dim=-1)))


class AttentionBlock(nn.Module):
    """Pre-norm attention and FFN with ordinary, nonzero residual paths."""
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.partner_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, partner):
        q, kv = self.query_norm(query), self.partner_norm(partner)
        update, _ = self.attention(q[None], kv[None], kv[None], need_weights=False)
        hidden = query + self.dropout(update[0])
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))


class EpiCoGE(nn.Module):
    """One complex at a time; accumulation gives exact complex-equal batches without padding."""
    def __init__(self, hidden=256, vector_hidden=16, layers=2, heads=4, dropout=.1):
        super().__init__()
        if hidden % heads or min(hidden, vector_hidden, layers, heads) <= 0:
            raise ValueError('invalid model dimensions')
        self.ab_local = LocalEncoder(542, hidden, vector_hidden, layers, dropout, antibody=True)
        self.ag_local = LocalEncoder(1310, hidden, vector_hidden, layers, dropout)
        self.ab_self = AttentionBlock(hidden, heads, dropout)
        self.ag_self = AttentionBlock(hidden, heads, dropout)
        self.ab_cross = AttentionBlock(hidden, heads, dropout)
        self.ag_cross = AttentionBlock(hidden, heads, dropout)
        self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.node = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, inputs):
        ab0, ag0 = self.ab_local(inputs, 'ab'), self.ag_local(inputs, 'ag')
        ab0, ag0 = self.ab_self(ab0, ab0), self.ag_self(ag0, ag0)
        ab, ag = self.ab_cross(ab0, ag0), self.ag_cross(ag0, ab0)
        # FP32 pair head, masked-free ragged aggregation, and node head.
        with torch.autocast(ab.device.type, enabled=False):
            ab, ag = ab.float(), ag.float()
            a = ab[:, None].expand(-1, len(ag), -1)
            g = ag[None].expand(len(ab), -1, -1)
            pair = self.pair(torch.cat((a, g, a * g, (a - g).abs()), dim=-1)).squeeze(-1)
            alpha, probability = pair.softmax(dim=0), pair.sigmoid()
            context = alpha.transpose(0, 1) @ ab
            statistics = torch.stack((probability.amax(dim=0), probability.mean(dim=0)), dim=-1)
            logits = self.node(torch.cat((ag, context, statistics), dim=-1)).squeeze(-1)
            entropy = -(alpha * alpha.clamp_min(1e-30).log()).sum(dim=0) / max(math.log(len(ab)), 1e-30)
        return {'epitope': logits, 'contact': pair, 'partner_entropy': entropy,
                'pair_statistics': statistics}

    def predict(self, inputs):
        """Public output: only candidate antigen epitope scores."""
        return self(inputs)['epitope'].sigmoid()


def weighted_bce(logits, labels, positive_weight, valid=None, components=False):
    """Normalize by valid class-weight sum inside each complex, never by batch prevalence."""
    logits, labels = logits.float(), labels.float()
    if valid is None:
        valid = torch.isfinite(labels) & (labels >= 0)
    labels, logits = labels[valid], logits[valid]
    if not labels.numel() or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('no valid labels or nonbinary labels')
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('positive class weight must be finite and positive')
    denominator = (positive_weight * labels + 1 - labels).sum()
    positive = (positive_weight * labels * F.softplus(-logits)).sum() / denominator
    negative = ((1 - labels) * F.softplus(logits)).sum() / denominator
    return (positive, negative) if components else positive + negative


def objective(output, targets, weights, lambda_contact):
    node = weighted_bce(output['epitope'], targets['epitope'], weights['node'])
    contact = weighted_bce(output['contact'], targets['contact'], weights['contact'])
    return node + lambda_contact * contact, node, contact


def optimizer_for(model, budget):
    norm_parameters = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith('bias') or id(parameter) in norm_parameters else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': budget['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0}],
                            lr=budget['lr'], betas=tuple(budget['betas']), eps=budget['eps'])
