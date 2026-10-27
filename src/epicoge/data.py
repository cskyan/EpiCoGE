"""AsEP input assembly; labels remain separate from model inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .features import antibody_annotations, residue_features


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def manifest(root, split, roles):
    records = rows(Path(root)/'manifests'/f'{split}.csv')
    return [r for r in records if r['role'] in roles]


def labels_and_masks(graph):
    masks = {p: torch.as_tensor(graph['mapping'][p][key], dtype=torch.bool)
             for p, key in [('ab','seqres2cdr'), ('ag','seqres2surf')]}
    if list(graph['seqres']['ab']) != ['H','L']:
        raise ValueError('Expected antibody chain order H, L')
    for prefix in ['ab','ag']:
        if len(masks[prefix]) != sum(map(len, graph['seqres'][prefix].values())):
            raise ValueError(f'{prefix}: sequence/candidate mask length mismatch')
    raw = graph['edges']['bipartite']
    contact = raw.to_dense().float()
    epitope = torch.as_tensor(graph['mapping']['ag']['seqres2epitope'])[masks['ag']].float()
    if contact.shape != (int(masks['ab'].sum()),int(masks['ag'].sum())):
        raise ValueError('Candidate axes do not match official contact labels')
    if not bool(((contact==0)|(contact==1)).all()) or not torch.equal(contact.amax(dim=0),epitope):
        raise ValueError('Official node/contact labels are inconsistent')
    if not 0 < epitope.sum() < len(epitope):
        raise ValueError('AsEP metrics require positive and negative antigen candidates')
    return masks, {'epitope':epitope,'contact':contact}


def validate_embedding(record, sequence):
    tensor = record['embedding']
    if (record['sequence'] != sequence or record['layer'] != 33
        or record['sequence_sha256'] != sequence_hash(sequence)
        or tensor.shape != (len(sequence),1280) or not torch.isfinite(tensor).all()):
        raise ValueError('ESM2 cache must contain the complete sequence, layer 33 and 1280 features')
    return tensor.float()


def assemble_sample(identity, graph, embedding, geometry_path):
    masks, targets = labels_and_masks(graph)
    sequence = ''.join(graph['seqres']['ag'].values())
    antigen = validate_embedding(embedding,sequence)
    antibody = graph['embedding']['ab']['igfold'].float()
    if antibody.shape != (len(masks['ab']),512) or not torch.isfinite(antibody).all():
        raise ValueError('Official AntiBERTy cache must have 512 features per antibody residue')
    inputs = {'ab':antibody[masks['ab']].clone(),'ag':antigen[masks['ag']].clone(),
              'ag_candidate_indices':masks['ag'].nonzero().flatten()}
    inputs['ab_chain_type'],inputs['ab_cdr_type'] = antibody_annotations(graph['seqres']['ab'],masks['ab'])
    with np.load(geometry_path,allow_pickle=False) as geometry:
        for prefix in ['ab','ag']:
            indices = masks[prefix].nonzero().flatten()
            for key,dtype in [('coordinates',torch.float32),('vectors',torch.float32),
                ('vector_mask',torch.bool),('spatial_edges',torch.long),('spatial_distances',torch.float32),
                ('sequence_edges',torch.long),('sequence_indices',torch.long),
                ('backbone_mask',torch.bool),('sasa',torch.float32)]:
                inputs[prefix+'_'+key] = torch.from_numpy(geometry[prefix+'_'+key].copy()).to(dtype)
            if not torch.equal(indices,inputs[prefix+'_sequence_indices']):
                raise ValueError(f'{identity}: {prefix} geometry/candidate order mismatch')
            coordinates = inputs[prefix+'_coordinates']
            vectors,mask = inputs[prefix+'_vectors'],inputs[prefix+'_vector_mask']
            backbone = torch.from_numpy(geometry[prefix+'_backbone'].copy()).float()
            bm = inputs[prefix+'_backbone_mask']
            if coordinates.shape != (len(indices),3) or vectors.shape != (len(indices),3,3):
                raise ValueError('Invalid coordinate/vector shape')
            if not torch.allclose(coordinates,backbone[:,1]):
                raise ValueError('CA/backbone mismatch')
            for vi,atom in enumerate([0,2,3]):
                if not torch.equal(mask[:,vi],bm[:,1]&bm[:,atom]):
                    raise ValueError('Backbone vector mask mismatch')
                if not torch.allclose(vectors[mask[:,vi],vi],(backbone[:,atom]-coordinates)[mask[:,vi]],atol=2e-5):
                    raise ValueError('Backbone vector mismatch')
            if bool((vectors[~mask]!=0).any()):
                raise ValueError('Missing backbone vectors must be zero')
            edges = inputs[prefix+'_spatial_edges']
            distance = inputs[prefix+'_spatial_distances']
            if edges.ndim!=2 or edges.shape[0]!=2 or distance.shape!=(edges.shape[1],):
                raise ValueError('Invalid spatial edges')
            if edges.numel():
                if edges.min()<0 or edges.max()>=len(indices) or bool((edges[0]==edges[1]).any()):
                    raise ValueError('Invalid spatial edge indices')
                if not bool((distance<10).all()) or int(torch.bincount(edges[0],minlength=len(indices)).max())>16:
                    raise ValueError('Spatial radius/degree mismatch')
                if not torch.allclose((coordinates[edges[0]]-coordinates[edges[1]]).norm(dim=-1),distance,atol=2e-5):
                    raise ValueError('Spatial edge distance mismatch')
            sequence_edges = inputs[prefix+'_sequence_edges']
            if sequence_edges.ndim!=2 or sequence_edges.shape[0]!=2:
                raise ValueError('Invalid sequence edges')
            if sequence_edges.numel():
                if sequence_edges.min()<0 or sequence_edges.max()>=len(indices):
                    raise ValueError('Sequence edge outside candidate axis')
                if not bool(((indices[sequence_edges[0]]-indices[sequence_edges[1]]).abs()==1).all()):
                    raise ValueError('Sequence edges must join consecutive original positions')
                for boundary in np.cumsum([len(s) for s in graph['seqres'][prefix].values()])[:-1]:
                    if bool(((indices[sequence_edges[0]]<boundary)!=(indices[sequence_edges[1]]<boundary)).any()):
                        raise ValueError('Sequence edge crosses chains')
            inputs[prefix+'_scalar_features'] = residue_features(''.join(graph['seqres'][prefix].values()),
                indices,inputs[prefix+'_sasa'],bm)
            if inputs[prefix+'_scalar_features'].shape != (len(indices),30):
                raise ValueError('Scalar feature width must be 30')
    if any(not torch.isfinite(value).all() for value in inputs.values()):
        raise ValueError('Nonfinite model input')
    return {'id':identity,'inputs':inputs,'targets':targets}


def load_samples(root,split,roles):
    """Training requests Train/Val only; Test is loaded by the evaluator."""
    root = Path(root)
    records = manifest(root,split,roles)
    output = {role:[] for role in roles}
    digests = {}
    for role in roles:
        prepared = read_json(root/'prepared'/f'{split}_{role}.json')
        digests.update({(role,r['id']):r['sha256'] for r in prepared['records']})
    for row in records:
        path = root/'processed'/split/row['role']/(row['complex_id']+'.pt')
        if sha256(path) != digests.get((row['role'],row['complex_id'])):
            raise ValueError(f'Prepared input hash mismatch: {path}')
        sample = load_tensor(path)
        if sample['id'] != row['complex_id']:
            raise ValueError(f'Prepared input identity mismatch: {path}')
        output[row['role']].append(sample)
    if any(not values for values in output.values()):
        raise ValueError('Requested split/role is empty; run build_geometry.py first')
    return output
