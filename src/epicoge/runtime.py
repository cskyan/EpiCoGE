"""Deterministic execution and complex-macro evaluation."""
from contextlib import nullcontext
import os
import random

import numpy as np
import torch

from .metrics import binary_metrics, macro_mean, select_global_mcc_threshold
from .model import objective


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def device_for(name,precision):
    device=torch.device(name)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(device)
    if precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Paper BF16 execution requires a supported CUDA GPU; CPU checks require --precision fp32')
    return device


def autocast(device,precision):
    return torch.autocast(device.type,dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def move(sample,device):
    return ({k:v.to(device) for k,v in sample['inputs'].items()},
            {k:v.to(device) for k,v in sample['targets'].items()})


@torch.no_grad()
def evaluate(model,samples,device,precision,weights,contact_coefficient,threshold=None):
    model.eval();examples=[];rows=[];losses=[];contacts=[]
    for sample in samples:
        inputs,targets=move(sample,device)
        with autocast(device,precision):
            output=model(inputs)
            loss,_,contact=objective(output,targets,weights,contact_coefficient)
        labels=targets['epitope'].cpu().numpy()
        scores=output['epitope'].sigmoid().cpu().numpy()
        examples.append((labels,scores));losses.append(float(loss));contacts.append(float(contact))
        rows.append({'id':sample['id'],'candidate_indices':inputs['ag_candidate_indices'].cpu().tolist(),
            'epitope_labels':labels.tolist(),'epitope_scores':scores.tolist()})
    if threshold is None: threshold,_=select_global_mcc_threshold(examples)
    for row,(labels,scores) in zip(rows,examples):
        row['metrics']=binary_metrics(labels,scores,threshold)
    summary=macro_mean([r['metrics'] for r in rows])
    summary.update(threshold=threshold,complexes=len(rows),loss=float(np.mean(losses)),
                   contact_loss=float(np.mean(contacts)))
    return summary,rows
