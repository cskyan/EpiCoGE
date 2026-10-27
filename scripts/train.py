"""Train EpiCoGE from random initialization with Validation-only checkpoint selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from epicoge import EpiCoGE
from epicoge.data import load_samples, read_json, save_tensor, sha256, write_json
from epicoge.model import objective, optimizer_for
from epicoge.runtime import autocast, device_for, evaluate, move, seed_all


def train_one(config,data,weights,seed,root,device,precision,config_hash,data_hashes):
    folder=root/config['split']/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    seed_all(seed)
    model=EpiCoGE(**config['model']).to(device)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=config['expected_parameters']: raise ValueError('Model parameter count differs from the paper')
    budget=config['training'];coefficient=config['contact_loss_coefficient']
    optimizer=optimizer_for(model,budget)
    metadata={'model':'EpiCoGE','split':config['split'],'seed':seed,'parameters':parameters,
        'class_weights':weights,'contact_loss_coefficient':coefficient,'config_sha256':config_hash,
        'data_sha256':data_hashes,'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'device':str(device),'precision':precision,
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'tf32':False,'deterministic_algorithms':True,'test_samples_read_during_training':0}
    write_json(folder/'run_metadata.json',metadata)
    history=[];best_score=-float('inf');stale=0;updates=0;started=time.perf_counter()
    for epoch in range(1,budget['max_epochs']+1):
        model.train();order=list(range(len(data['train'])));random.shuffle(order)
        losses=[];nodes=[];contacts=[];norms=[]
        for start in range(0,len(order),budget['effective_batch_complexes']):
            indices=order[start:start+budget['effective_batch_complexes']]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                inputs,targets=move(data['train'][index],device)
                with autocast(device,precision):
                    loss,node,contact=objective(model(inputs),targets,weights,coefficient)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/len(indices)).backward()
                losses.append(float(loss.detach()));nodes.append(float(node.detach()));contacts.append(float(contact.detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),budget['gradient_clip_norm'],error_if_nonfinite=True)
            norms.append(float(norm));optimizer.step();updates+=1
        validation,predictions=evaluate(model,data['validation'],device,precision,weights,coefficient)
        entry={'epoch':epoch,'updates':updates,'train_loss':float(np.mean(losses)),
            'train_node_loss':float(np.mean(nodes)),'train_contact_loss':float(np.mean(contacts)),
            'gradient_norm_mean':float(np.mean(norms)),
            'clipped_fraction':float(np.mean(np.array(norms)>budget['gradient_clip_norm'])),
            'validation':validation,'elapsed_seconds':time.perf_counter()-started}
        history.append(entry)
        if validation['mcc']>best_score:
            best_score=validation['mcc'];stale=0
            save_tensor(folder/'best.pt',{'model':model.state_dict(),'config':config,'metadata':metadata,
                'epoch':epoch,'threshold':validation['threshold'],'validation':validation})
            write_json(folder/'validation_predictions.json',predictions)
        else: stale+=1
        write_json(folder/'history.json',history)
        print({'split':config['split'],'seed':seed,'epoch':epoch,'validation_mcc':validation['mcc'],
               'threshold':validation['threshold'],'stale_epochs':stale},flush=True)
        if stale>=budget['patience']: break
    write_json(folder/'training_summary.json',{'model':'EpiCoGE','split':config['split'],'seed':seed,
        'epochs':len(history),'best_validation_mcc':best_score,'best_checkpoint_sha256':sha256(folder/'best.pt'),
        'test_evaluated':False})
    del model,optimizer;gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('runs'))
    parser.add_argument('--seed',type=int,nargs='+',help='Default: all three paper seeds')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--precision',choices=['bf16','fp32'],help='Default: BF16 from paper config')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    seeds=config['seeds'] if args.seed is None else args.seed
    if len(set(seeds))!=len(seeds) or any(s not in config['seeds'] for s in seeds):
        raise ValueError('Use the three distinct seeds listed in the configuration')
    precision=args.precision or config['precision']
    device=device_for(args.device,precision)
    for seed in seeds:
        if (args.output/config['split']/f'seed_{seed}').exists():
            raise FileExistsError('Run directory exists; use a new --output to repeat an experiment')
    data=load_samples(args.data_root,config['split'],['train','validation'])
    for role in data:
        if len(data[role])!=config['expected_counts'][role]: raise ValueError(f'{role}: incorrect sample count')
    positive=sum(int(s['targets']['contact'].sum()) for s in data['train'])
    total=sum(s['targets']['contact'].numel() for s in data['train'])
    weights={'node':config['node_positive_weight'],'contact':(total-positive)/positive}
    if weights['node']!=1 or not np.isclose(weights['contact'],config['expected_contact_positive_weight'],rtol=1e-12):
        raise ValueError('Train class counts differ from the paper')
    hashes={role:sha256(args.data_root/'prepared'/f"{config['split']}_{role}.json") for role in data}
    for seed in seeds:
        train_one(config,data,weights,seed,args.output,device,precision,sha256(args.config),hashes)


if __name__=='__main__':
    main()
