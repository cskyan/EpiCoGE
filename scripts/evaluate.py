"""Evaluate a trained checkpoint at its saved Validation threshold, or summarize three seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from epicoge import EpiCoGE
from epicoge.data import load_samples, load_tensor, read_json, sha256, write_json
from epicoge.runtime import device_for, evaluate, seed_all


def summarize(paths,output):
    records=[read_json(p) for p in paths]
    if not records: raise ValueError('No metric files supplied')
    if any(r['role']!='test' for r in records): raise ValueError('Paper summary expects Test results')
    metrics=['auroc','mcc','precision','recall','f1'];summary=[]
    for split in sorted({r['split'] for r in records}):
        group=[r for r in records if r['split']==split]
        if len(group)!=3 or {r['seed'] for r in group}!={20260908,20260919,20260920}:
            raise ValueError(f'{split}: supply exactly the three distinct paper seeds')
        if len({r['precision'] for r in group})!=1: raise ValueError('Cannot mix execution precisions')
        row={'model':'EpiCoGE','split':split,'seeds':3}
        for name in metrics:
            values=[r['metrics'][name] for r in group]
            row[name]=float(np.mean(values));row[name+'_sd']=float(np.std(values,ddof=1))
        summary.append(row)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'mean_sd.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_json(output/'summary.json',summary)
    print(output/'mean_sd.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint',type=Path)
    group.add_argument('--summaries',type=Path,nargs='+',help='Three Test metrics.json files per split')
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--role',choices=['validation','test'],default='test')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--precision',choices=['bf16','fp32'],help='Default: checkpoint training precision')
    args=parser.parse_args()
    if args.summaries:
        summarize(args.summaries,args.output);return
    checkpoint=load_tensor(args.checkpoint);config=checkpoint['config'];metadata=checkpoint['metadata']
    precision=args.precision or metadata['precision'];seed_all(metadata['seed'])
    device=device_for(args.device,precision)
    if args.output.exists(): raise FileExistsError('Evaluation output already exists')
    data=load_samples(args.data_root,config['split'],[args.role])[args.role]
    if len(data)!=config['expected_counts'][args.role]: raise ValueError('Incorrect evaluation sample count')
    model=EpiCoGE(**config['model']).to(device);model.load_state_dict(checkpoint['model'],strict=True)
    metrics,predictions=evaluate(model,data,device,precision,metadata['class_weights'],
        config['contact_loss_coefficient'],threshold=checkpoint['threshold'])
    args.output.mkdir(parents=True)
    write_json(args.output/'metrics.json',{'model':'EpiCoGE','split':config['split'],'seed':metadata['seed'],
        'role':args.role,'precision':precision,'threshold_source':'Validation',
        'checkpoint_sha256':sha256(args.checkpoint),
        'prepared_data_sha256':sha256(args.data_root/'prepared'/f"{config['split']}_{args.role}.json"),
        'metrics':metrics})
    write_json(args.output/'predictions.json',predictions)
    with (args.output/'per_complex.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','auroc','mcc','precision','recall','f1'])
        writer.writeheader();writer.writerows({'id':r['id'],**r['metrics']} for r in predictions)
    print(metrics)


if __name__=='__main__':
    main()
