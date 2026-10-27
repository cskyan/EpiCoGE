"""Register AsEP v1.2.0 and freeze the official Group/Ratio memberships."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from epicoge.data import load_tensor, rows, sequence_hash, sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True,help='Directory containing split_dict.pt and asepv1-AbDb-IDs.txt')
    parser.add_argument('--output',type=Path,default=Path('data'))
    args=parser.parse_args()
    raw=args.raw_root.resolve(); output=args.output.resolve()
    ids=[line.strip() for line in (raw/'asepv1-AbDb-IDs.txt').read_text().splitlines() if line.strip()]
    if len(ids)!=1723 or len(set(ids))!=1723 or any(Path(i).name!=i or '/' in i or '\\' in i for i in ids):
        raise ValueError('Expected 1723 unique, valid AsEP sample IDs')
    official=load_tensor(raw/'split_dict.pt')
    def index_files(folder,suffix):
        paths={}
        for path in folder.rglob('*'+suffix):
            if path.stem in paths:
                raise ValueError(f'Duplicate sample file: {path.stem}')
            paths[path.stem]=path
        if not set(ids)<=set(paths):
            raise ValueError(f'Missing {len(set(ids)-set(paths))} files in {folder}')
        return paths
    graphs=index_files(raw/'interim_graphs','.pt')
    structures=index_files(raw/'structures','.pdb')
    output.joinpath('manifests').mkdir(parents=True,exist_ok=True)
    for split in ['group','ratio']:
        membership={role:sorted(int(i) for i in official['epitope_'+split][key])
                    for role,key in [('train','train'),('validation','val'),('test','test')]}
        if {k:len(v) for k,v in membership.items()}!={'train':1383,'validation':170,'test':170}:
            raise ValueError('Unexpected official split sizes')
        joined=[i for group in membership.values() for i in group]
        if len(set(joined))!=1723 or set(joined)!=set(range(1723)):
            raise ValueError('Official roles overlap or do not form a complete partition')
        records=[{'role':role,'index':i,'complex_id':ids[i]} for role,indices in membership.items() for i in indices]
        paper_path=Path(__file__).resolve().parents[1]/'data/manifests'/f'{split}_train_validation.csv'
        paper=rows(paper_path)
        if paper != [dict(r,index=str(r['index'])) for r in records if r['role']!='test']:
            raise ValueError('Downloaded memberships differ from the paper manifests')
        with (output/'manifests'/f'{split}.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['role','index','complex_id'])
            writer.writeheader();writer.writerows(records)
    sequences={}; samples={}
    for n,identity in enumerate(ids,1):
        graph=load_tensor(graphs[identity])
        if list(graph['seqres']['ab'])!=['H','L']:
            raise ValueError(f'{identity}: antibody chains must be H,L in that order')
        sequence=''.join(graph['seqres']['ag'].values())
        digest=sequence_hash(sequence)
        sequences[digest]=sequence
        samples[identity]={'graph':str(graphs[identity].relative_to(raw)),
            'structure':str(structures[identity].relative_to(raw)),
            'graph_sha256':sha256(graphs[identity]),'structure_sha256':sha256(structures[identity]),
            'antigen_sequence_sha256':digest}
        if n%100==0: print(f'Registered {n}/{len(ids)}',flush=True)
    write_json(output/'sequences.json',sequences)
    write_json(output/'dataset.json',{'dataset':'AsEP','version':'1.2.0',
        'source':'https://doi.org/10.5281/zenodo.11495514','raw_root':str(raw),
        'official_ids_sha256':sha256(raw/'asepv1-AbDb-IDs.txt'),
        'official_split_sha256':sha256(raw/'split_dict.pt'),'samples':samples})
    print(f'Registered {len(samples)} complexes and {len(sequences)} unique antigen sequences in {output}')


if __name__=='__main__':
    main()
