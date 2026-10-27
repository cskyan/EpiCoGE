"""Build partner-separated monomer geometry and training-ready AsEP inputs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from epicoge.data import assemble_sample, load_tensor, manifest, read_json, save_tensor, sha256, write_json
from epicoge.geometry import add_graph_geometry, build_partner_separated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--split',choices=['group','ratio'],required=True)
    parser.add_argument('--roles',nargs='+',choices=['train','validation','test'],default=['train','validation'])
    parser.add_argument('--clustalo',default='clustalo',help='Clustal Omega 1.2.4 executable')
    parser.add_argument('--geometry-seed',type=int,default=20260906)
    args=parser.parse_args()
    executable=shutil.which(args.clustalo)
    if executable is None: raise FileNotFoundError('Install Clustal Omega 1.2.4 or pass --clustalo /path/to/clustalo')
    version=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True).stdout.strip()
    if '1.2.4' not in version: raise RuntimeError(f'The paper mapping uses Clustal Omega 1.2.4; found {version}')
    root=args.data_root;registry=read_json(root/'dataset.json');raw=Path(registry['raw_root'])
    all_records=manifest(root,args.split,['train','validation','test']);audit=[]
    for role in args.roles:
        if (root/'prepared'/f'{args.split}_{role}.json').exists():
            raise FileExistsError(f'{args.split}/{role} is already prepared; use a new output directory to rebuild')
    for n,row in enumerate(all_records):
        role=row['role']
        if role not in args.roles: continue
        identity=row['complex_id'];record=registry['samples'][identity]
        graph_path=raw/record['graph'];pdb_path=raw/record['structure']
        if sha256(graph_path)!=record['graph_sha256'] or sha256(pdb_path)!=record['structure_sha256']:
            raise ValueError(f'{identity}: source files changed after prepare_data.py')
        graph=load_tensor(graph_path)
        # Test preprocessing starts at its own origin, as a separate role manifest.
        position=n if role!='test' else n-1553
        pair=build_partner_separated(graph,pdb_path,args.geometry_seed+2*position,Path(executable))
        arrays={}
        for prefix,name in [('ab','antibody'),('ag','antigen')]:
            side=add_graph_geometry(pair[name])
            for key in ['coordinates','backbone','backbone_mask','vectors','vector_mask',
                        'sequence_edges','spatial_edges','spatial_distances','sequence_indices','sasa']:
                arrays[prefix+'_'+key]=side[key]
        geometry_path=root/'geometry'/args.split/role/(identity+'.npz')
        geometry_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(geometry_path,**arrays)
        embedding_path=root/'embeddings'/(record['antigen_sequence_sha256']+'.pt')
        sample=assemble_sample(identity,graph,load_tensor(embedding_path),geometry_path)
        path=root/'processed'/args.split/role/(identity+'.pt');save_tensor(path,sample)
        audit.append({'id':identity,'role':role,'sha256':sha256(path),
            'geometry_sha256':sha256(geometry_path),'embedding_sha256':sha256(embedding_path),
            'contact_positive':int(sample['targets']['contact'].sum()),
            'contact_total':sample['targets']['contact'].numel()})
        if len(audit)%20==0: print(f'Prepared {len(audit)} complexes',flush=True)
    for role in args.roles:
        records=[r for r in audit if r['role']==role]
        expected=1383 if role=='train' else 170
        if len(records)!=expected: raise ValueError(f'{role}: expected {expected} complexes')
        write_json(root/'prepared'/f'{args.split}_{role}.json',{'split':args.split,'role':role,'records':records,
            'clustalo':version,'geometry_seed':args.geometry_seed,
            'structure_source':'bound monomers with partner removed','independent_coordinate_frames':True,
            'radius_angstrom':10.0,'max_neighbors_per_source':16,
            'mapping_labels':'official AsEP masks and labels, unchanged'})
    print(f'Prepared {len(audit)} {args.split} complexes')


if __name__=='__main__':
    main()
