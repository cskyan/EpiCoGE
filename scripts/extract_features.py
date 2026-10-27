"""Extract complete-sequence ESM2-650M layer-33 features; reuse official AntiBERTy features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from epicoge.data import load_tensor, read_json, save_tensor, sha256, validate_embedding, write_json


MODEL_NAME='esm2_t33_650M_UR50D'
PRETRAINED_FILES={
    MODEL_NAME+'.pt':('models','ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c'),
    MODEL_NAME+'-contact-regression.pt':('regression','8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575'),
}


def pretrained_model(esm,torch):
    """Verify the paper's public ESM files, then explicitly load their trusted configuration objects."""
    folder=Path(torch.hub.get_dir())/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for filename,(category,digest) in PRETRAINED_FILES.items():
        path=folder/filename
        if not path.exists():
            temporary=path.with_suffix('.download')
            torch.hub.download_url_to_file('https://dl.fbaipublicfiles.com/fair-esm/'+category+'/'+filename,
                str(temporary),hash_prefix=digest)
            temporary.replace(path)
        if sha256(path)!=digest:
            raise ValueError(f'ESM pretrained file checksum mismatch: {path}; use a fresh TORCH_HOME cache')
        paths.append(path)
    # fair-esm's released checkpoints contain configuration objects as well as tensors.
    # weights_only=False is limited to these checksum-verified official files.
    return esm.pretrained.load_model_and_alphabet_core(MODEL_NAME,load_tensor(paths[0]),load_tensor(paths[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--tokens-per-batch',type=int,default=1024)
    args=parser.parse_args()
    import torch
    import esm
    if args.tokens_per_batch<=0: raise ValueError('Token batch budget must be positive')
    root=args.data_root; cache=root/'embeddings';cache.mkdir(parents=True,exist_ok=True)
    sequences=read_json(root/'sequences.json')
    pending=[]
    for key in sorted(sequences):
        path=cache/(key+'.pt')
        if path.exists(): validate_embedding(load_tensor(path),sequences[key])
        else: pending.append(key)
    if not pending:
        print(f'All {len(sequences)} sequence caches are complete')
        metadata=root/'embedding_metadata.json'
        if not metadata.exists():
            write_json(metadata,{'model':'esm2_t33_650M_UR50D','layer':33,'dimension':1280,
                'precision':'FP32','sequences':len(sequences),'sequence_truncation':False,
                'cache_validation':'All complete sequence tensors checked; generation process metadata unavailable'})
        return
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; use --device cpu if needed')
    torch.set_num_threads(2);torch.manual_seed(20260908);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,alphabet=pretrained_model(esm,torch)
    if model.num_layers!=33 or model.embed_dim!=1280: raise ValueError('Wrong ESM2 model')
    model=model.eval().to(device)
    dataset=esm.FastaBatchedDataset(pending,[sequences[key] for key in pending])
    batches=dataset.get_batch_indices(args.tokens_per_batch,extra_toks_per_seq=1)
    loader=torch.utils.data.DataLoader(dataset,collate_fn=alphabet.get_batch_converter(),batch_sampler=batches)
    started=time.perf_counter()
    with torch.inference_mode():
        for n,(keys,strings,tokens) in enumerate(loader,1):
            values=model(tokens.to(device),repr_layers=[33],return_contacts=False)['representations'][33]
            for i,(key,sequence) in enumerate(zip(keys,strings)):
                record={'sequence':sequence,'sequence_sha256':key,'layer':33,
                        'embedding':values[i,1:len(sequence)+1].float().cpu().clone()}
                validate_embedding(record,sequence);save_tensor(cache/(key+'.pt'),record)
            print(f'ESM2 batch {n}/{len(batches)}',flush=True)
    hub=Path(torch.hub.get_dir())/'checkpoints'/'esm2_t33_650M_UR50D.pt'
    write_json(root/'embedding_metadata.json',{'model':'esm2_t33_650M_UR50D','layer':33,'dimension':1280,
        'precision':'FP32','sequences':len(sequences),'sequence_truncation':False,
        'torch':torch.__version__,'fair_esm':getattr(esm,'__version__','unknown'),
        'pretrained_model_sha256':sha256(hub) if hub.exists() else None,
        'elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':
    main()
