"""Agrega un cuarto modelo realmente entrenado al catálogo estático."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint='checkpoints/web/v04.pt',catalog='docs/web/models.json',weights='docs/web/weights-v04.txt'):
    path=Path(checkpoint);catalog_path=Path(catalog);weights_path=Path(weights)
    state=torch.load(path,weights_only=True,map_location='cpu')
    if state.get('version')!=3 or not state.get('tokenizer'):
        raise ValueError('Checkpoint sin BPE y entrenamiento reproducible')
    info=json.loads(catalog_path.read_text(encoding='utf-8'))
    if info.get('format')!='mjonia.models.v1':raise ValueError('Catálogo inválido')
    if any(x.get('id')=='v04' for x in info['models']):raise ValueError('v04 duplicada')
    storages={}
    for tensor in state['state_dict'].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        storages[key]=tensor.numel()
    params=sum(storages.values())
    entry={
        'id':'v04', 'label':'v0.4 · Corpus Forge + 1,27M',
        'description':'Transformer de 1,27 millones de parámetros, BPE propio y corpus sintético multitemático. Aún experimental.',
        'file':weights_path.name,'sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        'steps':int(state['step']),'context':int(state['model_config']['context']),
        'parameters':params,'tokenizer':'bpe','split_mode':'dialogues'
    }
    info['models'].append(entry)
    catalog_path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Catálogo v0.4:',entry)
    return entry


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/web/v04.pt')
    p.add_argument('--catalog',default='docs/web/models.json')
    p.add_argument('--weights',default='docs/web/weights-v04.txt')
    a=p.parse_args();run(a.checkpoint,a.catalog,a.weights)
