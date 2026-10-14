"""Publicar el tercer checkpoint real en el catálogo de modelos de GitHub Pages."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint,catalog_path,weights_path):
    obj=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if obj.get('version')!=3:raise ValueError('Checkpoint no es v0.3')
    catalog_path=Path(catalog_path);weights_path=Path(weights_path)
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    if catalog['format']!='mjonia.models.v1':raise ValueError('Catálogo incompatible')
    unique={}
    for tensor in obj['state_dict'].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        unique[key]=tensor.numel()
    entry={'id':'v03','label':'v0.3 · BPE + modelo 500K',
           'description':'Transformer con 505.920 parámetros y tokenizador BPE propio. Sigue siendo experimental.',
           'file':weights_path.name,'sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
           'steps':obj['step'],'context':obj['model_config']['context'],
           'parameters':sum(unique.values()),'tokenizer':'bpe','split_mode':'dialogues'}
    if any(item['id']=='v03' for item in catalog['models']):raise ValueError('Modelo duplicado')
    catalog['models'].append(entry)
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Registrado:',entry)
    return entry


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/web/v03.pt')
    p.add_argument('--catalog',default='docs/web/models.json')
    p.add_argument('--weights',default='docs/web/weights-v03.txt')
    a=p.parse_args();run(a.checkpoint,a.catalog,a.weights)
