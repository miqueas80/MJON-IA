"""Exporta checkpoints v0.1-v0.3 cuantizados para inferencia local en navegador."""
import argparse
import base64
import json
import zlib
from pathlib import Path
import torch


def run(checkpoint,destination):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    manifest=[];binary=bytearray()
    for name,tensor in obj['state_dict'].items():
        vec=tensor.reshape(-1)
        scale=max(vec.abs().max().item()/127.,1e-12)
        q=(vec/scale).round().clamp(-127,127).to(torch.int8).tolist()
        binary.extend(v&255 for v in q)
        manifest.append([name,list(tensor.shape),scale,len(q)])
    meta={'format':'mjonia.int8.binary.v1','config':obj['model_config'],
          'steps':obj['step'],'tensors':manifest}
    if obj.get('tokenizer'):
        meta['tokenizer']=obj['tokenizer']
    header=json.dumps(meta,separators=(',',':')).encode('utf-8')
    if len(header)>65535:raise ValueError('Manifiesto demasiado largo')
    payload=len(header).to_bytes(2,'big')+header+binary
    encoded=base64.b64encode(zlib.compress(payload,9)).decode('ascii')
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded,encoding='ascii')
    print('Modelo web:',path, 'bytes:',len(encoded))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/browser.pt')
    p.add_argument('--out',default='docs/web/weights.txt')
    a=p.parse_args();run(a.checkpoint,a.out)
