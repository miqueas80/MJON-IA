"""Generación con checkpoints propios de MJON IA v0.3 y tokenizer entrenado localmente."""
import argparse
import torch
from .model import ByteTransformer,ModelConfig
from .tokenizer import ByteBPETokenizer


def generate(checkpoint,prompt,new_tokens=80,temperature=.8,top_k=30,device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('version')!=3:raise ValueError('Se requiere modelo MJON IA v0.3')
    tok=ByteBPETokenizer.from_dict(state['tokenizer'])
    model=ByteTransformer(ModelConfig(**state['model_config'])).to(device).eval()
    model.load_state_dict(state['state_dict'])
    ids=tok.encode(prompt)
    if not ids:raise ValueError('Prompt vacío')
    input_tokens=torch.tensor([ids],dtype=torch.long,device=device)
    output=model.generate(input_tokens,new_tokens=new_tokens,temperature=temperature,top_k=top_k)
    return tok.decode(output[0].tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/mjon-v03.pt')
    p.add_argument('--prompt',default='Usuario: Hola\nAsistente:')
    p.add_argument('--tokens',type=int,default=80)
    p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--top-k',type=int,default=30)
    a=p.parse_args()
    print(generate(a.checkpoint,a.prompt,a.tokens,a.temperature,a.top_k))
