"""Entrenamiento MJON IA: checkpoints reanudables y evaluación estable (v0.2)."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from .data import random_batch, read_bytes, split_data, split_dialogues
from .model import ByteTransformer, ModelConfig


@torch.no_grad()
def evaluate(model, validation, context, device, windows=8):
    """Ventanas de validación fijas para comparar puntos de entrenamiento."""
    model.eval()
    total = len(validation) - context
    if total <= 0:
        raise ValueError('Validación insuficiente para el contexto')
    offsets = torch.linspace(0, total - 1, steps=min(windows, total)).long().tolist()
    losses = []
    for offset in offsets:
        x = validation[offset:offset+context].unsqueeze(0).to(device)
        y = validation[offset+1:offset+context+1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def _atomic_save(checkpoint, output):
    """Evitar dejar un checkpoint incompleto si se interrumpe una escritura."""
    temp = output.with_name(output.name + '.tmp')
    try:
        torch.save(checkpoint, temp)
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 1:
        raise ValueError('steps, batch-size, eval-every y save-every deben ser >= 1')
    if args.lr <= 0 or args.cpu_threads < 1:
        raise ValueError('lr y cpu-threads deben ser positivos')
    if args.split_mode not in {'dialogues', 'bytes'}:
        raise ValueError('split-mode desconocido')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    cfg = ModelConfig(context=args.context, d_model=args.d_model, layers=args.layers,
                      heads=args.heads, dropout=args.dropout)
    content = Path(args.data).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if args.split_mode == 'dialogues':
        train_data, val_data = split_dialogues(content.decode('utf-8'), seed=args.seed)
    else:
        train_data, val_data = split_data(read_bytes(args.data))
    if min(len(train_data), len(val_data)) <= cfg.context:
        raise ValueError('Dataset insuficiente para contexto: reducí --context o agregá bloques')
    resume = None
    start_step = 0
    history = []
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resume.get('model_config') != cfg.to_dict():
            raise ValueError('El checkpoint pertenece a otra configuración: usá los mismos parámetros del modelo')
        if resume.get('dataset_sha256') != digest or resume.get('split_mode') != args.split_mode or resume.get('seed') != args.seed:
            raise ValueError('El dataset, modo de partición o semilla cambió: no se puede reanudar este entrenamiento')
        if not all(key in resume for key in ('optimizer_state_dict', 'torch_rng_state', 'history')):
            raise ValueError('Checkpoint antiguo sin estado completo: entrená de nuevo con v0.2')
        start_step = int(resume['step'])
        if args.steps <= start_step:
            raise ValueError('--steps debe superar el paso guardado en el checkpoint')
        history = list(resume['history'])
    model = ByteTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    if resume is not None:
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        torch.set_rng_state(resume['torch_rng_state'])
        if device == 'cuda' and 'cuda_rng_states' in resume:
            torch.cuda.set_rng_state_all(resume['cuda_rng_states'])
        elif device != 'cpu' and device != 'cuda':
            raise ValueError('La reanudación exacta en este dispositivo aún no está soportada')
    print(f'Dispositivo={device} | parámetros={sum(p.numel() for p in model.parameters()):,} | '
          f'bytes={len(content):,} | pasos {start_step} → {args.steps}')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(step):
        state = {
            'format_version': 2,
            'model_config': cfg.to_dict(), 'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'step': step, 'seed': args.seed, 'split_mode': args.split_mode,
            'dataset_sha256': digest, 'history': history,
        }
        if device == 'cuda':
            state['cuda_rng_states'] = torch.cuda.get_rng_state_all()
        _atomic_save(state, output)
        output.with_suffix('.metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x,y = random_batch(train_data, args.batch_size, cfg.context, device)
        _,loss = model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            validation_loss = evaluate(model, val_data, cfg.context, device)
            point = {'step':step, 'train_loss':round(loss.item(),4),
                     'val_loss':round(validation_loss,4)}
            history.append(point)
            print(f"Paso {step}/{args.steps} | train={point['train_loss']} | val={point['val_loss']}")
        if step % args.save_every == 0 or step == args.steps:
            save(step)
    print(f'Checkpoint guardado en {output}')
    return output


def parse_args():
    p = argparse.ArgumentParser(description='Entrenar MJON IA v0.2')
    p.add_argument('--data',default='data/demo.txt')
    p.add_argument('--out',default='checkpoints/demo.pt')
    p.add_argument('--resume',default=None,help='Checkpoint v0.2 para continuar entrenando')
    p.add_argument('--steps',type=int,default=1000,help='Paso objetivo total, no pasos adicionales')
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--split-mode',choices=('dialogues','bytes'),default='dialogues')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--d-model',type=int,default=128)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--device',choices=('auto','cpu','cuda','mps'),default='auto')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
