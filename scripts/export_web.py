"""Exportar un checkpoint propio de PyTorch a formato int8 para la web.

Los pesos generados se almacenan únicamente en el artefacto de GitHub Pages,
sin modificar otras ramas ni requerir un servidor de inferencia.
"""
import argparse
import base64
import json
import zlib
from pathlib import Path

import torch


def run(checkpoint: str, destination: str) -> None:
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = []
    binary = bytearray()
    for name, tensor in obj["state_dict"].items():
        vec = tensor.reshape(-1)
        scale = max(vec.abs().max().item() / 127.0, 1e-12)
        q = (vec / scale).round().clamp(-127, 127).to(torch.int8).tolist()
        binary.extend(v & 255 for v in q)
        manifest.append([name, list(tensor.shape), scale, len(q)])

    meta = json.dumps({
        "format": "mjonia.int8.binary.v1",
        "config": obj["model_config"],
        "steps": obj["step"],
        "tensors": manifest,
    }, separators=(",", ":")).encode("utf-8")
    if len(meta) > 65535:
        raise ValueError("Manifiesto demasiado largo")
    payload = len(meta).to_bytes(2, "big") + meta + binary
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode("ascii")
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(encoded, encoding="ascii")
    print(f"Web model: {len(encoded)} caracteres, {sum(len(x) for x in obj['state_dict'].values())} filas")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/browser.pt")
    p.add_argument("--out", default="docs/web/weights.txt")
    a = p.parse_args()
    run(a.checkpoint, a.out)
