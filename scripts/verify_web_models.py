"""Verificación automática de cada checkpoint web antes del despliegue."""
import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path


def verify(location="docs/web"):
    directory = Path(location)
    catalog = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    assert catalog["format"] == "mjonia.models.v1" and len(catalog["models"]) >= 4
    seen = set()
    for entry in catalog["models"]:
        assert entry["id"] not in seen
        seen.add(entry["id"])
        assert Path(entry["file"]).name == entry["file"]
        payload = (directory / entry["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        decoded = zlib.decompress(base64.b64decode(payload, validate=True))
        meta_len = int.from_bytes(decoded[:2], "big")
        meta = json.loads(decoded[2:2+meta_len].decode("utf-8"))
        assert meta["format"] == "mjonia.int8.binary.v1"
        assert meta["steps"] == entry["steps"]
        assert meta["config"]["context"] == entry["context"]
        actual = sum(tensor[3] for tensor in meta["tensors"])
        assert actual == len(decoded) - 2 - meta_len
        assert actual > 0 and entry["parameters"] > 0
        print(f"OK: {entry['label']} | {entry['parameters']} parámetros | {actual} bytes cuantizados")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="docs/web")
    args = parser.parse_args()
    verify(args.directory)
