"""Regresión MJON IA v0.6: aislamiento, correcciones, máscara SFT y reinicio."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.finetune_v05 import encode_record
from mjon_ia.finetune_v06 import run
from mjon_ia.instruction_v05 import TEST_ITEMS, write_dataset
from mjon_ia.memory_v06 import build, make_question, save
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer


class MemoryV06Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_turns_and_corrections(self):
        q = make_question("color", "frasco", "rojo", "azul", "update_a")
        self.assertIn("rojo", q)
        self.assertIn("azul", q)
        self.assertIn("\nAsistente: Anotado.\nUsuario:", q)
        self.assertTrue(q.endswith("color actual del frasco?"))
        with self.assertRaises(ValueError):
            make_question("color", "frasco", "rojo", "azul", "unknown")

    def test_test_objects_never_in_training(self):
        train, reserved = build(seed=606, limit=500)
        self.assertTrue(train and reserved)
        self.assertFalse(set(x["question"] for x in train) & set(x["question"] for x in reserved))
        self.assertTrue(all(not any(item in x["question"] for item in TEST_ITEMS) for x in train))
        self.assertTrue(any(x["case"] == "correction" and x["answer"] != "rojo" for x in train))
        self.assertEqual(train, build(seed=606, limit=500)[0])

    def test_answer_only_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, holdout = save(root / "v06", limit=60)
            old_train, _ = write_dataset(root / "v05")
            # Se entrena este BPE de prueba con texto propio, nunca con los casos reservados.
            sample = "\n".join(x["question"] + "\n" + x["answer"] for x in train)
            codec = ByteBPETokenizer.train(sample, vocab_size=320)
            cfg = ModelConfig(context=256, vocab_size=codec.vocab_size,
                              d_model=16, layers=1, heads=2, dropout=0)
            question = train[0]["question"]
            encoded, labels = encode_record(codec, question, train[0]["answer"], cfg.context)
            self.assertEqual(len(encoded), len(labels))
            suffix_length = len(codec.encode(" " + train[0]["answer"] + "\n"))
            self.assertTrue(all(x == -100 for x in labels[:-suffix_length]))
            self.assertEqual(sum(x != -100 for x in labels), suffix_length)
            model = ByteTransformer(cfg)
            base = root / "v05.pt"
            torch.save({"version": 5, "model_config": cfg.to_dict(), "tokenizer": codec.to_dict(),
                        "state_dict": model.state_dict(), "step": 1}, base)
            checkpoint = root / "v06.pt"
            args = Namespace(base=str(base), train=str(root / "v06/train.json"),
                             holdout=str(root / "v06/holdout.json"),
                             replay=str(root / "v05/train.json"), out=str(checkpoint),
                             steps=1, batch_size=2, eval_every=1, save_every=1,
                             lr=.0002, replay_fraction=.2, seed=6, cpu_threads=1,
                             resume=None)
            run(args)
            args.steps = 2
            args.resume = str(checkpoint)
            run(args)
            stored = torch.load(checkpoint, weights_only=True)
            self.assertEqual(stored["version"], 6)
            self.assertEqual(stored["step"], 2)
            self.assertTrue(stored["optimizer_state_dict"])


if __name__ == "__main__":
    unittest.main()
