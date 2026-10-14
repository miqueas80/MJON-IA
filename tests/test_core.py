import tempfile
import unittest
from pathlib import Path

import torch

from mjon_ia.data import random_batch, read_bytes, split_data
from mjon_ia.model import ByteTransformer, ModelConfig


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_model_dimensions_and_backprop(self):
        model = ByteTransformer(ModelConfig(context=16, d_model=32, layers=1, heads=2, dropout=0))
        inputs = torch.randint(0, 256, (2, 16))
        logits, loss = model(inputs, inputs)
        self.assertEqual(tuple(logits.shape), (2, 16, 256))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.token.weight.grad)

    def test_causality(self):
        model = ByteTransformer(ModelConfig(context=12, d_model=32, layers=1, heads=2, dropout=0)).eval()
        first = torch.randint(0, 256, (1, 12))
        second = first.clone()
        second[0, -1] = (second[0, -1] + 1) % 256
        with torch.no_grad():
            a, _ = model(first)
            b, _ = model(second)
        self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5))

    def test_generation_bytes(self):
        model = ByteTransformer(ModelConfig(context=8, d_model=32, layers=1, heads=2))
        x = torch.tensor([[72, 111]])
        generated = model.generate(x, new_tokens=4, top_k=10)
        self.assertEqual(tuple(generated.shape), (1, 6))
        self.assertTrue(((generated >= 0) & (generated <= 255)).all().item())

    def test_data_read_and_batch(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "test.txt"
            file.write_text("Hola. Buen día. " * 30, encoding="utf-8")
            tr, val = split_data(read_bytes(file))
            x, y = random_batch(tr, 3, 8, "cpu")
            self.assertEqual(tuple(x.shape), (3, 8))
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertGreater(len(val), 8)


if __name__ == "__main__":
    unittest.main()
