"""Regresión: integridad de datos, medidas repetibles y reinicio exacto CPU."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from mjon_ia.data import split_dialogues
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.train import evaluate, run


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _args(self, data, out, **kwargs):
        options = dict(data=str(data), out=str(out), resume=None, steps=4,
                       eval_every=2, save_every=2, split_mode='dialogues',
                       batch_size=2, context=12, d_model=16, layers=1, heads=2,
                       dropout=0.1, lr=0.001, seed=7, cpu_threads=1, device='cpu')
        options.update(kwargs)
        return Namespace(**options)

    def _data(self, folder):
        path = folder / 'corpus.txt'
        path.write_text('\n\n'.join(f'Usuario: Pregunta {i}?\nAsistente: Respuesta {i} comprobable.' for i in range(24)), encoding='utf-8')
        return path

    def test_split_does_not_leak_duplicates(self):
        corpus = '\n\n'.join([f'bloque-{x}' for x in range(12)] + ['bloque-0', 'bloque-1'])
        a, b = split_dialogues(corpus, seed=42)
        ta, tb = a.tolist(), b.tolist()
        self.assertTrue(len(ta) and len(tb))
        train = bytes(ta).decode().strip().split('\n\n')
        valid = bytes(tb).decode().strip().split('\n\n')
        self.assertFalse(set(train).intersection(valid))
        c, d = split_dialogues(corpus, seed=42)
        self.assertTrue(torch.equal(a, c) and torch.equal(b, d))

    def test_validation_is_stable(self):
        m = ByteTransformer(ModelConfig(context=12, d_model=16, layers=1, heads=2, dropout=.1))
        val = torch.arange(300, dtype=torch.long) % 256
        first = evaluate(m, val, 12, 'cpu')
        second = evaluate(m, val, 12, 'cpu')
        self.assertAlmostEqual(first, second, places=7)

    def test_resume_matches_uninterrupted_weights(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            data=self._data(root)
            full=root/'full.pt'
            partial=root/'partial.pt'
            run(self._args(data,full,steps=4))
            run(self._args(data,partial,steps=2))
            run(self._args(data,partial,resume=str(partial),steps=4))
            x=torch.load(full,weights_only=True)
            y=torch.load(partial,weights_only=True)
            self.assertEqual((x['step'],y['step']), (4,4))
            self.assertEqual(x['dataset_sha256'], y['dataset_sha256'])
            for name in x['state_dict']:
                self.assertTrue(torch.equal(x['state_dict'][name],y['state_dict'][name]),name)

    def test_resume_rejects_changed_corpus(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            data=self._data(root)
            ck=root/'part.pt'
            run(self._args(data,ck,steps=2))
            data.write_text(data.read_text(encoding='utf-8')+'\n\nNuevo material que cambia los datos.', encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'dataset'):
                run(self._args(data,ck,resume=str(ck),steps=3))

    def test_resume_rejects_legacy_checkpoint(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            data=self._data(root)
            ck=root/'part.pt'
            run(self._args(data,ck,steps=2))
            payload=torch.load(ck,weights_only=True)
            del payload['optimizer_state_dict']
            torch.save(payload,ck)
            with self.assertRaisesRegex(ValueError,'antiguo'):
                run(self._args(data,ck,resume=str(ck),steps=3)


if __name__=='__main__':
    unittest.main()
