"""Pruebas de Corpus Forge y aislamiento del benchmark v0.4."""
import json
import tempfile
import unittest
from pathlib import Path
from mjon_ia.corpus_forge import generate,benchmark,build
from mjon_ia.tokenizer import ByteBPETokenizer


class V04Tests(unittest.TestCase):
    def test_reproducible_unique_records(self):
        first=generate(seed=8,examples=160)
        self.assertEqual(first,generate(seed=8,examples=160))
        self.assertEqual(len(first),len(set(first)))
        self.assertEqual(len(first),160)
        self.assertTrue(all(x.startswith('Usuario: ') and '\nAsistente: ' in x for x in first))

    def test_out_of_template_probes(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus=Path(temp)/'corpus.txt'
            probes=Path(temp)/'probes.json'
            build(corpus,probes,examples=200,seed=9)
            text=corpus.read_text(encoding='utf-8')
            data=json.loads(probes.read_text(encoding='utf-8'))
            self.assertEqual(len(data['probes']),36)
            self.assertTrue(all('Usuario: '+x['question']+'\n' not in text for x in data['probes']))
            self.assertTrue(all(x['expected'] and x['question'] for x in data['probes']))

    def test_tokenizer_train_only_on_train_prefix(self):
        corpus='\n'.join(generate(seed=4,examples=150))
        codec=ByteBPETokenizer.train(corpus[:2000],vocab_size=270)
        value='¡Qué día, Córdoba! 🧪'
        self.assertEqual(codec.decode(codec.encode(value)),value)
        self.assertTrue(codec.vocab_size>256)


if __name__=='__main__':unittest.main()
