"""Comprobar tokenizador, entrenamiento reanudable y reentrenamiento seguro (v0.3)."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from mjon_ia.synthetic import make_dialogues
from mjon_ia.tokenizer import ByteBPETokenizer
from mjon_ia.train_v03 import run
from mjon_ia.generate_v03 import generate


class V03Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    @staticmethod
    def args(data,out,**kw):
        settings=dict(data=str(data),out=str(out),resume=None,steps=4,eval_every=2,save_every=2,
                      context=16,vocab_size=270,d_model=24,layers=1,heads=2,batch_size=2,
                      grad_accum=2,lr=.001,dropout=0.,seed=21,cpu_threads=1,device='cpu')
        settings.update(kw)
        return Namespace(**settings)

    def test_utf8_bpe_roundtrip(self):
        tokenizer=ByteBPETokenizer.train('Hola, ¿qué tal? 🚀\nCórdoba, español y ñandú.\n'*20,vocab_size=310)
        for text in ['Hola, ¿qué tal? 🚀','🧪 ácido nítrico, 30 °C','a\nb\t c','']:
            self.assertEqual(tokenizer.decode_bytes(tokenizer.encode(text)),text.encode('utf-8'))
            self.assertEqual(ByteBPETokenizer.from_dict(tokenizer.to_dict()).decode(tokenizer.encode(text)),text)
        self.assertGreater(tokenizer.vocab_size,256)

    def test_invalid_merge_rejected(self):
        with self.assertRaises(ValueError):
            ByteBPETokenizer.from_dict({'type':'byte_bpe_v1','merges':[[256,1]],'vocab_size':257})

    def test_checkpoint_resume(self):
        with TemporaryDirectory() as td:
            root=Path(td);data=root/'train.txt'
            data.write_text('\n\n'.join(make_dialogues(13,130)),encoding='utf-8')
            full=root/'full.pt';resume=root/'resume.pt'
            run(self.args(data,full,steps=4))
            run(self.args(data,resume,steps=2))
            run(self.args(data,resume,resume=str(resume),steps=4))
            x=torch.load(full,weights_only=True);y=torch.load(resume,weights_only=True)
            self.assertEqual(x['tokenizer'],y['tokenizer'])
            for key in x['state_dict']:
                self.assertTrue(torch.equal(x['state_dict'][key],y['state_dict'][key]),key)
            self.assertEqual(x['history'][-1],y['history'][-1])
            self.assertTrue(generate(str(full),'Usuario: Hola\nAsistente:',new_tokens=4))

    def test_changed_dataset_rejected(self):
        with TemporaryDirectory() as td:
            root=Path(td);data=root/'train.txt'
            data.write_text('\n\n'.join(make_dialogues(12,120)),encoding='utf-8')
            ck=root/'trained.pt';run(self.args(data,ck,steps=2))
            data.write_text(data.read_text(encoding='utf-8')+'\n\nUn ejemplo adicional.',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'dataset'):
                run(self.args(data,ck,resume=str(ck),steps=3))


if __name__=='__main__':unittest.main()
