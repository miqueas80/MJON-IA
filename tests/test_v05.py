"""Pruebas del ajuste SFT: datos independientes, máscara de respuesta, checkpoint."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import torch
from mjon_ia.instruction_v05 import build_dataset,write_dataset,TRAIN_ITEMS,TEST_ITEMS
from mjon_ia.model import ByteTransformer,ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from mjon_ia.finetune_v05 import encode_record,collate,run

class FirstWordsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_holdout_is_disjoint(self):
        a,b=build_dataset(seed=504)
        self.assertTrue(set(x["question"] for x in a).isdisjoint(x["question"] for x in b))
        self.assertTrue(all(not any(item in x["question"] for item in TEST_ITEMS) for x in a))
        self.assertTrue(all(not any(item in x["question"] for item in TRAIN_ITEMS) for x in b))
        self.assertEqual(a,build_dataset(seed=504)[0])

    def test_only_answer_has_gradient(self):
        codec=ByteBPETokenizer.train("Usuario: ficha roja\nAsistente: rojo\n"*20,vocab_size=278)
        x,y=encode_record(codec,"¿De qué color es el frasco rojo?","rojo",context=120)
        suffix=codec.encode(" rojo\n")
        self.assertTrue(all(z==-100 for z in y[:-len(suffix)]))
        self.assertEqual(sum(z!=-100 for z in y),len(suffix))
        xx,yy=collate([(x,y),(x[:-5],y[:-5])])
        self.assertEqual(xx.shape,yy.shape)
        self.assertEqual(xx.shape[0],2)

    def test_finetune_checkpoint_saves_compatibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train,_=write_dataset(root)
            corpus="\n\n".join("Usuario: "+row["question"]+"\nAsistente: "+row["answer"] for row in train[:300])
            codec=ByteBPETokenizer.train(corpus,vocab_size=290)
            cfg=ModelConfig(context=128,vocab_size=codec.vocab_size,d_model=24,layers=1,heads=2,dropout=0)
            model=ByteTransformer(cfg)
            base=root/"base.pt";dest=root/"tuned.pt"
            torch.save({"version":3,"model_config":cfg.to_dict(),"tokenizer":codec.to_dict(),"state_dict":model.state_dict(),"step":1},base)
            args=Namespace(base=str(base),train=str(root/"train.json"),holdout=str(root/"holdout.json"),
                           out=str(dest),steps=2,batch_size=2,grad_accum=1,save_every=1,
                           eval_every=1,lr=.0003,seed=5,cpu_threads=1,resume=None)
            run(args)
            state=torch.load(dest,weights_only=True)
            self.assertEqual(state["version"],5)
            self.assertEqual(state["step"],2)
            self.assertEqual(state["model_config"],cfg.to_dict())
            self.assertTrue(state["state_dict"] and state["tokenizer"])

if __name__=="__main__":unittest.main()
