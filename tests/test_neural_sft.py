"""Prueba end-to-end: MJON aprende instrucciones en SU propia red Neural Core."""
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.neural_sft import (
    read_instructions,encode_supervised,pad_batch,run as fine_tune
)
from mjon_ia.scalable_train import run as pretrain
from scripts.build_neural_curriculum import build_corpus,build_pretrain
from scripts.eval_neural_sft import run as evaluate


class NeuralSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_corpus_splits_by_concept(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"knowledge.jsonl"
            rows=build_corpus(path)
            self.assertEqual(len(rows),540)
            train,valid,meta=read_instructions(path,seed=37)
            groups_train={r["group_id"] for r in train}
            groups_valid={r["group_id"] for r in valid}
            self.assertFalse(groups_train&groups_valid)
            self.assertEqual(len(groups_train|groups_valid),120)
            self.assertEqual(len(train)+len(valid),540)
            self.assertEqual((train,valid,meta),read_instructions(path,seed=37))
            faulty=[dict(r) for r in rows[:16]]
            faulty[1]={**faulty[1],
                       "instruction":faulty[0]["instruction"],
                       "group_id":"different-group"}
            path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in faulty),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                read_instructions(path)
            faulty[1]={**faulty[1],"group_id":faulty[0]["group_id"],
                       "license":"proprietary"}
            path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in faulty),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                read_instructions(path)

    def test_neural_sft_from_own_pretrained_weights(self):
        try:
            import sentencepiece as spm
        except ImportError:
            self.skipTest("Requiere sentencepiece, instalado por neural-smoke workflow")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            pretrain_rows=build_pretrain(root/"pretrain.jsonl")
            self.assertEqual(len(pretrain_rows),120)
            build_corpus(root/"knowledge.jsonl")
            pretrain_path=root/"original.pt"
            tokenizer_path=root/"mjon.model"
            pretrain(Namespace(
                data=str(root/"pretrain.jsonl"),
                tokenizer=str(tokenizer_path),out=str(pretrain_path),
                resume=None,vocab_size=512,context=256,width=32,
                layers=2,heads=4,kv_heads=2,dropout=0.,
                steps=2,batch_size=2,grad_accum=1,
                lr=.001,weight_decay=.01,clip_grad=1.,
                warmup=0,eval_every=1,save_every=1,
                cpu_threads=1,seed=91,device="cpu",precision="fp32",
            ))
            own=torch.load(pretrain_path,weights_only=True)
            self.assertEqual(own["format"],"mjon-neural-v2")
            self.assertEqual(own["tokenizer_sha256"],
                             hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
            tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            records,valid,_=read_instructions(root/"knowledge.jsonl")
            enc=encode_supervised(tokenizer,records[0],256)
            self.assertIsNotNone(enc)
            x,y=enc
            prefix_count=len(tokenizer.encode(
                "Usuario: "+records[0]["instruction"]+"\nAsistente:",out_type=int))
            self.assertTrue((y[:prefix_count]==-100).all())
            self.assertTrue((y[prefix_count:]!=-100).all())
            xb,yb=pad_batch([enc,enc],torch.device("cpu"))
            self.assertEqual(tuple(xb.shape),tuple(yb.shape))
            out=root/"mjon-instruct.pt"
            args=Namespace(
                base=str(pretrain_path),tokenizer=str(tokenizer_path),
                data=str(root/"knowledge.jsonl"),out=str(out),resume=None,
                steps=2,batch_size=2,grad_accum=1,
                lr=.001,weight_decay=.01,clip_grad=1.,
                warmup=0,eval_every=1,save_every=1,
                cpu_threads=1,seed=2026,device="cpu",precision="fp32",
            )
            fine_tune(args)
            trained=torch.load(out,map_location="cpu",weights_only=True)
            last=torch.load(root/"mjon-instruct.last.pt",weights_only=True)
            self.assertEqual(trained["format"],"mjon-neural-sft-v1")
            self.assertEqual(last["step"],2)
            self.assertFalse(trained["teacher_weights_used"])
            self.assertTrue(trained["validation_examples"]>0)
            self.assertTrue(trained["training_examples"]>100)
            self.assertGreaterEqual(trained["best_validation_loss"],0.)
            evaluation=evaluate(
                pretrain_path,out,tokenizer_path,root/"knowledge.jsonl",
                root/"evaluation.json",max_cases=3,seed=2026)
            self.assertEqual(len(evaluation["models"]),2)
            self.assertEqual(evaluation["models"][0]["total"],3)
            self.assertTrue((root/"evaluation.json").exists())


if __name__=="__main__":
    unittest.main()
