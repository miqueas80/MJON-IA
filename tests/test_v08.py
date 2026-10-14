"""Regresiones v0.8: procedencia del profesor, aislamiento y alumno real."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from mjon_ia.teacher_v08 import TOPICS, build_prompts, manifest, verify_separation
from mjon_ia.distill_v08 import prepare, run
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer


class TeacherStudentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_teacher_prompts_are_isolated(self):
        train,holdout=build_prompts()
        self.assertEqual(len(train),42)
        self.assertEqual(len(holdout),6)
        self.assertTrue(verify_separation(train,holdout))
        self.assertEqual(train,build_prompts()[0])
        self.assertEqual(len({x["subject"] for x in train}),len(TOPICS))
        with self.assertRaises(ValueError):
            verify_separation(train,[dict(train[0])])

    def test_manifest_declares_source(self):
        train,holdout=build_prompts()
        sample=manifest("Qwen/Qwen2.5-0.5B-Instruct","test-revision",train[:2],holdout[:2])
        self.assertEqual(sample["teacher_weight_license"],"Apache-2.0")
        self.assertEqual(sample["teacher_revision"],"test-revision")
        self.assertEqual(sample["method"],"response-level supervised distillation")

    def test_train_distillation_checkpoint_and_resume(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=root/"v07.pt"
            prompts=[
                {"question":"¿Qué es una célula?","answer":"Unidad básica de los seres vivos.","subject":"biologia","concept":"biologia:0"},
                {"question":"¿Qué es un algoritmo?","answer":"Secuencia ordenada de instrucciones.","subject":"informatica","concept":"informatica:0"},
                {"question":"¿Qué es una fracción?","answer":"Una parte de un total.","subject":"matematica","concept":"matematica:0"}
            ]
            held=[
                {"question":"¿Qué es la fotosíntesis?","answer":"Proceso para obtener energía a partir de la luz.","subject":"biologia","concept":"biologia:2"},
            ]
            codec=ByteBPETokenizer.train("\n".join(x["question"]+" "+x["answer"] for x in prompts),vocab_size=280)
            cfg=ModelConfig(context=256,vocab_size=codec.vocab_size,d_model=16,layers=1,heads=2,dropout=0.)
            original=ByteTransformer(cfg)
            torch.save({"version":7,"model_config":cfg.to_dict(),
                        "tokenizer":codec.to_dict(),"state_dict":original.state_dict()},base)
            corpus=root/"teacher.json"
            data=manifest("Qwen/Qwen2.5-0.5B-Instruct","test-sha",prompts,held)
            corpus.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
            replay=root/"school.json";memory=root/"memory.json"
            replay.write_text(json.dumps(prompts,ensure_ascii=False),encoding="utf-8")
            memory.write_text(json.dumps(prompts[:2],ensure_ascii=False),encoding="utf-8")
            ck=root/"v08.pt"
            args=Namespace(base=str(base),teacher=str(corpus),replay=str(replay),memory=str(memory),
                           out=str(ck),resume=None,steps=1,batch_size=2,teacher_fraction=.7,
                           memory_fraction=.1,lr=.00012,eval_every=1,save_every=1,seed=8,cpu_threads=1)
            run(args)
            args.steps=2;args.resume=str(ck)
            run(args)
            result=torch.load(ck,weights_only=True)
            self.assertEqual(result["step"],2)
            self.assertEqual(result["version"],8)
            self.assertEqual(result["teacher_id"],"Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(result["model_config"]["context"],256)
            self.assertTrue(result["optimizer_state_dict"])

    def test_rejects_dataset_leak(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            train=[{"question":"Hola","answer":"Saludo","subject":"lengua","concept":"x"}]
            sample=manifest("teacher","sha",train,[{"question":"Adiós","answer":"Despedida","subject":"lengua","concept":"y"}])
            sample["holdout"][0]["concept"]="x"
            path=root/"teacher.json";path.write_text(json.dumps(sample),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_separation(sample["training"],sample["holdout"])


if __name__=="__main__":
    unittest.main()
