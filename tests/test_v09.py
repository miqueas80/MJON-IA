"""v0.9: transferencia SOLO por entrenamiento de pesos originales, sin motor externo."""
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from mjon_ia.teacher_v09 import build_prompts,variants
from mjon_ia.teacher_v08 import manifest,verify_separation
from mjon_ia.model import ByteTransformer,ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from mjon_ia.distill_v09 import run


class V09Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_many_questions_and_heldout_concepts(self):
        a,b=build_prompts()
        self.assertEqual(len(a),336)
        self.assertEqual(len(b),18)
        self.assertTrue(verify_separation(a,b))
        self.assertEqual(len({r["concept"] for r in a}),42)
        self.assertEqual(len({r["concept"] for r in b}),6)
        self.assertFalse({r["concept"] for r in a} & {r["concept"] for r in b})
        self.assertEqual(len(variants("¿Qué es un átomo?")),8)
        self.assertEqual((a,b),build_prompts())

    def test_best_checkpoint_and_resume(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            train,test=build_prompts(variants_per_concept=3,holdout_variants=1)
            for r in train+test:r["answer"]="Es una explicación muy corta."
            corpus=manifest("Qwen/Qwen2.5-0.5B-Instruct","test-sha",train,test)
            corpus["generation_version"]=9
            p=root/"teacher.json";p.write_text(json.dumps(corpus,ensure_ascii=False),encoding="utf-8")
            codec=ByteBPETokenizer.train(
                "\n".join(r["question"]+r["answer"] for r in train[:60]),vocab_size=285)
            cfg=ModelConfig(context=256,vocab_size=codec.vocab_size,d_model=16,
                            layers=1,heads=2,dropout=0)
            base=root/"base.pt"
            torch.save({"version":8,"model_config":cfg.to_dict(),
                        "tokenizer":codec.to_dict(),
                        "state_dict":ByteTransformer(cfg).state_dict()},base)
            school=root/"school.json";memory=root/"memory.json"
            school.write_text(json.dumps(train[:10],ensure_ascii=False),encoding="utf-8")
            memory.write_text(json.dumps(train[10:20],ensure_ascii=False),encoding="utf-8")
            out=root/"v09.pt"
            args=Namespace(base=str(base),teacher=str(p),replay=str(school),
                           memory=str(memory),out=str(out),resume=None,
                           steps=2,batch_size=2,teacher_fraction=.6,memory_fraction=.1,
                           lr=.0002,eval_every=1,save_every=1,seed=1,cpu_threads=1)
            run(args)
            checkpoint=torch.load(out,map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["version"],9)
            self.assertTrue(0<=checkpoint["step"]<=2)
            self.assertEqual(checkpoint["best_step"],checkpoint["step"])
            self.assertTrue((root/"v09.last.pt").exists())
            args.steps=3;args.resume=str(root/"v09.last.pt")
            run(args)
            latest=torch.load(root/"v09.last.pt",map_location="cpu",weights_only=True)
            self.assertEqual(latest["step"],3)
            self.assertIn("optimizer_state_dict",latest)
            self.assertEqual(latest["teacher_id"],"Qwen/Qwen2.5-0.5B-Instruct")


if __name__=="__main__":
    unittest.main()
