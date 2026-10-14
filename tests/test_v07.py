"""MJON Escuela: separación de materias, cambio de contexto y reanudación."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from mjon_ia.school_v07 import SUBJECTS,TEST_NAMES,TEST_ANIMALS,generate,save
from mjon_ia.train_v07 import run,upgrade_model
from mjon_ia.model import ModelConfig,ByteTransformer
from mjon_ia.tokenizer import ByteBPETokenizer


class SchoolV07Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_four_subjects_and_transfer(self):
        a,b,c=generate(seed=707,per_subject=32,heldout_per_subject=8)
        self.assertEqual(len(a),128)
        self.assertEqual(len(b),32)
        self.assertEqual(len(c),32)
        self.assertEqual({x["subject"] for x in a},set(SUBJECTS))
        self.assertEqual(a,generate(seed=707,per_subject=32,heldout_per_subject=8)[0])
        seen={x["question"] for x in a}
        self.assertTrue(seen.isdisjoint(x["question"] for x in b))
        self.assertTrue(seen.isdisjoint(x["question"] for x in c))
        for record in a:
            if record["subject"] in ("lengua","ciencias"):
                self.assertFalse(any(item in record["question"] for item in TEST_NAMES))
            if record["subject"]=="logica":
                self.assertFalse(any(item in record["question"] for item in TEST_ANIMALS))

    def test_upgrade_context_preserves_logits(self):
        codec=ByteBPETokenizer.train("Usuario: colores y pruebas.\n"*30,vocab_size=266)
        cfg=ModelConfig(context=128,vocab_size=codec.vocab_size,d_model=16,layers=1,heads=2,dropout=0.)
        old=ByteTransformer(cfg).eval()
        checkpoint={"model_config":cfg.to_dict(),"state_dict":old.state_dict()}
        new=upgrade_model(checkpoint,256).eval()
        tokens=torch.randint(codec.vocab_size,(1,25))
        with torch.no_grad():
            expected=old(tokens)[0]
            actual=new(tokens)[0]
        self.assertEqual(new.config.context,256)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))

    def test_run_and_resume_with_small_checkpoint(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            save(root/"school",seed=707,per_subject=15,heldout=5)
            rows=json.loads((root/"school/train.json").read_text(encoding="utf-8"))
            tiny=ByteBPETokenizer.train(
                "\n".join(x["question"] for x in rows),vocab_size=285)
            cfg=ModelConfig(context=128,vocab_size=tiny.vocab_size,d_model=16,
                            layers=1,heads=2,dropout=0.)
            base=root/"base.pt"
            torch.save({"version":6,"model_config":cfg.to_dict(),
                        "tokenizer":tiny.to_dict(),
                        "state_dict":ByteTransformer(cfg).state_dict()},base)
            (root/"rep05.json").write_text(json.dumps(rows[:10]),encoding="utf-8")
            (root/"rep06.json").write_text(json.dumps(rows[10:20]),encoding="utf-8")
            out=root/"school.pt"
            args=Namespace(base=str(base),
                train=str(root/"school/train.json"),holdout=str(root/"school/holdout.json"),
                transfer=str(root/"school/transfer.json"),
                replay05=str(root/"rep05.json"),replay06=str(root/"rep06.json"),
                out=str(out),resume=None,context=256,steps=1,batch_size=2,
                lr=.0002,replay_v05=.1,replay_v06=.1,eval_every=1,
                save_every=1,seed=707,cpu_threads=1)
            run(args)
            args.resume=str(out)
            args.steps=2
            run(args)
            ck=torch.load(out,weights_only=True)
            self.assertEqual(ck["version"],7)
            self.assertEqual(ck["step"],2)
            self.assertEqual(ck["model_config"]["context"],256)
            self.assertTrue(ck["optimizer_state_dict"])


if __name__=="__main__":
    unittest.main()
