"""Regresión v0.10: antiolvido, selección sin regresión y entrenamiento propio."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.distill_v10 import choose_candidate, frozen_distribution_loss, run
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.teacher_v08 import manifest
from mjon_ia.tokenizer import ByteBPETokenizer


class V10Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_kl_zero_when_distributions_match(self):
        x=torch.tensor([[[1.2,-.1,.5],[.2,.4,.1]]],requires_grad=True)
        mask=torch.tensor([[-100,1]])
        result=frozen_distribution_loss(x,x.detach(),mask)
        self.assertLess(abs(float(result)),1e-5)
        result.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_checkpoint_requires_preservation_and_gain(self):
        base={"school":5,"transfer":2,"memory":7}
        self.assertFalse(choose_candidate(base,{"school":4,"transfer":3,"memory":7},5.,4.))
        self.assertFalse(choose_candidate(base,{"school":5,"transfer":3,"memory":6},5.,4.))
        self.assertFalse(choose_candidate(base,{"school":5,"transfer":2,"memory":7},5.,5.))
        self.assertTrue(choose_candidate(base,{"school":5,"transfer":2,"memory":7},5.,4.5))

    def test_tiny_train_resume_and_only_original_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=[{"question":f"Definí concepto número {i}.",
                    "answer":"Es una definición breve.","subject":"quimica",
                    "concept":f"quimica:{i}"} for i in range(120)]
            held=[{"question":f"Explicá concepto nuevo {i}.",
                   "answer":"Definición diferente.","subject":"biologia",
                   "concept":f"biologia:{i}"} for i in range(6)]
            teacher=manifest("Qwen/Qwen2.5-0.5B-Instruct","sha-test",train,held)
            teacher["generation_version"]=9
            corpus=root/"teacher.json"
            corpus.write_text(json.dumps(teacher,ensure_ascii=False),encoding="utf-8")
            codec=ByteBPETokenizer.train(
                "\n".join(r["question"]+" "+r["answer"] for r in train[:30]),vocab_size=280)
            cfg=ModelConfig(context=256,vocab_size=codec.vocab_size,
                            d_model=16,layers=1,heads=2,dropout=0)
            base=root/"v08.pt"
            torch.save({"version":8,"model_config":cfg.to_dict(),
                        "tokenizer":codec.to_dict(),
                        "state_dict":ByteTransformer(cfg).state_dict()},base)
            replay=[{"question":f"Ejercicio viejo {i}.","answer":"Bien."} for i in range(20)]
            memory=[{"question":f"Recuerdo viejo {i}.","answer":"Anotado."} for i in range(12)]
            school_holdout=[]
            transfer=[]
            for subject in ("matematica","lengua","ciencias","logica"):
                for i in range(16):
                    school_holdout.append({"question":f"Prueba {subject} {i}",
                                           "answer":"Cierto.","subject":subject})
                    transfer.append({"question":f"Reformulada {subject} {i}",
                                     "answer":"Cierto.","subject":subject})
            memory_holdout=[{"question":f"Memoria reservada {i}",
                             "answer":"Correcto.","case":"recall"} for i in range(8)]
            paths={}
            for name,rows in (("school",replay),("memory",memory),
                              ("holdout",school_holdout),("transfer",transfer),
                              ("memhold",memory_holdout)):
                p=root/(name+".json")
                p.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
                paths[name]=str(p)
            output=root/"v10.pt"
            opts=Namespace(base=str(base),teacher=str(corpus),replay=paths["school"],
                memory=paths["memory"],school_holdout=paths["holdout"],
                school_transfer=paths["transfer"],memory_holdout=paths["memhold"],
                out=str(output),resume=None,steps=1,batch_size=3,
                teacher_fraction=.3,memory_fraction=.25,kd_weight=.6,
                lr=.00005,eval_every=1,save_every=1,seed=1010,cpu_threads=1)
            run(opts)
            best=torch.load(output,weights_only=True)
            self.assertEqual(best["version"],10)
            self.assertIn("state_dict",best)
            self.assertEqual(best["teacher_id"],"Qwen/Qwen2.5-0.5B-Instruct")
            self.assertTrue(0<=best["step"]<=1)
            opts.resume=str(root/"v10.last.pt")
            opts.steps=2
            run(opts)
            last=torch.load(root/"v10.last.pt",weights_only=True)
            self.assertEqual(last["step"],2)
            self.assertIn("optimizer_state_dict",last)


if __name__=="__main__":
    unittest.main()
