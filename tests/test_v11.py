"""MJON Original v1.1: hechos, límites, aislamiento, antirregresión."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.knowledge_v11 import parse, build, save
from mjon_ia.learn_v11 import run
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_factual_curriculum_and_isolation(self):
        subjects=parse()
        self.assertEqual(len(subjects),10)
        self.assertEqual(sum(len(rows) for rows in subjects.values()),120)
        train,paraphrase,unknown=build()
        self.assertEqual((len(train),len(paraphrase),len(unknown)),(400,100,40))
        self.assertEqual(train,build()[0])
        self.assertEqual(len({r["concept"] for r in train}),100)
        self.assertFalse({r["concept"] for r in train} &
                         {r["concept"] for r in unknown})
        self.assertFalse({r["question"] for r in train} &
                         {r["question"] for r in paraphrase+unknown})
        self.assertTrue(any(r["subject"]=="limites" and
                            r["answer"].startswith("No ") for r in train))
        self.assertTrue(any(r["subject"]=="biologia" for r in unknown))

    def test_real_small_network_trains_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            save(path/"knowledge")
            records=json.loads((path/"knowledge/train.json").read_text(encoding="utf-8"))
            tokenizer=ByteBPETokenizer.train(
                "\n".join(r["question"]+" "+r["answer"] for r in records[:70]),vocab_size=285)
            cfg=ModelConfig(context=256,vocab_size=tokenizer.vocab_size,
                            d_model=16,layers=1,heads=2,dropout=0)
            base=path/"v10.pt"
            torch.save({"version":10,"model_config":cfg.to_dict(),
                        "tokenizer":tokenizer.to_dict(),
                        "state_dict":ByteTransformer(cfg).state_dict()},base)
            school=[{"question":f"Ejercicio histórico {i}?","answer":"Es correcto."} for i in range(25)]
            memory=[{"question":f"Recuerdo conocido {i}?","answer":"Anotado."} for i in range(25)]
            sh,st=[],[]
            for subject in ("matematica","lengua","ciencias","logica"):
                for i in range(16):
                    sh.append({"question":f"Examen {subject} {i}?","answer":"Cierto.","subject":subject})
                    st.append({"question":f"Reformulación {subject} {i}?","answer":"Cierto.","subject":subject})
            mh=[{"question":f"Memoria reservada {i}?","answer":"Anotado.","case":"recall"} for i in range(8)]
            mapping={}
            for key,data in (("school",school),("memory",memory),
                              ("school_holdout",sh),("school_transfer",st),
                              ("memory_holdout",mh)):
                filename=path/(key+".json")
                filename.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                mapping[key]=str(filename)
            output=path/"v11.pt"
            args=Namespace(
                base=str(base),knowledge=str(path/"knowledge/train.json"),
                paraphrase=str(path/"knowledge/paraphrase.json"),
                unknown=str(path/"knowledge/unknown.json"),
                out=str(output),resume=None,steps=1,batch_size=2,
                knowledge_fraction=.42,memory_fraction=.23,kd_weight=.55,
                lr=.00008,eval_every=1,save_every=1,seed=1111,cpu_threads=1,
                **mapping
            )
            run(args)
            ck=torch.load(output,weights_only=True)
            self.assertEqual(ck["version"],11)
            self.assertTrue(0<=ck["step"]<=1)
            self.assertTrue((path/"v11.last.pt").exists())
            args.resume=str(path/"v11.last.pt")
            args.steps=2
            run(args)
            last=torch.load(path/"v11.last.pt",weights_only=True)
            self.assertEqual(last["step"],2)
            self.assertIn("optimizer_state_dict",last)
            self.assertEqual(last["curriculum_source"],"mjon-author-written-cc0")


if __name__=="__main__":unittest.main()
