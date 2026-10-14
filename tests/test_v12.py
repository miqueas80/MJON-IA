"""Pruebas de Graphify: extracción, ejemplos negativos y gradientes MJON."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.knowledge_v11 import save as save_knowledge
from mjon_ia.graph_learn_v12 import run
from mjon_ia.model import ByteTransformer,ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from scripts.graphify_curriculum_v12 import extract_edges,make_examples,run as make_graph


def fake_graph(n=30):
    nodes=[]
    links=[]
    for i in range(n):
        source=f"s{i}"
        target=f"t{i}"
        nodes.extend([
            {"id":source,"label":f"func{i}","type":"function",
             "source_file":f"src/mod{i}.py"},
            {"id":target,"label":f"target{i}","type":"function",
             "source_file":f"src/util{i}.py"}
        ])
        links.append({"source":source,"target":target,
                      "relation":"calls","confidence":"EXTRACTED"})
    links.append({"source":"s0","target":"t1","relation":"calls",
                  "confidence":"INFERRED"})
    return {"nodes":nodes,"links":links}


class GraphifyNeuronsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_only_extracted_and_deterministic(self):
        data=fake_graph()
        edges=extract_edges(data)
        self.assertEqual(len(edges),30)
        self.assertEqual(make_examples(edges),make_examples(edges))
        generated=make_examples(edges)
        true=set(edges)
        anchors=[{x["positive_anchor"] for x in generated[key]} for key in
                 ("train","dev","test")]
        self.assertFalse(anchors[0]&anchors[1] or anchors[0]&anchors[2] or anchors[1]&anchors[2])
        self.assertTrue(all((x["source"],x["relation"],x["target"]) not in true
                            for batch in generated.values() for x in batch if x["answer"]=="No."))
        self.assertEqual(set(x["answer"] for x in generated["train"]),{"Sí.","No."})

    def test_graph_training_updates_only_original_neurons(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            graph=folder/"graph.json"
            graph.write_text(json.dumps(fake_graph()),encoding="utf-8")
            make_graph(graph,folder/"graph_data")
            save_knowledge(folder/"knowledge")
            know=json.loads((folder/"knowledge/train.json").read_text(encoding="utf-8"))
            codec=ByteBPETokenizer.train(
                "\n".join(r["question"]+r["answer"] for r in know[:30]),vocab_size=280)
            cfg=ModelConfig(context=512,vocab_size=codec.vocab_size,
                            d_model=16,layers=1,heads=2,dropout=0.)
            model=ByteTransformer(cfg)
            base=folder/"v11.pt"
            torch.save({"version":11,"model_config":cfg.to_dict(),
                        "tokenizer":codec.to_dict(),"state_dict":model.state_dict()},base)
            school=[{"question":f"Ejercicio {i}?","answer":"Sí."} for i in range(25)]
            memory=[{"question":f"Memoria {i}?","answer":"Registrada."} for i in range(25)]
            sh,st=[],[]
            for subject in ("matematica","lengua","ciencias","logica"):
                for i in range(16):
                    sh.append({"question":f"Escuela {subject} {i}?",
                               "answer":"Bien.","subject":subject})
                    st.append({"question":f"Parafraseo {subject} {i}?",
                               "answer":"Bien.","subject":subject})
            memhold=[{"question":f"Memoria reservada {i}?",
                       "answer":"Bien.","case":"recall"} for i in range(8)]
            paths={}
            for name,data in (("school",school),("memory",memory),
                              ("school_holdout",sh),("school_transfer",st),
                              ("memory_holdout",memhold)):
                p=folder/(name+".json")
                p.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                paths[name]=str(p)
            out=folder/"v12.pt"
            a=Namespace(base=str(base),graph_manifest=str(folder/"graph_data/manifest.json"),
                graph=str(folder/"graph_data/train.json"),graph_dev=str(folder/"graph_data/dev.json"),
                graph_test=str(folder/"graph_data/test.json"),
                knowledge=str(folder/"knowledge/train.json"),
                knowledge_dev=str(folder/"knowledge/paraphrase.json"),
                out=str(out),resume=None,steps=1,batch_size=2,
                graph_fraction=.35,knowledge_fraction=.35,memory_fraction=.12,
                kd_weight=.55,lr=.00006,eval_every=1,save_every=1,
                seed=1212,cpu_threads=1,**paths)
            run(a)
            best=torch.load(out,weights_only=True)
            self.assertEqual(best["version"],12)
            self.assertEqual(best["graphify_source"],"Graphify-Labs/graphify")
            self.assertTrue(0<=best["step"]<=1)
            self.assertTrue((folder/"v12.last.pt").exists())
            a.resume=str(folder/"v12.last.pt");a.steps=2
            run(a)
            last=torch.load(folder/"v12.last.pt",weights_only=True)
            self.assertEqual(last["step"],2)
            self.assertTrue(last["optimizer_state_dict"])


if __name__=="__main__":unittest.main()
