"""Pruebas rápidas del siguiente núcleo neuronal propio de MJON IA.

Comprueba RoPE, GQA, cache autoregresiva, pérdida, gradientes, corpus sin
filtraciones documentales y entrenamiento real CPU de dos pasos.
"""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.neural_core import MJONNeuralCore, NeuralConfig
from mjon_ia.scalable_train import split_documents, run as train_neural, learning_rate


class NeuralCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def config():
        return NeuralConfig(vocab_size=256,context=64,width=32,layers=2,heads=4,
                            kv_heads=2,dropout=0.)

    def test_gradients_and_weight_tying(self):
        torch.manual_seed(4)
        network=MJONNeuralCore(self.config())
        self.assertEqual(network.output.weight.data_ptr(),network.token.weight.data_ptr())
        x=torch.randint(0,256,(2,9))
        y=torch.randint(0,256,(2,9))
        logits,loss,cached=network(x,targets=y)
        self.assertEqual(logits.shape,(2,9,256))
        self.assertIsNone(cached)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(network.token.weight.grad).all())
        self.assertGreater(network.parameter_count(),1000)

    def test_cache_matches_full_attention(self):
        torch.manual_seed(42)
        network=MJONNeuralCore(self.config()).eval()
        x=torch.randint(0,256,(1,16))
        with torch.inference_mode():
            full,_,_=network(x)
            first,_,past=network(x[:,:11],use_cache=True)
            second,_,new_cache=network(x[:,11:],past_key_values=past,use_cache=True)
        self.assertEqual(len(new_cache),self.config().layers)
        self.assertEqual(new_cache[0][0].shape,(1,2,16,8))
        self.assertTrue(torch.allclose(full[:,:11],first,atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.allclose(full[:,11:],second,atol=2e-5,rtol=2e-5))
        generated=network.generate(x,max_new_tokens=3)
        self.assertEqual(generated.shape,(1,19))

    def test_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            NeuralConfig(vocab_size=256,context=128,width=64,layers=2,heads=4,kv_heads=3)
        network=MJONNeuralCore(self.config())
        with self.assertRaises(ValueError):
            network(torch.ones((1,65),dtype=torch.long))
        with self.assertRaises(ValueError):
            network(torch.ones((1,4),dtype=torch.float))

    def test_document_split_dedup_and_license(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"text.jsonl"
            docs=[{"text":f"Documento {i}: "+"hola planeta y ciencia "*12,
                   "source_id":"mjon-original","document_id":f"d{i}",
                   "license":"CC0-1.0"} for i in range(20)]
            docs.append(docs[0])
            target.write_text("\n".join(json.dumps(d,ensure_ascii=False) for d in docs)+"\n",
                              encoding="utf-8")
            _,train,val,tr,te=split_documents(target,seed=5)
            self.assertEqual(len(tr)+len(te),20)
            self.assertFalse(set(tr)&set(te))
            self.assertEqual(len(train)+len(val),20)
            self.assertEqual((tr,te),split_documents(target,seed=5)[3:5])
            docs[0]["license"]="unverified"
            target.write_text("\n".join(json.dumps(d,ensure_ascii=False) for d in docs),
                              encoding="utf-8")
            with self.assertRaises(ValueError):
                split_documents(target)
        self.assertGreater(learning_rate(3,12,.001,2),0)
        self.assertLess(learning_rate(12,12,.001,2),.0002)

    def test_train_own_weights_smoke_cpu(self):
        try:
            import sentencepiece
        except ImportError:
            self.skipTest("Instalá extras de entrenamiento neural: sentencepiece")
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            data=folder/"corpus.jsonl"
            # Diferentes documentos y vocabulario; ninguna descarga ni API.
            docs=[]
            topics=["átomo","célula","ecuación","código","energía","átomo estable",
                    "proyecto","conexión","transformación","inventario","ética","lógica",
                    "ecosistema","reacción","sistemas","algoritmo","química","ciencia",
                    "experimento","memoria"]
            for index,subject in enumerate(topics):
                text=" ".join(
                    f"Texto científico {subject} número {index}; el ejemplo describe "
                    f"causas, efectos, variables, preguntas y explicaciones en español. "
                    for _ in range(6))
                docs.append({"text":text,"source_id":"mjon-original",
                             "document_id":f"doc-{index}","license":"CC0-1.0"})
            data.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in docs)+"\n",
                            encoding="utf-8")
            output=folder/"neural.pt"
            args=Namespace(data=str(data),tokenizer=str(folder/"tiny.model"),
                out=str(output),resume=None,vocab_size=300,context=32,
                width=32,layers=2,heads=4,kv_heads=2,dropout=0.,
                steps=2,batch_size=2,grad_accum=1,lr=.001,
                weight_decay=.01,clip_grad=1.,warmup=0,
                eval_every=1,save_every=1,cpu_threads=1,
                seed=8,device="cpu",precision="fp32")
            train_neural(args)
            checkpoint=torch.load(output,weights_only=True)
            latest=torch.load(folder/"neural.last.pt",weights_only=True)
            self.assertEqual(checkpoint["format"],"mjon-neural-v2")
            self.assertEqual(latest["step"],2)
            self.assertTrue(1<=checkpoint["best_step"]<=2)
            self.assertIn("optimizer_state_dict",latest)
            self.assertGreater(checkpoint["total_train_tokens"],1000)
            self.assertGreater(checkpoint["total_valid_tokens"],32)
            self.assertTrue(output.with_suffix(".metrics.json").exists())
            self.assertEqual(checkpoint["config"]["layers"],2)


if __name__=="__main__":
    unittest.main()
