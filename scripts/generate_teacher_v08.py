"""Genera corpus de destilación REAL con Qwen2.5-0.5B-Instruct (Apache-2.0).

Requiere transformers/huggingface_hub y descarga pesos preentrenados solo
durante el build. Nunca se incluyen pesos Qwen dentro del checkpoint MJON.
"""
import argparse
import re
from pathlib import Path

import torch

from mjon_ia.teacher_v08 import build_prompts, manifest, save_manifest, verify_separation


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("Respondé en español, de forma directa y correcta. "
          "Una sola oración breve, sin introducciones ni listas. "
          "Si no conocés la respuesta, decí: No lo sé.")


def clean_answer(text):
    answer = text.strip()
    answer = re.split(r"\n(?:Usuario|Asistente|User|Assistant)\s*:", answer, maxsplit=1)[0]
    answer = answer.split("\n", 1)[0].strip()
    answer = re.sub(r"\s+", " ", answer)
    if len(answer) < 9 or len(answer) > 300:
        return None
    if answer.lower().startswith(("no lo sé", "no lo se", "disculpa", "lo siento")):
        return None
    if answer.count("¿") > 1 or answer.endswith((":", ",", ";")):
        return None
    return answer


def batches(values, size):
    if size < 1:
        raise ValueError("batch debe ser positivo")
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.inference_mode()
def run(args):
    # Importación diferida: los tests y entrenamientos MJON no necesitan Transformers.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import model_info

    if args.model != DEFAULT_MODEL:
        raise ValueError("Por seguridad/procedencia solo se admite el maestro revisado")
    if args.max_new_tokens < 8 or args.max_new_tokens > 96:
        raise ValueError("max_new_tokens fuera de límites")
    torch.manual_seed(808)
    torch.set_num_threads(args.cpu_threads)
    training, holdout = build_prompts(seed=args.seed, train_limit=args.train_limit)
    verify_separation(training, holdout)
    info = model_info(args.model)
    if info.cardData and info.cardData.get("license") not in ("apache-2.0",):
        raise ValueError("Licencia del modelo maestro cambió, abortando")
    revision = info.sha
    print("Maestro autorizado:", args.model, "revision", revision, "license=apache-2.0", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, trust_remote_code=False,
        torch_dtype=torch.float32, low_cpu_mem_usage=False,
    ).eval()
    generated_rows = []
    total = training + holdout
    for block in batches(total, args.batch_size):
        messages = [
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": row["question"]}]
            for row in block
        ]
        rendered = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in messages]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True,
                           max_length=512)
        outputs = model.generate(
            **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        prefix = inputs["input_ids"].shape[1]
        for row, generated in zip(block, outputs):
            raw = tokenizer.decode(generated[prefix:], skip_special_tokens=True)
            result = clean_answer(raw)
            generated_rows.append({**row, "answer": result, "accepted": result is not None})
        print("Respuestas generadas:", len(generated_rows), "/", len(total), flush=True)
    train_rows = [dict(question=x["question"],answer=x["answer"],subject=x["subject"],
                       concept=x["concept"]) for x in generated_rows[:len(training)] if x["accepted"]]
    test_rows = [dict(question=x["question"],answer=x["answer"],subject=x["subject"],
                      concept=x["concept"]) for x in generated_rows[len(training):] if x["accepted"]]
    if len(train_rows) < max(15,int(len(training)*.5)) or len(test_rows)<3:
        raise ValueError(f"Ejemplos del profesor insuficientes: {len(train_rows)} entrenamiento, {len(test_rows)} evaluación")
    result=manifest(args.model,revision,train_rows,test_rows)
    result["candidate_questions"] = len(training)
    result["accepted_training"] = len(train_rows)
    result["accepted_holdout"] = len(test_rows)
    save_manifest(args.out,result)
    print("Corpus del maestro:",len(train_rows),"ejemplos |",len(test_rows),
          "reservados, guardados en",args.out,flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--out",default="data/teacher_v08.json")
    p.add_argument("--seed",type=int,default=808)
    p.add_argument("--train-limit",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=40)
    p.add_argument("--cpu-threads",type=int,default=2)
    run(p.parse_args())
