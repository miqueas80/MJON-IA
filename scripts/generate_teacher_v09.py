"""Amplía la enseñanza real de Qwen a MJON IA Original, sin servicio en producción.

Generación durante la compilación, con licencia, revisión y exclusión
de ejemplos demasiado extensos para el Transformer alumno.
"""
import argparse
from pathlib import Path

import torch

from mjon_ia.teacher_v09 import build_prompts
from mjon_ia.teacher_v08 import manifest, save_manifest
from mjon_ia.finetune_v05 import encode_record
from mjon_ia.tokenizer import ByteBPETokenizer
from generate_teacher_v08 import DEFAULT_MODEL, clean_answer, batches

SYSTEM = ("Respondé en español, de forma precisa y correcta. "
          "Una sola oración corta de hasta quince palabras. "
          "No agregues comentarios, introducciones ni listas. "
          "Si no sabés algo, decí: No lo sé.")


def valid_student_example(codec, context, question, answer):
    try:
        encode_record(codec,question,answer,context)
        return True
    except ValueError:
        return False


@torch.inference_mode()
def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    if args.model!=DEFAULT_MODEL:
        raise ValueError("Modelo maestro no autorizado")
    if not 8<=args.max_new_tokens<=96 or not 1<=args.batch_size<=12:
        raise ValueError("Límite de generación o lote inválido")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    training,holdout=build_prompts(
        seed=args.seed,variants_per_concept=args.variants_per_concept,
        holdout_variants=args.holdout_variants)
    student=torch.load(args.student_base,map_location="cpu",weights_only=True)
    if student.get("version")!=8:raise ValueError("Alumno debe proceder de MJON v0.8")
    codec=ByteBPETokenizer.from_dict(student["tokenizer"])
    context=student["model_config"]["context"]
    info=model_info(args.model)
    license=(info.cardData or {}).get("license")
    if license!="apache-2.0":raise ValueError("No pudo verificarse la licencia Apache-2.0 del maestro")
    revision=info.sha
    tok=AutoTokenizer.from_pretrained(args.model,revision=revision,trust_remote_code=False)
    tok.padding_side="left"
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    master=AutoModelForCausalLM.from_pretrained(
        args.model,revision=revision,trust_remote_code=False,
        torch_dtype=torch.float32,low_cpu_mem_usage=False).eval()
    all_questions=training+holdout
    output=[]
    for block in batches(all_questions,args.batch_size):
        prompts=[tok.apply_chat_template(
            [{"role":"system","content":SYSTEM},{"role":"user","content":r["question"]}],
            tokenize=False,add_generation_prompt=True) for r in block]
        inputs=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=512)
        raw=master.generate(**inputs,max_new_tokens=args.max_new_tokens,
                            do_sample=False,pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
        offset=inputs["input_ids"].shape[1]
        for row,token_ids in zip(block,raw):
            answer=clean_answer(tok.decode(token_ids[offset:],skip_special_tokens=True))
            if answer and valid_student_example(codec,context,row["question"],answer):
                output.append({**row,"answer":answer})
        print(f"Qwen profesor: {min(len(all_questions),len(output)+0)} ejemplos aceptados hasta ahora",flush=True)
    concepts={row["concept"] for row in training}
    train_rows=[r for r in output if r["concept"] in concepts]
    test_rows=[r for r in output if r["concept"] not in concepts]
    if len(train_rows)<max(120,int(len(training)*0.4)) or len(test_rows)<6:
        raise ValueError(f"Respuestas compatibles insuficientes: {len(train_rows)} entreno, {len(test_rows)} evaluación")
    result=manifest(args.model,revision,train_rows,test_rows)
    result.update({"schema":"mjonia.teacher.v1","generation_version":9,
                   "total_candidate_training":len(training),"total_candidate_holdout":len(holdout),
                   "accepted_training":len(train_rows),"accepted_holdout":len(test_rows),
                   "source":"Preguntas originales + respuestas Qwen, licencia de pesos Apache 2.0; no se copian pesos",
                   "student_version":8})
    save_manifest(args.out,result)
    print(f"Enseñanza v0.9: {len(train_rows)} respuestas de entrenamiento, "
          f"{len(test_rows)} conceptos reservados, Qwen SHA {revision}",flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--student-base",default="checkpoints/web/v08.pt")
    p.add_argument("--out",default="data/teacher_v09.json")
    p.add_argument("--seed",type=int,default=909)
    p.add_argument("--variants-per-concept",type=int,default=8)
    p.add_argument("--holdout-variants",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--max-new-tokens",type=int,default=40)
    p.add_argument("--cpu-threads",type=int,default=2)
    run(p.parse_args())
