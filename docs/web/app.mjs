import {decodeWeights, MJONTransformer} from './runtime.mjs';

const $ = id => document.getElementById(id);
const form=$('chat-form'), input=$('prompt'), messages=$('messages');
const status=$('status'),send=$('send'),stop=$('stop'),reset=$('reset'),selector=$('model-select');
let model=null, models=[], benchmark=null, firstwords=null, memoryScores=null, schoolScores=null, distillScores=null, busy=false, abort=false, serial=0;
let session=[];

function setStatus(value,kind=''){status.textContent=value;status.dataset.kind=kind;}
function bubble(role,text){
  const el=document.createElement('article');el.className='bubble '+role;
  const head=document.createElement('small');
  head.textContent=role==='user'?'VOS':role==='system'?'INFORMACIÓN':'MJON IA · LOCAL';
  const content=document.createElement('p');content.textContent=text;
  el.append(head,content);messages.append(el);messages.scrollTop=messages.scrollHeight;
  return content;
}
function setBusy(value){
  busy=value;send.disabled=value||!model;stop.disabled=!value;
  input.disabled=value;selector.disabled=value;
}
function clear(){
  if(busy)return;
  session=[];
  messages.replaceChildren();
  bubble('system','Estos modelos son experimentales y generan tokens, no respuestas fiables. Elegí una versión para comparar. No ingreses información sensible.');
}
function enableModelSelection(items){
  selector.replaceChildren();
  for(const variant of items){
    const option=document.createElement('option');
    option.value=variant.id;option.textContent=variant.label;
    selector.append(option);
  }
}
async function loadModel(id){
  if(busy)return;
  const selected=models.find(entry=>entry.id===id);
  if(!selected)throw new Error('Versión inexistente');
  const request=++serial;
  model=null;selector.disabled=true;send.disabled=true;
  setStatus('Cargando '+selected.label+'…');
  try{
    const response=await fetch('./web/'+selected.file,{cache:'no-cache'});
    if(!response.ok)throw new Error('No se encontraron los pesos: HTTP '+response.status);
    const encoded=await response.text();
    const digest=new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(encoded)));
    const hash=[...digest].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==selected.sha256)throw new Error('Error de integridad del archivo de pesos');
    const state=await decodeWeights(encoded);
    if(request!==serial)return;
    const candidate=new MJONTransformer(state);
    if(candidate.cfg.context!==selected.context || candidate.steps!==selected.steps)
      throw new Error('Los pesos no corresponden con el catálogo publicado');
    model=candidate;
    const number=new Intl.NumberFormat('es-AR').format(selected.parameters);
    $('model-info').textContent=`${number} parámetros · ${selected.context} ${selected.tokenizer==='bpe'?'tokens BPE':'bytes'} de contexto · ${selected.steps} pasos · ${selected.split_mode==='bytes'?'partición por bytes':'partición de diálogos'}`;
    $('model-description').textContent=selected.description;
    session=[];
    const distill=distillScores?.models?.find(x=>x.model===selected.id);
    const school=schoolScores?.models?.find(x=>x.checkpoint===selected.id);
    const schoolBefore=schoolScores?.models?.find(x=>x.checkpoint==='v06');
    const memory=memoryScores?.models?.find(x=>x.checkpoint===selected.id);
    const priorMemory=memoryScores?.models?.find(x=>x.checkpoint==='v05');
    const metric=benchmark?.results?.find(x=>x.model===selected.id);
    const result=$('benchmark-info');
    const first=firstwords?.models?.find(x=>x.checkpoint===selected.id);
    const baseline=firstwords?.models?.find(x=>x.checkpoint==='v04');
    if(result) result.textContent=distill?
      `MJON Destilación: profesor ${distillScores.teacher}; pérdida en conceptos reservados ${distill.teacher_holdout_loss}. Examen escolar: ${distill.school_holdout.correct}/${distill.school_holdout.total}; transferencias: ${distill.school_transfer.correct}/${distill.school_transfer.total}. El profesor y el alumno pueden equivocarse; no son el mismo modelo.`:
      school?
      `MJON Examen: ${school.holdout.correct}/${school.holdout.total} con entidades nuevas; ${school.transfer.correct}/${school.transfer.total} con preguntas reformuladas. Repaso: ${school.previous_memory.correct}/${school.previous_memory.total} de memoria y ${school.previous_short.correct}/${school.previous_short.total} de respuestas breves. Referencia v0.6: ${schoolBefore?.holdout.correct??'?'}/${schoolBefore?.holdout.total??'?'}. Los errores se publican.`:
      memory?`MJON Memory: ${memory.correct}/${memory.total} aciertos; recordar ${memory.by_case.recall.correct}/${memory.by_case.recall.total}, corregir ${memory.by_case.correction.correct}/${memory.by_case.correction.total}. v0.5: ${priorMemory?.correct??'?'}/${priorMemory?.total??'?'}. Retención de tarea anterior: ${memoryScores.retention.after.correct}/${memoryScores.retention.after.total}.`:first?`MJON First Words: ${first.exact}/${first.total} extracciones correctas; v0.4 sin ajuste: ${baseline?.exact??'?'}/${baseline?.total??'?'}. No implica razonamiento general.`:
      metric?`MJON Bench: ${metric.greedy_exact_count}/${metric.evaluated_numeric_probes} respuestas numéricas correctas, ${metric.pairwise_count}/${metric.evaluated_numeric_probes} preferencias acertadas.`:
      'Este modelo no tiene benchmark publicado.';
    const examples=$('examples');examples.replaceChildren();
    const prompts=selected.id==='v08'?
      ['¿Qué es una célula?','¿Para qué sirve una variable en programación?','¿Qué es una fracción?','¿Qué significa código abierto?']:
      selected.id==='v07'?
      ['En una caja hay 12 fichas. Entran 5 fichas y salen 3. ¿Cuántas hay?',
       'Texto: el objeto periscopio tiene color azul. El objeto catalejo es rojo. ¿De qué color es periscopio?',
       'Ficha experimental: el instrumento sextante mide longitud. El anemómetro mide velocidad. ¿Qué mide sextante?',
       'Si zorro llegó antes que gato y gato llegó antes que perro, ¿quién llegó primero?']:
      selected.id==='v06'?
      ['El frasco tiene color rojo.','Corrección: color azul.','¿Cuál es el color actual del frasco?','¿Cuál es el color del frasco?']:
      selected.id==='v05'?
      ['Ficha del prisma: color azul. ¿Qué color indica la ficha?','El reactor es de vidrio. Decime el material.','Registro del depósito. Estado = líquido. Respondé el estado.']:
      ['Hola.','¿Qué es una inteligencia artificial?','¿Qué es aprender?'];
    for(const q of prompts){
      const btn=document.createElement('button');btn.type='button';btn.dataset.prompt=q;
      btn.textContent=q.length>42?q.slice(0,39)+'…':q;examples.append(btn);
    }
    $('last-run').textContent='—';
    messages.replaceChildren();clear();
    setStatus('Modelo cargado · listo','ready');
  }catch(error){
    if(request!==serial)return;
    setStatus('No se pudo cargar','error');
    bubble('system','Error al cargar '+selected.label+': '+error.message);
  }finally{
    if(request===serial){send.disabled=!model;selector.disabled=false;}
  }
}
reset.addEventListener('click',clear);
stop.addEventListener('click',()=>{abort=true;stop.disabled=true});
selector.addEventListener('change',()=>{if(!busy)loadModel(selector.value)});
$('examples').addEventListener('click',e=>{
  const button=e.target.closest('button[data-prompt]');
  if(button&&!busy){input.value=button.dataset.prompt;input.focus();}
});
$('temperature').addEventListener('input',e=>{$('temp-value').textContent=Number(e.target.value).toFixed(2)});
form.addEventListener('submit',async e=>{
  e.preventDefault();if(!model||busy)return;
  const question=input.value.trim();if(!question)return;
  bubble('user',question);input.value='';
  const answer=bubble('assistant','Generando…');abort=false;setBusy(true);
  setStatus('Generando en tu dispositivo','working');const start=performance.now();
  try{
    const history=['v06','v07','v08'].includes(selector.value)
     ?session.map(turn=>`Usuario: ${turn.question}\nAsistente: ${turn.answer}\n`).join('')
     :'';
   const result=await model.generate(`${history}Usuario: ${question}\nAsistente:`,{
      count:['v05','v06','v07','v08'].includes(selector.value)?24:Number($('count').value),
      temperature:['v05','v06','v07','v08'].includes(selector.value)?1:Number($('temperature').value),
      topK:['v05','v06','v07','v08'].includes(selector.value)?1:24,
      stopAtNewline:['v05','v06','v07','v08'].includes(selector.value),
      shouldStop:()=>abort,
      onToken:s=>{answer.textContent=s||'…';messages.scrollTop=messages.scrollHeight;}
    });
    answer.textContent=result||'(Sin texto)';
    if(['v06','v07','v08'].includes(selector.value))session=[...session,{question,answer:result}].slice(-3);
    $('last-run').textContent=`${((performance.now()-start)/1000).toFixed(1)} s · ${new TextEncoder().encode(result).length} bytes`;
    setStatus(abort?'Detenido':'Modelo listo','ready');
  }catch(error){answer.textContent='Error: '+error.message;setStatus('Error','error');}
  finally{setBusy(false);input.focus();}
});
async function init(){
  clear();setStatus('Buscando modelos…');
  try{
    const response=await fetch('./web/models.json',{cache:'no-cache'});
    if(!response.ok)throw new Error('Catálogo no disponible: HTTP '+response.status);
    const catalog=await response.json();
    if(catalog.format!=='mjonia.models.v1'||!Array.isArray(catalog.models)||!catalog.models.length)
      throw new Error('Catálogo de modelos inválido');
    models=catalog.models;
    enableModelSelection(models);
    try { const scores=await fetch('./web/benchmarks.json',{cache:'no-cache'}); if(scores.ok) benchmark=await scores.json(); } catch (_) { /* evaluación opcional */ }
    try { const scores=await fetch('./web/firstwords.json',{cache:'no-cache'}); if(scores.ok) firstwords=await scores.json(); } catch (_) { /* evaluación opcional */ }
    try { const scores=await fetch('./web/memory.json',{cache:'no-cache'}); if(scores.ok) memoryScores=await scores.json(); } catch (_) { /* evaluación opcional */ }
    try { const scores=await fetch('./web/school.json',{cache:'no-cache'}); if(scores.ok) schoolScores=await scores.json(); } catch (_) { /* evaluación opcional */ }
    try { const scores=await fetch('./web/distillation.json',{cache:'no-cache'}); if(scores.ok) distillScores=await scores.json(); } catch (_) { /* evaluación opcional */ }
    const selected=models[models.length-1].id;
    selector.value=selected;
    await loadModel(selected);
  }catch(error){setStatus('Error al iniciar','error');bubble('system','No se pudo inicializar MJON IA: '+error.message+'. Revisá GitHub Actions.');}
}
init();
