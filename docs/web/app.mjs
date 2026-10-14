import {decodeWeights, MJONTransformer} from './runtime.mjs';

const $ = id => document.getElementById(id);
const form=$('chat-form'), input=$('prompt'), messages=$('messages');
const status=$('status'),send=$('send'),stop=$('stop'),reset=$('reset'),selector=$('model-select');
let model=null, models=[], busy=false, abort=false, serial=0;

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
    const result=await model.generate(`Usuario: ${question}\nAsistente:`,{
      count:Number($('count').value),temperature:Number($('temperature').value),
      shouldStop:()=>abort,
      onToken:s=>{answer.textContent=s||'…';messages.scrollTop=messages.scrollHeight;}
    });
    answer.textContent=result||'(Sin texto)';
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
    await loadModel(models[0].id);
  }catch(error){setStatus('Error al iniciar','error');bubble('system','No se pudo inicializar MJON IA: '+error.message+'. Revisá GitHub Actions.');}
}
init();
