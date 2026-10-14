import {decodeWeights,MJONTransformer} from './runtime.mjs';
const $=id=>document.getElementById(id);
const form=$('chat-form'), input=$('prompt'), messages=$('messages'), status=$('status'),send=$('send'),stop=$('stop'),reset=$('reset');
let model=null,busy=false,abort=false;
function setStatus(value,kind=''){status.textContent=value;status.dataset.kind=kind;}
function bubble(role,text){
  const el=document.createElement('article');el.className='bubble '+role;
  const head=document.createElement('small');head.textContent=role==='user'?'VOS':role==='system'?'INFORMACIÓN':'MJON IA · LOCAL';
  const content=document.createElement('p');content.textContent=text;
  el.append(head,content);messages.append(el);messages.scrollTop=messages.scrollHeight;return content;
}
function setBusy(v){busy=v;send.disabled=v||!model;stop.disabled=!v;input.disabled=v;}
function clear(){if(busy)return;messages.replaceChildren();bubble('system','Modelo experimental de apenas 8.176 parámetros. Sus respuestas pueden no tener sentido. Es una prueba real de inferencia local, no un asistente fiable.');}
reset.addEventListener('click',clear);
stop.addEventListener('click',()=>{abort=true;stop.disabled=true});
$('examples').addEventListener('click',e=>{const b=e.target.closest('button[data-prompt]');if(b&&!busy){input.value=b.dataset.prompt;input.focus()}});
$('temperature').addEventListener('input',e=>{$('temp-value').textContent=Number(e.target.value).toFixed(2)});
form.addEventListener('submit',async e=>{
 e.preventDefault();if(!model||busy)return;
 const question=input.value.trim();if(!question)return;
 bubble('user',question);input.value='';
 const answer=bubble('assistant','Generando…');abort=false;setBusy(true);setStatus('Generando en tu dispositivo','working');
 const start=performance.now();
 try{
   const result=await model.generate(`Usuario: ${question}\nAsistente:`,{
     count:Number($('count').value),temperature:Number($('temperature').value),
     shouldStop:()=>abort,
     onToken:s=>{answer.textContent=s||'…';messages.scrollTop=messages.scrollHeight}
   });
   answer.textContent=result||'(Sin texto)';
   $('last-run').textContent=`${((performance.now()-start)/1000).toFixed(1)} s · ${new TextEncoder().encode(result).length} bytes`;
   setStatus(abort?'Detenido':'Modelo listo','ready');
 }catch(error){answer.textContent='Error: '+error.message;setStatus('Error','error')}
 finally{setBusy(false);input.focus()}
});
async function init(){
 clear();setBusy(false);setStatus('Cargando pesos…');
 try{
   const response=await fetch('./web/weights.txt',{cache:'no-cache'});
   if(!response.ok)throw new Error('Archivo de pesos ausente: HTTP '+response.status);
   const state=await decodeWeights(await response.text());model=new MJONTransformer(state);
   $('model-info').textContent=`Transformer causal · ${model.steps} pasos de entrenamiento · 48 bytes de contexto · sin API externa`;
   send.disabled=false;setStatus('Modelo cargado · listo','ready');
 }catch(error){setStatus('Error al cargar','error');bubble('system','No se pudo inicializar el modelo: '+error.message+'. Verificá que el despliegue por GitHub Actions haya terminado.')}
}
init();
