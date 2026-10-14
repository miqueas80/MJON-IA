// MJON IA: puente opcional a su cerebro heredado; no toca el núcleo MJON Original.
// Las solicitudes se procesan localmente en un Web Worker, tras aprobación del usuario.
export class InheritedEngine {
  constructor({onProgress=()=>{}}={}){
    this.worker=null;this.ready=false;this.pending=null;this.counter=0;
    this.onProgress=onProgress;
  }
  static async compatible(){
    if(!navigator.gpu)return false;
    try{return Boolean(await navigator.gpu.requestAdapter())}catch{return false}
  }
  async load(precision='q4f16'){
    if(this.pending)throw Error('Ya hay una operación en curso');
    if(!['q4f16','q4'].includes(precision))throw Error('Cuantización inválida');
    if(this.ready)return;
    if(!await InheritedEngine.compatible())throw Error('Se requiere WebGPU compatible');
    const worker=new Worker(new URL('./inherited-worker.mjs',import.meta.url),{type:'module'});
    this.worker=worker;
    return await new Promise((resolve,reject)=>{
      this.pending={resolve,reject,id:null};
      worker.addEventListener('message',({data})=>{
        if(this.worker!==worker)return;
        if(data.type==='progress'){
          this.onProgress(data.progress);
        }else if(data.type==='ready'){
          this.ready=true;
          const active=this.pending;this.pending=null;
          active?.resolve();
        }else if(data.type==='result'){
          const active=this.pending;
          if(active && active.id===data.id){this.pending=null;active.resolve(data.answer);}
        }else if(data.type==='error'){
          const active=this.pending;this.pending=null;
          if(!this.ready){this.worker=null;worker.terminate()}
          active?.reject(new Error(data.message||'Falla del motor heredado'));
        }
      });
      worker.addEventListener('error',e=>{
        const active=this.pending;this.pending=null;this.ready=false;
        this.worker=null;worker.terminate();
        active?.reject(new Error(e.message||'El motor heredado no pudo iniciarse'));
      });
      worker.postMessage({type:'load',precision});
    });
  }
  async generate(question,history=[]){
    if(!this.ready||!this.worker)throw Error('Cargá el cerebro heredado primero');
    if(this.pending)throw Error('Ya se está procesando otra solicitud');
    const messages=[
      {role:'system',content:'Sos MJON IA usando un cerebro preentrenado a partir de Qwen2.5-0.5B-Instruct. Indicá que esta capacidad proviene de Qwen si preguntan. Respondé en español claro. No inventes datos ni simules capacidades que no tenés.'},
      ...history.slice(-4).flatMap(turn=>[
        {role:'user',content:turn.question.slice(0,1100)},
        {role:'assistant',content:turn.answer.slice(0,1100)}
      ]),
      {role:'user',content:question}
    ];
    const id=++this.counter;
    return await new Promise((resolve,reject)=>{
      this.pending={resolve,reject,id};
      this.worker.postMessage({type:'generate',id,messages});
    });
  }
  release(){
    const active=this.pending;this.pending=null;
    this.worker?.terminate();this.worker=null;this.ready=false;
    active?.reject(new Error('Descarga o generación cancelada'));
  }
}
