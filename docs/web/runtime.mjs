// MJON IA: inferencia cuantizada int8, tokenización BPE/bytes y caché KV.
export async function decodeWeights(encoded){
  if(typeof DecompressionStream!=='function')throw Error('DecompressionStream no soportado');
  const compressed=Uint8Array.from(atob(encoded.trim()),c=>c.charCodeAt(0));
  const stream=new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'));
  const buffer=new Uint8Array(await new Response(stream).arrayBuffer());
  const size=(buffer[0]<<8)|buffer[1];
  const meta=JSON.parse(new TextDecoder().decode(buffer.slice(2,2+size)));
  if(meta.format!=='mjonia.int8.binary.v1')throw Error('Formato de pesos desconocido');
  const weights={};let offset=2+size;
  for(const [name,shape,scale,count] of meta.tensors){
    if(!Number.isInteger(count)||count<0||offset+count>buffer.length)throw Error('Checkpoint incompleto');
    weights[name]=[shape,scale,new Int8Array(buffer.slice(offset,offset+count).buffer)];
    offset+=count;
  }
  if(offset!==buffer.length)throw Error('Checkpoint corrupto');
  return {...meta,weights};
}
const encoder=new TextEncoder(),decoder=new TextDecoder();
function pairMerge(ids,a,b,newId){
  const out=[];
  for(let i=0;i<ids.length;i++){
    if(i+1<ids.length&&ids[i]===a&&ids[i+1]===b){out.push(newId);i++}
    else out.push(ids[i]);
  }
  return out;
}
export class TokenCodec{
  constructor(config,vocabSize){
    this.merges=config?.type==='byte_bpe_v1'?config.merges:[];
    if(config&&config.type!=='byte_bpe_v1')throw Error('Tokenizer incompatible');
    if(256+this.merges.length!==vocabSize)throw Error('Tamaño del vocabulario inconsistente');
    this.lookup=Array.from({length:256},(_,i)=>Uint8Array.of(i));
    this.merges.forEach(([a,b],i)=>{
      const newId=i+256;
      if(!Number.isInteger(a)||!Number.isInteger(b)||a<0||b<0||a>=newId||b>=newId)throw Error('Merge BPE inválida');
      const l=this.lookup[a],r=this.lookup[b],bytes=new Uint8Array(l.length+r.length);
      bytes.set(l);bytes.set(r,l.length);this.lookup.push(bytes);
    });
  }
  encode(text){
    let ids=Array.from(encoder.encode(text));
    this.merges.forEach(([a,b],i)=>{ids=pairMerge(ids,a,b,256+i)});
    return ids;
  }
  decodeBytes(ids){
    const length=ids.reduce((sum,id)=>sum+this.lookup[id].length,0),out=new Uint8Array(length);
    let index=0;for(const id of ids){out.set(this.lookup[id],index);index+=this.lookup[id].length}return out;
  }
  decode(ids){return decoder.decode(this.decodeBytes(ids))}
}
function add(a,b){const z=new Float32Array(a.length);for(let i=0;i<z.length;i++)z[i]=a[i]+b[i];return z}
function linear(x,w,b=null){
  const [shape,scale,q]=w,[rows,cols]=shape;
  if(x.length!==cols)throw Error('Tensor incompatible');
  const out=new Float32Array(rows);
  for(let i=0;i<rows;i++){
    let sum=b?b[2][i]*b[1]:0;
    for(let j=0;j<cols;j++)sum+=q[i*cols+j]*scale*x[j];
    out[i]=sum;
  }
  return out;
}
function norm(x,w,b){
  const n=x.length;let mean=0,variance=0;
  for(const z of x)mean+=z;mean/=n;
  for(const z of x)variance+=(z-mean)*(z-mean);variance/=n;
  const inv=1/Math.sqrt(variance+1e-5),out=new Float32Array(n);
  for(let i=0;i<n;i++)out[i]=(x[i]-mean)*inv*w[2][i]*w[1]+b[2][i]*b[1];return out;
}
const gelu=x=>.5*x*(1+Math.tanh(.7978845608028654*(x+.044715*x*x*x)));
function softmax(scores){
  const mx=Math.max(...scores),vals=scores.map(x=>Math.exp(x-mx)),sum=vals.reduce((a,b)=>a+b,0);
  return vals.map(x=>x/sum);
}
export class MJONTransformer{
  constructor(state){
    this.cfg=state.config;this.weights=state.weights;this.steps=state.steps;
    const c=this.cfg;
    if(c.d_model%c.heads!==0||c.context<2)throw Error('Configuración incompatible');
    this.codec=new TokenCodec(state.tokenizer||null,c.vocab_size);
  }
  w(name){const x=this.weights[name];if(!x)throw Error('Falta tensor '+name);return x}
  freshCache(){return Array.from({length:this.cfg.layers},()=>({k:[],v:[]}))}
  step(tokenId,pos,cache){
    const c=this.cfg,d=c.d_model,heads=c.heads,hd=d/heads;
    const token=this.w('token.weight'),position=this.w('position.weight');
    let x=new Float32Array(d);
    for(let j=0;j<d;j++)x[j]=token[2][tokenId*d+j]*token[1]+position[2][pos*d+j]*position[1];
    for(let layer=0;layer<c.layers;layer++){
      const p=`blocks.${layer}.`;
      const z=norm(x,this.w(p+'norm1.weight'),this.w(p+'norm1.bias'));
      const qkv=linear(z,this.w(p+'attn.qkv.weight'),this.w(p+'attn.qkv.bias'));
      cache[layer].k.push(qkv.slice(d,2*d));cache[layer].v.push(qkv.slice(2*d,3*d));
      const merged=new Float32Array(d);
      for(let h=0;h<heads;h++){
        const offset=h*hd,scores=[];
        for(const key of cache[layer].k){
          let dot=0;for(let j=0;j<hd;j++)dot+=qkv[offset+j]*key[offset+j];
          scores.push(dot/Math.sqrt(hd));
        }
        const prob=softmax(scores);
        for(let j=0;j<hd;j++){
          let sum=0;for(let t=0;t<prob.length;t++)sum+=prob[t]*cache[layer].v[t][offset+j];
          merged[offset+j]=sum;
        }
      }
      x=add(x,linear(merged,this.w(p+'attn.proj.weight'),this.w(p+'attn.proj.bias')));
      const z2=norm(x,this.w(p+'norm2.weight'),this.w(p+'norm2.bias'));
      const hidden=linear(z2,this.w(p+'mlp.0.weight'),this.w(p+'mlp.0.bias'));
      for(let j=0;j<hidden.length;j++)hidden[j]=gelu(hidden[j]);
      x=add(x,linear(hidden,this.w(p+'mlp.2.weight'),this.w(p+'mlp.2.bias')));
    }
    return linear(norm(x,this.w('norm.weight'),this.w('norm.bias')),token);
  }
  logits(ids){
    const seq=ids.slice(-this.cfg.context),cache=this.freshCache();
    if(!seq.length)throw Error('Prompt vacío');
    let logits;for(let i=0;i<seq.length;i++)logits=this.step(seq[i],i,cache);
    return logits;
  }
  async generate(prompt,{count=80,temperature=.85,topK=24,onToken=()=>{},shouldStop=()=>false}={}){
    if(temperature<=0)throw Error('Temperatura inválida');
    const history=this.codec.encode(prompt),newIds=[];
    if(!history.length)throw Error('Prompt vacío');
    const c=this.cfg;let cache=this.freshCache(),logits;
    const prefill=()=>{
      cache=this.freshCache();
      const tokens=history.slice(-c.context);
      for(let i=0;i<tokens.length;i++)logits=this.step(tokens[i],i,cache);
    };
    prefill();
    for(let i=0;i<count;i++){
      if(shouldStop())break;
      const ids=Array.from({length:c.vocab_size},(_,k)=>k).sort((a,b)=>logits[b]-logits[a])
        .slice(0,Math.max(1,Math.min(topK,c.vocab_size)));
      const prob=softmax(ids.map(id=>logits[id]/temperature));
      let r=Math.random(),chosen=ids[ids.length-1];
      for(let j=0;j<ids.length;j++){r-=prob[j];if(r<=0){chosen=ids[j];break}}
      history.push(chosen);newIds.push(chosen);onToken(this.codec.decode(newIds),newIds.length);
      if(i+1<count){
        if(cache[0].k.length>=c.context)prefill();
        else logits=this.step(chosen,cache[0].k.length,cache);
      }
      if(i%5===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return this.codec.decode(newIds);
  }
}
