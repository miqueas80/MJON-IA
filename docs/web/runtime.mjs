// MJON IA v0.1: inferencia del Transformer PyTorch con pesos propios, cuantizados int8.
// Ejecuta en navegador (CPU), sin llamadas a APIs de IA ni bibliotecas externas.
export async function decodeWeights(encoded) {
  if (!('DecompressionStream' in globalThis)) throw new Error('Este navegador necesita DecompressionStream (Chrome/Firefox recientes).');
  const compressed = Uint8Array.from(atob(encoded.trim()), c=>c.charCodeAt(0));
  const decompressed = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'));
  const buffer = new Uint8Array(await new Response(decompressed).arrayBuffer());
  const length = (buffer[0]<<8)|buffer[1];
  const manifest = JSON.parse(new TextDecoder().decode(buffer.slice(2,2+length)));
  if (manifest.format!=='mjonia.int8.binary.v1') throw new Error('Formato de pesos desconocido');
  let cursor=2+length; const weights={};
  for(const [name,shape,scale,count] of manifest.tensors) {
    const ints = new Int8Array(buffer.slice(cursor,cursor+count).buffer);
    weights[name]=[shape,scale,ints];cursor+=count;
  }
  if(cursor!==buffer.byteLength) throw new Error('Checkpoint dañado');
  return {...manifest, weights};
}
const add=(a,b)=>{const r=new Float32Array(a.length);for(let i=0;i<a.length;i++)r[i]=a[i]+b[i];return r};
function linear(x,w,b=null) {
  const [shape,scale,values]=w, [out,inn]=shape;
  if(x.length!==inn)throw new Error('Forma inválida de pesos');
  const result=new Float32Array(out);
  for(let o=0;o<out;o++) {
    let z=b?b[2][o]*b[1]:0;
    for(let j=0;j<inn;j++)z+=values[o*inn+j]*scale*x[j];
    result[o]=z;
  }
  return result;
}
function norm(x,w,b) {
  const n=x.length;
  let mean=0,v=0;for(const a of x)mean+=a;mean/=n;
  for(const a of x)v+=(a-mean)**2;v/=n;
  const inv=1/Math.sqrt(v+1e-5),result=new Float32Array(n);
  for(let i=0;i<n;i++)result[i]=(x[i]-mean)*inv*w[2][i]*w[1]+b[2][i]*b[1];
  return result;
}
const gelu=x=>0.5*x*(1+Math.tanh(0.7978845608028654*(x+0.044715*x*x*x)));
function softmax(x) {
  const m=Math.max(...x),v=x.map(a=>Math.exp(a-m)),s=v.reduce((a,b)=>a+b,0);
  return v.map(a=>a/s);
}
export class MJONTransformer {
  constructor(exported) {
    this.cfg=exported.config;this.weights=exported.weights;this.steps=exported.steps;
    if(this.cfg.vocab_size!==256)throw new Error('Se necesitan 256 tokens-byte');
  }
  w(name) {const w=this.weights[name];if(!w)throw new Error('Falta '+name);return w}
  logits(tokens) {
    const c=this.cfg,tokensIn=tokens.slice(-c.context),n=tokensIn.length,d=c.d_model,h=c.heads,dk=d/h;
    if(!n)throw new Error('Texto vacío');
    const tok=this.w('token.weight'),pos=this.w('position.weight');
    let x=tokensIn.map((id,t)=>{
      const r=new Float32Array(d);
      for(let j=0;j<d;j++)r[j]=tok[2][id*d+j]*tok[1]+pos[2][t*d+j]*pos[1];
      return r;
    });
    for(let layer=0;layer<c.layers;layer++) {
      const p=`blocks.${layer}.`;
      const v=x.map(row=>norm(row,this.w(p+'norm1.weight'),this.w(p+'norm1.bias')));
      const qkv=v.map(row=>linear(row,this.w(p+'attn.qkv.weight'),this.w(p+'attn.qkv.bias')));
      const attention=[];
      for(let i=0;i<n;i++) {
        const merged=new Float32Array(d);
        for(let head=0;head<h;head++) {
          const offset=head*dk,scores=[];
          for(let j=0;j<=i;j++){
            let score=0;
            for(let k=0;k<dk;k++)score+=qkv[i][offset+k]*qkv[j][d+offset+k];
            scores.push(score/Math.sqrt(dk));
          }
          const prob=softmax(scores);
          for(let k=0;k<dk;k++) {
            let sum=0;for(let j=0;j<=i;j++)sum+=prob[j]*qkv[j][2*d+offset+k];
            merged[offset+k]=sum;
          }
        }
        attention.push(linear(merged,this.w(p+'attn.proj.weight'),this.w(p+'attn.proj.bias')));
      }
      x=x.map((row,i)=>add(row,attention[i]));
      x=x.map(row=>{
        const a=norm(row,this.w(p+'norm2.weight'),this.w(p+'norm2.bias'));
        const hidden=linear(a,this.w(p+'mlp.0.weight'),this.w(p+'mlp.0.bias'));
        for(let k=0;k<hidden.length;k++)hidden[k]=gelu(hidden[k]);
        return add(row,linear(hidden,this.w(p+'mlp.2.weight'),this.w(p+'mlp.2.bias')));
      });
    }
    const last=norm(x[n-1],this.w('norm.weight'),this.w('norm.bias'));
    return linear(last,tok); // output weight tied to input token embedding
  }
  async generate(prompt,{count=80,temperature=0.85,topK=24,onToken=()=>{},shouldStop=()=>false}={}) {
    if(temperature<=0)throw new Error('Temperatura inválida');
    const input=[...new TextEncoder().encode(prompt)],out=[];
    for(let i=0;i<count;i++){
      if(shouldStop())break;
      const logits=this.logits(input);
      const ids=[...Array(256).keys()].sort((a,b)=>logits[b]-logits[a]).slice(0,Math.max(1,topK));
      const p=softmax(ids.map(id=>logits[id]/temperature));
      let r=Math.random(),pick=ids[ids.length-1];
      for(let j=0;j<ids.length;j++){r-=p[j];if(r<=0){pick=ids[j];break}}
      input.push(pick);out.push(pick);
      onToken(new TextDecoder().decode(new Uint8Array(out)),out.length);
      if(i%6===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return new TextDecoder().decode(new Uint8Array(out));
  }
}
