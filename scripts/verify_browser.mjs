// Comprobación de inferencia real en Node.js de cada modelo publicado.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {decodeWeights,MJONTransformer} from '../docs/web/runtime.mjs';

const directory=new URL('../docs/web/',import.meta.url);
const catalog=JSON.parse(fs.readFileSync(new URL('models.json',directory),'utf8'));
assert.equal(catalog.format,'mjonia.models.v1');
for(const item of catalog.models){
  const encoded=fs.readFileSync(new URL(item.file,directory),'utf8');
  const sha=crypto.createHash('sha256').update(encoded).digest('hex');
  assert.equal(sha,item.sha256,'Firma SHA-256 inválida');
  const state=await decodeWeights(encoded);
  const model=new MJONTransformer(state);
  assert.equal(model.cfg.context,item.context);
  assert.equal(model.steps,item.steps);
  const phrase='Córdoba, español y 🧪';
  assert.equal(model.codec.decode(model.codec.encode(phrase)),phrase,'Tokenizer debe ser reversible');
  if(['v03','v04','v05','v06','v07','v08','v09','v10','v11'].includes(item.id))assert.ok(model.cfg.vocab_size>256 && model.codec.merges.length>0);
  const logits=model.logits(model.codec.encode('Hola'));
  assert.equal(logits.length,model.cfg.vocab_size);
  assert.ok([...logits].every(Number.isFinite));
  assert.equal(typeof await model.generate('Hola',{count:3}),'string');
  console.log('Inferencia real OK:',item.id);
}
