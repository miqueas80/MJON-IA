// Pruebas estáticas de integración Open-Base; no descargan pesos externos.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname,resolve} from 'node:path';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const load=path=>readFileSync(resolve(root,path),'utf8');
const html=load('docs/open-base.html');
const client=load('docs/web/open-base.mjs');
const worker=load('docs/web/open-base-worker.mjs');
const home=load('docs/index.html');
const docs=load('docs/OPEN_BASE.md');
for (const id of ['load','cancel','precision','status','download','compat','messages',
'chat-form','prompt','send','good','bad','clear','export','scores','storage']) {
  assert.match(html,new RegExp('id="'+id+'"'),'Control faltante: '+id);
}
assert.ok(home.includes('./open-base.html'),'Falta acceso desde el inicio');
assert.ok(html.includes('Qwen2.5-0.5B-Instruct'));
assert.ok(html.includes('Apache 2.0'));
assert.ok(client.includes('new Worker('));
assert.ok(client.includes("ui.load.addEventListener('click'"));
assert.ok(client.includes("engine.postMessage({type:'load'"));
assert.ok(!/\bfetch\(/.test(client),'La interfaz no debería enviar consultas por red');
assert.ok(worker.includes("await import(TRANSFORMERS_URL)"));
assert.ok(worker.includes('await pipeline('));
assert.ok(worker.includes("device: 'webgpu'"));
assert.ok(worker.includes("env.useBrowserCache = true"));
assert.ok(worker.includes('validateMessages(messages)'));
assert.ok(!/api[_-]?key|Bearer\s+[A-Za-z]/i.test(worker),'No exponer credenciales');
assert.ok(docs.includes('no se garantiza'),'Falta advertencia de uso offline');
console.log('Open-Base: estructura, carga optativa, atribución y privacidad básica PASS');
