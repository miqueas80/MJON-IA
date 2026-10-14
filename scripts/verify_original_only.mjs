// Regresión de producto: la página principal no importa el profesor externo.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const path='docs';
const html=fs.readFileSync(path+'/index.html','utf8');
const app=fs.readFileSync(path+'/web/app.mjs','utf8');
const readme=fs.readFileSync('README.md','utf8');
for(const file of [
 'docs/open-base.html','docs/web/open-base.mjs','docs/web/open-base-worker.mjs',
 'docs/web/open-base.css','docs/web/inherited.mjs',
 'docs/web/inherited-worker.mjs','docs/OPEN_BASE.md','docs/INHERITANCE.md',
 'scripts/verify_open_base.mjs','scripts/verify_inherited.mjs']){
  assert.equal(fs.existsSync(file),false,'No debe existir módulo externo: '+file);
}
assert.ok(!html.includes('inherited-controls'));
assert.ok(!app.includes('InheritedEngine'));
assert.ok(!app.includes('inherited'));
assert.ok(!app.includes('transformers.js'));
assert.ok(html.includes('MJON IA'));
assert.ok(readme.includes('v0.9'));
assert.ok(app.includes("decodeWeights"),'Mantener inferencia local propia');
console.log('PASS: una sola MJON IA, sin módulo o motor externo de Qwen.');
