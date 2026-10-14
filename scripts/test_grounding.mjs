// MJON IA: pruebas funcionales del ASISTENTE LOCAL en un runtime real Node.js.
// No se consulta una API, no se descargan pesos ajenos, no se usa eval.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createMJONGroundedCore} from '../docs/web/grounding.mjs';

const knowledge=JSON.parse(readFileSync(new URL('../docs/web/knowledge-core.json',import.meta.url),'utf8'));
assert.equal(knowledge.schema,'mjonia.grounded.v1');
assert.equal(knowledge.items.length,120);
assert.equal(new Set(knowledge.items.map(i=>i.id)).size,120);
const core=createMJONGroundedCore(knowledge);
assert.equal(core.size,120);
let asked=0;
for(const item of knowledge.items.filter(x=>x.subject!=='limites')){
  // Rephrasing is handled by lexical matching, not neural generation.
  const response=core.respond('Explicá brevemente '+item.concept+'.');
  assert.equal(response.kind,'definicion',item.concept);
  assert.ok(response.sources.includes(item.id),item.concept);
  assert.ok(response.text.includes(item.answer),item.concept);
  asked++;
}
assert.equal(asked,108);
const calc=[
  ['¿Cuánto es (18 + 6) / 3?','8'],
  ['Calculá 12 + 3 * 4','24'],
  ['Calcula 18 - 23','-5'],
  ['Calcula -8 + 3','-5'],
  ['Calcula 10 / 4','2,5'],
  ['Calcula 2 * (3 + 7)','20'],
  ['¿Cuánto es el 15% de 240?','36'],
];
for(const [question,expected] of calc){
  const result=core.respond(question);
  assert.equal(result.kind,'calculo',question+' -> '+result.text);
  assert.ok(result.text.includes(expected),question);
  asked++;
}
const divzero=core.respond('Calculá 10 / 0');
assert.equal(divzero.kind,'limite');
assert.match(divzero.text,/cero/);
const comparison=core.respond('¿Qué diferencia hay entre átomo y molécula?');
assert.equal(comparison.kind,'comparacion');
assert.equal(comparison.sources.length,2);
assert.match(comparison.text,/átomo:/);
assert.match(comparison.text,/molécula:/);
const definition=core.respond('¿Qué es una célula?');
assert.equal(definition.kind,'definicion');
assert.equal(core.lastConcept,'célula');
const example=core.respond('Y un ejemplo, por favor');
assert.equal(example.kind,'ejemplo');
assert.match(example.text,/neurona/);
assert.equal(core.respond('¿Cómo me llamo?').kind,'limite');
core.respond('Me llamo Alex');
assert.match(core.respond('¿Cómo me llamo?').text,/Alex/);
core.reset();
assert.equal(core.respond('¿Cómo me llamo?').kind,'limite');
const privateInfo=core.respond('¿Cuál es mi contraseña?');
assert.equal(privateInfo.kind,'limite');
assert.ok(!privateInfo.text.includes('123456'));
for(const question of [
 '¿Quién ganó las elecciones de 2026?',
 '¿En qué calle vive una persona desconocida?',
 '¿Cuánto dinero tengo en el banco?',
 '¿Cuál es el resultado de la lotería mañana?'
]){
  const result=core.respond(question);
  assert.equal(result.kind,'limite',question+' -> '+result.text);
}
assert.equal(core.respond('Hola!').kind,'conversacion');
assert.equal(core.respond('Gracias!').kind,'conversacion');
assert.equal(core.respond('¿Quién sos?').kind,'identidad');
assert.equal(core.respond('¿Qué podés hacer?').kind,'orientacion');
const html=readFileSync(new URL('../docs/index.html',import.meta.url),'utf8');
const app=readFileSync(new URL('../docs/web/app.mjs',import.meta.url),'utf8');
const sw=readFileSync(new URL('../docs/sw.js',import.meta.url),'utf8');
for(const id of ['grounded-mode','grounding-status','answer-evidence','model-select','chat-form']){
  assert.ok(html.includes('id="'+id+'"'),id);
}
assert.ok(app.includes('createMJONGroundedCore'));
assert.ok(app.includes('grounded.respond(question)'));
assert.ok(app.includes("if(groundedMode&&grounded)"));
assert.ok(sw.includes('knowledge-core.json'));
assert.ok(!sw.includes('weights-v'));
assert.ok(!app.includes('import {InheritedEngine}'));
console.log('MJON IA FUNCIONAL · '+asked+' definiciones/cálculos comprobados y pruebas de conversación, memoria, evidencia, límites y modo de interfaz PASS');
