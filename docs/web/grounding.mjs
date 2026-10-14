// MJON IA · Núcleo de respuestas locales verificables.
// NO es otro modelo, ni pesos aprendidos. Complementa honestamente el
// Transformer experimental de MJON y nunca consulta servidores de inferencia.
export function normalize(text) {
  return String(text).normalize('NFD').replace(/[\u0300-\u036f]/g,'')
    .toLowerCase().replace(/[^a-z0-9+\-*/%()., ]+/g,' ')
    .replace(/\s+/g,' ').trim();
}
function cleanNumber(value) {
  if (!Number.isFinite(value)) throw Error('El resultado no es finito.');
  if (Math.abs(value)>1e15) throw Error('El número supera el rango permitido.');
  return Number(value.toPrecision(12)).toLocaleString('es-AR',{maximumFractionDigits:10});
}
function answer(text, kind, sources=[], extra={}) {
  return {text,kind,sources, ...extra};
}
const SUBJECTS={
  biologia:'Biología',quimica:'Química',fisica:'Física',matematica:'Matemática',
  lengua:'Lengua',historia:'Historia',geografia:'Geografía',informatica:'Informática',
  pensamiento:'Pensamiento crítico',limites:'Límites del conocimiento'
};
const ALIASES={
  'adn':['acido desoxirribonucleico'],
  'fotosíntesis':['fotosintesis'],
  'respiración celular':['respiracion celular'],
  'evolución biológica':['evolucion'],
  'célula':['celula'],
  'átomo':['atomo'],
  'molécula':['molecula'],
  'función matemática':['funcion en matematica'],
  'función informática':['funcion de programacion','funcion de codigo'],
  'memoria RAM':['ram'],
  'código abierto':['software libre de codigo abierto','open source'],
  'reacción química':['reaccion quimica'],
  'corriente eléctrica':['corriente electrica'],
};
const EXAMPLES={
  'átomo':'Por ejemplo, un átomo de oxígeno tiene ocho protones.',
  'molécula':'Por ejemplo, el agua (H₂O) está formada por dos átomos de hidrógeno y uno de oxígeno.',
  'célula':'Por ejemplo, una neurona es una célula especializada en transmitir señales.',
  'fotosíntesis':'Por ejemplo, una planta utiliza energía de la luz para sintetizar compuestos orgánicos.',
  'fracción':'Por ejemplo, 1/2 representa una de dos partes iguales.',
  'número primo':'Por ejemplo, 7 es primo porque solo es divisible por 1 y por 7.',
  'algoritmo':'Por ejemplo: tomar dos números, compararlos y devolver el mayor.',
  'variable informática':'Por ejemplo, una variable llamada temperatura puede guardar el valor 25.',
  'base de datos':'Por ejemplo, una tabla de inventario almacena códigos, cantidades y ubicaciones.',
  'correlación':'Por ejemplo, vender más helados y tener más días calurosos puede ocurrir a la vez, sin que uno cause directamente al otro.',
  'hipótesis':'Por ejemplo: «si aumento la luz disponible, la planta crecerá más» es una hipótesis comprobable.',
  'ecosistema':'Por ejemplo, un bosque integra árboles, hongos, animales, suelo y clima.',
  'reacción química':'Por ejemplo, la formación de óxido en el hierro es una reacción química.',
  'porcentaje':'Por ejemplo, 25 % de 200 equivale a 50.'
};
const NOISE=new Set(['que','es','un','una','el','la','los','las','de','del',
'y','a','en','para','por','como','cual','cuales','significa','significado',
'explica','explicame','brevemente','resumen','definicion','define','defini',
'concepto','sobre','quiero','necesito','saber','puedes','podes','decime','dime',
'me','lo','tu','su','con','algo','se','diferencia','entre','tiene','consiste',
'funciona','sirve','del','al','simple','palabras','faciles','estudiante','secundaria']);

function tokenize(text){
  return normalize(text).split(' ').filter(x=>x&&!NOISE.has(x));
}
function contained(text, phrase) {
  return (' '+text+' ').includes(' '+phrase+' ');
}
function safelyParseMath(text) {
  // Input parser with deliberately tiny grammar. Never uses eval/Function.
  const source=String(text).trim().replace(/\s+/g,'')
    .replace(/×/g,'*').replace(/[÷:]/g,'/');
  if (source.length>110 || !/^[\d.,()+*/%-]+$/.test(source)) return null;
  if (!/[+\-*/%]/.test(source)) return null;
  // Decimal comma is legal only inside a number, no thousands separators.
  const tokens=source.match(/(?:\d+(?:[.,]\d+)?)|[()+*/%-]/g);
  if (!tokens || tokens.join('')!==source || tokens.length>75) return null;
  let pos=0;
  function primary(depth) {
    if (depth>20)throw Error('Expresión demasiado compleja.');
    const cur=tokens[pos];
    if (cur==='+'||cur==='-') {
      pos++;
      const next=primary(depth+1);
      return cur==='-'?-next:next;
    }
    if(cur==='(') {
      pos++;
      const value=expression(depth+1);
      if(tokens[pos]!==')')throw Error('Falta un paréntesis de cierre.');
      pos++;
      return value;
    }
    if (!cur||!/^\d/.test(cur))throw Error('Expresión incompleta.');
    pos++;
    const v=Number(cur.replace(',','.'));
    if(!Number.isFinite(v))throw Error('Número inválido.');
    return v;
  }
  function product(depth) {
    let value=primary(depth+1);
    while(tokens[pos]==='*'||tokens[pos]==='/'||tokens[pos]==='%'){
      const operator=tokens[pos++];
      const right=primary(depth+1);
      if((operator==='/'||operator==='%')&&right===0)
        throw Error('No se puede dividir por cero.');
      value=operator==='*'?value*right:operator==='/'?value/right:value%right;
    }
    return value;
  }
  function expression(depth) {
    let value=product(depth+1);
    while(tokens[pos]==='+'||tokens[pos]==='-'){
      const op=tokens[pos++];
      const right=product(depth+1);
      value=op==='+'?value+right:value-right;
    }
    return value;
  }
  const value=expression(0);
  if(pos!==tokens.length)throw Error('Hay símbolos que no puedo interpretar.');
  return cleanNumber(value);
}
function mathQuestion(input){
  const text=normalize(input);
  if(!/\d/.test(text)) return null;
  const percent=text.match(/(?:cuanto\s+es\s+)?(?:el\s+)?(\d+(?:[.,]\d+)?)\s*%\s*(?:de|sobre)\s*(\d+(?:[.,]\d+)?)/);
  if(percent){
    const lhs=Number(percent[1].replace(',','.'));
    const rhs=Number(percent[2].replace(',','.'));
    return answer(percent[1]+' % de '+percent[2]+' = '+cleanNumber(lhs*rhs/100)+'.','calculo');
  }
  const normalized=String(input).trim().replace(/[¿?¡!]/g,'');
  let expression=normalized.replace(/^(cu[aá]nto es|calcul[aá]|calcula|resolv[eé]|resuelve|resultado de|eval[uú]a)\s*(?:el\s*)?/i,'');
  expression=expression.replace(/=\s*$/,'').replace(/\.$/,'').trim();
  const signal=/^(?:cu[aá]nto es|calcul[aá]|calcula|resolv[eé]|resuelve|resultado de|eval[uú]a)\b/i.test(normalized);
  if(!signal&&!/^[\d\s.,()+\-*/%×÷:]+$/.test(expression))return null;
  if(signal&&!/^[\d\s.,()+\-*/%×÷:]+$/.test(expression))return null;
  try {
    const result=safelyParseMath(expression);
    return result===null?null:answer(expression+' = '+result+'.','calculo');
  } catch(e) {
    return answer('No puedo resolver esa expresión tal como está escrita: '+e.message,'limite');
  }
}
function findConcepts(text,items){
  const words=tokenize(text);
  // Separar la puntuación final de los conceptos: «¿Qué es célula?» debe
  // encontrar «célula», sin tomar el punto como parte de la palabra.
  const n=normalize(text).replace(/[.,()!?]/g,' ').replace(/\s+/g,' ').trim();
  const direct=[];
  for(const item of items){
    if(item.subject==='limites')continue;
    const aliases=[item.concept,...(ALIASES[item.concept]||[])];
    for(const alias of aliases){
      const key=normalize(alias);
      if(key && contained(n,key)){
        direct.push({item,score:key.split(' ').length*20+key.length});break;
      }
    }
  }
  direct.sort((a,b)=>b.score-a.score);
  // Remove nested aliases when one is a sub-concept of another.
  const retained=[];
  for(const match of direct){
    const key=normalize(match.item.concept);
    if(retained.some(r=>normalize(r.item.concept).includes(key)))continue;
    retained.push(match);
  }
  if(retained.length)return retained.map(r=>r.item);
  // Lexical backup: require at least 2 meaningful words with >=80% overlap,
  // preventing answers to unrelated questions based on one generic word.
  if(words.length<2)return [];
  const candidates=[];
  for(const item of items){
    if(item.subject==='limites')continue;
    const key=tokenize(item.concept);
    if(key.length<2)continue;
    const matched=key.filter(x=>words.includes(x)).length;
    const fraction=matched/key.length;
    if(matched>=2&&fraction>=.8)candidates.push({item,score:fraction*100+matched});
  }
  candidates.sort((a,b)=>b.score-a.score);
  return candidates.length?[candidates[0].item]:[];
}
function looksLikeRequest(n){
  return /\b(que|cual|cuales|cuanto|cuanta|cuantos|cuantas|como|cuando|donde|por que|explica|defini|decime|dime|calcula|resuelve|quien|ejemplo|diferencia|sirve|significa)\b/.test(n);
}
export function createMJONGroundedCore(document){
  if(document?.schema!=='mjonia.grounded.v1'||!Array.isArray(document.items)
     ||document.items.length<100)throw Error('Conocimientos MJON inválidos o incompletos.');
  const ids=new Set();
  const items=document.items.map(item=>{
    if(!item||typeof item.id!=='string'||typeof item.concept!=='string'
        ||typeof item.answer!=='string'||!SUBJECTS[item.subject]||ids.has(item.id))
      throw Error('Ficha MJON corrupta o repetida.');
    ids.add(item.id);return {...item};
  });
  const state={last:null,name:null};
  function reset(){state.last=null;state.name=null;}
  function respond(prompt){
    const question=String(prompt||'').trim().slice(0,1200);
    if(!question)return answer('Escribí una pregunta para empezar.','orientacion');
    const n=normalize(question);
    // No registrar ni volver a mostrar contraseñas ni secretos.
    if(/\b(contrasena|password|clave secreta|pin personal)\b/.test(n))
      return answer('No guardo ni conozco tus contraseñas. Evitá compartirlas en un chat.','limite');
    if(/^(hola|buenas|buen dia|buenas tardes|buenas noches|hey|saludos)[.! ]*$/.test(n))
      return answer('¡Hola! Soy MJON IA. Puedo explicar conceptos de diez áreas, resolver operaciones y seguir preguntas cortas usando información local. ¿Qué querés aprender?','conversacion');
    if(/^(gracias|muchas gracias|genial|perfecto)[.! ]*$/.test(n))
      return answer('¡De nada! ¿Seguimos con otra pregunta?','conversacion');
    if(/\b(quien sos|como te llamas|quien eres|sos una ia)\b/.test(n))
      return answer('Soy MJON IA. Mi núcleo neuronal es propio y experimental; estas respuestas verificables se apoyan en fichas locales y cálculo determinista, sin un modelo externo ni conexión a una API.','identidad');
    if(/\b(que podes hacer|que puedes hacer|ayuda|tus capacidades|como funcionas)\b/.test(n))
      return answer('Puedo consultar 120 conceptos locales, explicar diferencias, dar ejemplos verificados, hacer cuentas y recordar tu nombre durante esta sesión. Fuera de esos temas puedo decir que no sé, en lugar de inventar.','orientacion');
    const myName=n.match(/^(?:me llamo|mi nombre es)\s+([a-záéíóúüñ -]{2,40})[.! ]*$/i);
    if(myName){
      const value=question.replace(/^(?:me llamo|mi nombre es)\s+/i,'').replace(/[.! ]+$/,'').trim();
      if(value.length<2)return answer('Decime tu nombre para recordarlo durante esta conversación.','orientacion');
      state.name=value.slice(0,40);
      return answer('Mucho gusto, '+state.name+'. Recordaré ese nombre solamente en esta conversación.','memoria');
    }
    if(/\b(cual es mi nombre|como me llamo|te acordas de mi nombre)\b/.test(n))
      return state.name?answer('Me dijiste que te llamás '+state.name+'.','memoria'):
      answer('Todavía no me dijiste tu nombre. Podés escribir «Me llamo …».','limite');
    const math=mathQuestion(question);
    if(math)return math;
    let matches=findConcepts(question,items);
    const followup=/^(?:y eso|y un ejemplo|dame un ejemplo|otro ejemplo|podes darme un ejemplo|explicalo mejor|explicame mas|por ejemplo|y para que sirve|y cual es su importancia)/.test(n);
    if(!matches.length && followup && state.last){
      matches=[state.last];
    }
    if(matches.length){
      const primary=matches[0];
      state.last=primary;
      const contrast=/\b(diferencia|diferencias|comparar|compara|versus|vs)\b/.test(n);
      if(contrast && matches.length>=2){
        const itemsShown=matches.slice(0,2);
        return answer(itemsShown.map(x=>x.concept+': '+x.answer).join('\n')+
          '\nEstas son sus definiciones verificadas; para una comparación más profunda necesitaría más información.',
          'comparacion',itemsShown.map(x=>x.id));
      }
      if(/\b(ejemplo|ejemplos)\b/.test(n)){
        const e=EXAMPLES[primary.concept];
        return e?answer(e,'ejemplo',[primary.id],{concept:primary.concept}):
          answer('Sobre '+primary.concept+' tengo una definición verificada, pero no un ejemplo concreto revisado: '+primary.answer,
                 'definicion',[primary.id],{concept:primary.concept});
      }
      if(/\b(explicalo mejor|explicame mas|amplia)\b/.test(n))
        return answer('Esto es lo que tengo verificado sobre '+primary.concept+': '+primary.answer+
        ' No dispongo de una explicación más detallada validada para este concepto.', 'definicion',[primary.id],{concept:primary.concept});
      return answer(primary.concept+': '+primary.answer,'definicion',[primary.id],
                    {subject:SUBJECTS[primary.subject],concept:primary.concept});
    }
    // Out-of-domain: do not let a tiny network hallucinate as a "verified" answer.
    if(looksLikeRequest(n)){
      return answer('No tengo información local suficiente para responder con seguridad. Puedo ayudarte con conceptos de ciencias, matemática, informática, historia, lengua, geografía y pensamiento crítico, o resolver una operación.','limite');
    }
    return answer('Puedo conversar sobre contenidos verificables. Probá «¿Qué es un átomo?» o «¿Cuánto es (18 + 6) / 3?».','orientacion');
  }
  return {respond,reset,get size(){return items.length;},get lastConcept(){return state.last?.concept||null;}};
}
