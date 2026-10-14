// Interfaz MJON Open-Base: nunca descarga pesos sin una acción del usuario.
const $ = id => document.getElementById(id);
const ui = {
  status: $('status'), compat: $('compat'), download: $('download'), storage: $('storage'),
  load: $('load'), cancel: $('cancel'), precision: $('precision'),
  messages: $('messages'), form: $('chat-form'), prompt: $('prompt'), send: $('send'),
  examples: $('examples'), good: $('good'), bad: $('bad'), scores: $('scores'),
  clear: $('clear'), export: $('export'),
};
const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct';
const SYSTEM = 'Respondé en español claro, con rigor y ejemplos cuando sirvan. Si no sabés algo, indicalo. Para problemas matemáticos, explicá el procedimiento sin inventar datos. No afirmes ser MJON Original.';
const history = [];
const observations = [];
let engine = null, ready = false, working = false, lastIndex = -1, pendingId = 0;
let loading = false, selectedPrecision = 'q4f16';

function status(message, kind = '') {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
}
function notice(text) {
  const message = document.createElement('article');
  message.className = 'bubble system';
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  message.append(paragraph);
  ui.messages.append(message);
  ui.messages.scrollTop = ui.messages.scrollHeight;
}
function bubble(role, text) {
  const message = document.createElement('article');
  message.className = 'bubble ' + role;
  const label = document.createElement('small');
  label.textContent = role === 'user' ? 'VOS' : 'MJON Open-Base · Qwen (modelo derivado)';
  const value = document.createElement('p');
  value.textContent = text;
  message.append(label, value);
  ui.messages.append(message);
  ui.messages.scrollTop = ui.messages.scrollHeight;
  return value;
}
function refresh() {
  ui.load.disabled = ready || loading || working;
  ui.cancel.disabled = !engine;
  ui.precision.disabled = loading || ready || working;
  ui.prompt.disabled = !ready || working;
  ui.send.disabled = !ready || working;
  ui.good.disabled = !ready || working || lastIndex < 0;
  ui.bad.disabled = !ready || working || lastIndex < 0;
  ui.export.disabled = observations.length === 0;
  const useful = observations.filter(row => row.rating === 'useful').length;
  const incorrect = observations.filter(row => row.rating === 'incorrect').length;
  ui.scores.textContent = `${useful} útiles · ${incorrect} incorrectas · ${observations.length} respuestas en sesión.`;
}
async function checkDevice() {
  let ok = !!navigator.gpu;
  if (ok) {
    try { ok = !!(await navigator.gpu.requestAdapter()); } catch { ok = false; }
  }
  ui.compat.dataset.ok = String(ok);
  ui.compat.textContent = ok
    ? 'WebGPU detectado. La memoria y compatibilidad se comprobarán al cargar.'
    : 'Sin WebGPU compatible. Probá Chrome actualizado u otro dispositivo; no se descargará el modelo.';
  ui.load.disabled = !ok;
  ui.load.dataset.supported = String(ok);
  try {
    if (navigator.storage?.estimate) {
      const info = await navigator.storage.estimate();
      const free = typeof info.quota === 'number' && typeof info.usage === 'number'
        ? Math.max(0, (info.quota - info.usage) / (1024 ** 3)).toFixed(2)
        : null;
      ui.storage.textContent = free
        ? `Cuota disponible estimada: ${free} GiB (no garantiza almacenamiento permanente).`
        : 'No se pudo calcular la cuota libre.';
    }
  } catch {
    ui.storage.textContent = 'El navegador no informó la cuota disponible.';
  }
  refresh();
  if (!ok) ui.load.disabled = true;
}
function stopEngine({show = true} = {}) {
  if (engine) engine.terminate();
  engine = null;
  ready = false;
  loading = false;
  working = false;
  pendingId++;
  history.length = 0;
  lastIndex = -1;
  refresh();
  if (show) {
    status('Modelo liberado');
    ui.download.textContent = 'Modelo liberado de memoria. Los archivos pueden seguir en la caché del navegador.';
    notice('Se liberó el modelo. Para cambiar cuantización o reiniciar, usá «Descargar y ejecutar».');
  }
}
function formatProgress(progress) {
  const raw = String(progress?.status || 'cargando');
  if (raw === 'loading-library') return 'Preparando el motor local…';
  if (raw === 'loading-model') return 'Solicitando pesos del modelo abierto…';
  const percent = Number.isFinite(progress?.progress) ? ` · ${Math.floor(progress.progress)}%` : '';
  const filename = progress?.file ? String(progress.file).split('/').pop().slice(0,90) : '';
  return `${raw}${filename ? ': ' + filename : ''}${percent}`;
}
function handleWorkerMessage(event) {
  const message = event.data || {};
  if (message.type === 'progress') {
    ui.download.textContent = formatProgress(message.progress);
    return;
  }
  if (message.type === 'ready') {
    ready = true;
    loading = false;
    status('Modelo cargado · listo', 'ready');
    ui.download.textContent = 'Modelo disponible en esta sesión. La inferencia usa WebGPU local.';
    notice('Profesor abierto cargado. Ya podés hacer preguntas de distintas materias.');
    refresh();
    ui.prompt.focus();
    return;
  }
  if (message.type === 'result') {
    if (message.id !== pendingId) return;
    const answer = String(message.answer);
    bubble('assistant', answer);
    const question = history[history.length - 1]?.content || '';
    observations.push({
      question,
      answer,
      rating: null,
      model: MODEL_ID,
      precision: selectedPrecision,
    });
    history.push({role: 'assistant', content: answer});
    lastIndex = observations.length - 1;
    working = false;
    status('Respuesta generada en tu dispositivo', 'ready');
    refresh();
    return;
  }
  if (message.type === 'error') {
    if (message.id != null && message.id !== pendingId) return;
    loading = false;
    working = false;
    status('Error en Open-Base', 'error');
    ui.download.textContent = String(message.message || 'Error desconocido');
    notice('No se pudo ejecutar: ' + String(message.message || 'Error desconocido'));
    if (!ready) stopEngine({show: false});
    refresh();
  }
}
ui.load.addEventListener('click', async () => {
  if (engine || loading || ready || ui.load.dataset.supported !== 'true') return;
  selectedPrecision = ui.precision.value;
  if (!['q4f16','q4'].includes(selectedPrecision)) return;
  loading = true;
  status('Descargando · primera ejecución', 'working');
  ui.download.textContent = 'Descargando de Hugging Face. Puede consumir entre 0,5 y 0,8 GB, más archivos auxiliares.';
  refresh();
  // La persistencia solo se solicita tras la interacción explícita; puede ser denegada.
  try { if (navigator.storage?.persist) await navigator.storage.persist(); } catch {}
  if (!loading) return;
  try {
    engine = new Worker(new URL('./open-base-worker.mjs', import.meta.url), {type: 'module'});
    engine.addEventListener('message', handleWorkerMessage);
    engine.addEventListener('error', e => {
      status('Error del motor', 'error');
      ui.download.textContent = 'No se pudo iniciar el motor WebGPU. ' + (e.message || 'Revisá red y navegador.');
      stopEngine({show: false});
    });
    engine.postMessage({type:'load', precision:selectedPrecision});
    refresh();
  } catch (error) {
    stopEngine({show:false});
    ui.download.textContent = 'Error de inicialización: ' + error.message;
    status('No disponible', 'error');
  }
});
ui.cancel.addEventListener('click', () => stopEngine());
ui.clear.addEventListener('click', () => {
  if (working) return;
  history.length = 0;
  observations.length = 0;
  lastIndex = -1;
  ui.messages.replaceChildren();
  notice('Conversación nueva. No se guardan chats en ningún servidor.');
  refresh();
});
ui.examples.addEventListener('click', ({target}) => {
  const button = target.closest('button[data-prompt]');
  if (button && ready && !working) {
    ui.prompt.value = button.dataset.prompt;
    ui.prompt.focus();
  }
});
ui.form.addEventListener('submit', event => {
  event.preventDefault();
  if (!ready || working || !engine) return;
  const question = ui.prompt.value.trim();
  if (!question) return;
  ui.prompt.value = '';
  bubble('user',question);
  history.push({role: 'user',content:question});
  working = true;
  lastIndex = -1;
  status('Pensando con Qwen en tu equipo…', 'working');
  refresh();
  const messages = [
    {role:'system',content:SYSTEM},
    ...history.slice(-10)
  ];
  engine.postMessage({type: 'generate', id: ++pendingId, messages});
});
function rate(value) {
  if (lastIndex < 0 || !observations[lastIndex]) return;
  observations[lastIndex].rating = value;
  notice('Evaluación registrada solo en esta sesión: ' + (value === 'useful' ? 'útil' : 'incorrecta'));
  lastIndex = -1;
  refresh();
}
ui.good.addEventListener('click', () => rate('useful'));
ui.bad.addEventListener('click', () => rate('incorrect'));
ui.export.addEventListener('click', () => {
  if (!observations.length) return;
  const results = {
    schema:'mjonia.open-base.review.v1',
    model:MODEL_ID,
    attribution:'Qwen/Qwen2.5-0.5B-Instruct · Apache-2.0',
    warning:'Evaluaciones humanas manuales, no un benchmark automáticamente verificado.',
    observations
  };
  const file = new Blob([JSON.stringify(results,null,2)],{type:'application/json'});
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'mjon-open-base-evaluacion.json';
  link.click();
  URL.revokeObjectURL(url);
});
notice('Esta es una variante derivada de Qwen con ejecución local opcional. No usa los pesos de MJON Original. La descarga comienza solo al pulsar «Descargar y ejecutar».');
refresh();
checkDevice();
