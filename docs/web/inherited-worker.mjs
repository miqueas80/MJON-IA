// MJON IA — cerebro heredado: pesos Qwen preentrenados, explícitamente derivados.
// La biblioteca se importa solo tras la acción explícita "Descargar y ejecutar".
const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct';
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';
const ALLOWED_PRECISIONS = new Set(['q4f16', 'q4']);
let generator = null;
let precision = null;
let inUse = false;

function report(type, detail = {}) {
  self.postMessage({ type, ...detail });
}
function describeProgress(event) {
  if (!event || typeof event !== 'object') return;
  const allowed = ['status', 'file', 'progress', 'loaded', 'total', 'name'];
  const progress = {};
  for (const key of allowed) {
    if (typeof event[key] === 'string' || Number.isFinite(event[key])) {
      progress[key] = event[key];
    }
  }
  report('progress', { progress });
}
async function load(requested) {
  if (generator) {
    if (requested !== precision) throw Error('La precisión cambió; liberá el modelo antes de volver a cargar.');
    report('ready', {model: MODEL_ID, precision});
    return;
  }
  if (!ALLOWED_PRECISIONS.has(requested)) throw Error('Cuantización no admitida');
  if (!self.navigator?.gpu) throw Error('Este navegador no ofrece WebGPU al worker.');
  const adapter = await self.navigator.gpu.requestAdapter();
  if (!adapter) throw Error('WebGPU está disponible pero no hay adaptador compatible.');
  report('progress', {progress: {status: 'loading-library'}});
  const {pipeline, env} = await import(TRANSFORMERS_URL);
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  // No descargar ningún recurso hasta que lo solicite expresamente el usuario.
  report('progress', {progress: {status: 'loading-model'}});
  generator = await pipeline('text-generation', MODEL_ID, {
    device: 'webgpu',
    dtype: requested,
    progress_callback: describeProgress,
  });
  precision = requested;
  report('ready', {model: MODEL_ID, precision});
}

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > 14) {
    throw Error('Historial de conversación inválido');
  }
  for (const item of messages) {
    if (!['system', 'user', 'assistant'].includes(item?.role) ||
        typeof item?.content !== 'string' || item.content.length > 1300) {
      throw Error('Mensaje inválido o demasiado largo');
    }
  }
  if (messages[messages.length - 1].role !== 'user') {
    throw Error('La conversación debe terminar con una pregunta');
  }
}
async function generate(id, messages) {
  if (!generator) throw Error('Cargá el modelo antes de conversar');
  if (inUse) throw Error('El modelo ya está generando');
  validateMessages(messages);
  inUse = true;
  try {
    const output = await generator(messages, {
      max_new_tokens: 160,
      do_sample: false,
      repetition_penalty: 1.08,
    });
    const generated = output?.[0]?.generated_text;
    const answer = Array.isArray(generated)
      ? generated[generated.length - 1]?.content
      : (typeof generated === 'string' ? generated : null);
    if (typeof answer !== 'string' || !answer.trim()) {
      throw Error('El modelo no devolvió texto de respuesta');
    }
    report('result', {id, answer: answer.trim()});
  } finally {
    inUse = false;
  }
}
self.addEventListener('message', async ({data}) => {
  const {type, id} = data || {};
  try {
    if (type === 'load') await load(data.precision);
    else if (type === 'generate') await generate(id, data.messages);
    else throw Error('Acción no reconocida');
  } catch (error) {
    report('error', {id, message: error?.message || String(error)});
  }
});
