// MJON IA · caché sin API para el asistente local.
// Solo se descargan recursos estáticos pequeños. Los modelos neuronales no se
// precargan ni se interfieren; cada versión gestiona su propia integridad SHA.
const CACHE_NAME = 'mjon-verified-local-v1';
const CORE_PATHS = [
  './',
  './index.html',
  './web/style.css',
  './web/app.mjs',
  './web/runtime.mjs',
  './web/grounding.mjs',
  './web/knowledge-core.json'
];
const staticFiles = new Set(CORE_PATHS.map(path=>new URL(path,self.registration.scope).pathname));

self.addEventListener('install',event=>{
  event.waitUntil((async()=>{
    const cache=await caches.open(CACHE_NAME);
    await cache.addAll(CORE_PATHS);
    await self.skipWaiting();
  })());
});
self.addEventListener('activate',event=>{
  event.waitUntil((async()=>{
    const names=await caches.keys();
    await Promise.all(names.filter(x=>x.startsWith('mjon-verified-local-')&&x!==CACHE_NAME)
      .map(x=>caches.delete(x)));
    await self.clients.claim();
  })());
});
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET')return;
  const u=new URL(event.request.url);
  if(u.origin!==self.location.origin || !staticFiles.has(u.pathname))return;
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE_NAME);
    const copy=await cache.match(event.request,{ignoreSearch:true});
    // Una versión previamente verificada permanece utilizable sin Internet.
    const update=fetch(event.request).then(async response=>{
      if(response.ok && response.type==='basic'){
        await cache.put(event.request,response.clone());
      }
      return response;
    }).catch(()=>null);
    if(copy){
      event.waitUntil(update);
      return copy;
    }
    const response=await update;
    if(response)return response;
    return new Response('MJON IA: recurso local todavía no disponible',{status:503,
      headers:{'content-type':'text/plain; charset=utf-8'}});
  })());
});
