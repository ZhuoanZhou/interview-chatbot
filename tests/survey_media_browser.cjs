/* Run the fixture app on port 8514 with server.baseUrlPath=study. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  if(process.argv.includes('--fixture')){
   const clip=await require('./survey_video_fixture.cjs')(browser);
   fs.mkdirSync('tmp',{recursive:true});fs.writeFileSync('tmp/survey-demo-fixture.mp4',clip.bytes);
   console.log(`Generated ${clip.bytes.length}-byte ${clip.type} test clip`);return;
  }
  const p=await browser.newPage();
  const errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://127.0.0.1:8514/study');
  const app=p.frameLocator('iframe[title*="survey_media_test"]');
  const video=app.locator('video');
  await video.waitFor();
  const url=await video.getAttribute('src');
  assert.match(url,/^http:\/\/127\.0\.0\.1:8514\/study\/media\/[^/]+\.mp4$/);
  await video.evaluate(el=>new Promise((resolve,reject)=>{
   const timeout=setTimeout(()=>reject(new Error('Video load timed out')),10000);
   const ready=()=>{clearTimeout(timeout);resolve()};
   const failed=()=>{clearTimeout(timeout);reject(new Error('Video load failed'))};
   if(el.error)failed();else if(el.readyState>=2)ready();else{
    el.addEventListener('loadeddata',ready,{once:true});el.addEventListener('error',failed,{once:true});
   }
  }));
  const response=await p.request.get(url,{headers:{Range:'bytes=0-31'}});
  assert.equal(response.status(),206);
  assert.equal((await response.body()).length,32);
  assert.equal(response.headers()['accept-ranges'],'bytes');
  await video.evaluate(async el=>{el.dataset.original='yes';el.muted=true;await el.play()});
  await p.getByRole('button',{name:'Rerun host',exact:true}).click();
  await p.waitForTimeout(600);
  assert.equal(await video.getAttribute('data-original'),'yes','Host reruns preserve the player');
  assert.equal((await p.request.get(url,{headers:{Range:'bytes=-32'}})).status(),206,'Media survives host rerun');
  const ready=await video.evaluate(el=>({ready:el.readyState,error:el.error}));
  assert(ready.ready>=2&&!ready.error);
  assert.deepEqual(errors,[]);
  console.log('PASS: real Streamlit media delivery, base-path URL, byte-range requests, playable video, and player/media retained across host reruns.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
