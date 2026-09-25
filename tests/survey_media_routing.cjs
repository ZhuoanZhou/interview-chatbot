/* Host routing regression tests. Fabricated clip and generic local URLs only. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const schema=JSON.parse(fs.readFileSync('survey/schema.json','utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const clip=await require('./survey_video_fixture.cjs')(browser);
  for(const prefix of ['', '/study', '/~/+', '/~/+/study']){
   const page=await browser.newPage();const requests=[];
   await page.route('http://127.0.0.1:8516/**',route=>{
    const pathname=new URL(route.request().url()).pathname;
    if(pathname===prefix+'/')return route.fulfill({contentType:'text/html',
     headers:{'Referrer-Policy':'origin'},
     body:`<!doctype html><iframe id="app" src="${prefix}/component/fixture/index.html"></iframe><script>
      addEventListener('message',e=>{if(e.data.type==='streamlit:componentReady')document.querySelector('iframe').contentWindow.postMessage({type:'streamlit:render',args:{schema:${JSON.stringify(schema)},record:{revision:0,state:{page:'demo_video',status:'active',answers:{demo_consent:{status:'answered',choices:['Yes']}}}},session_key:'routing',demo_url:'/media/fixture.mp4'}},'*')});
     </script>`});
    if(pathname===prefix+'/media/fixture.mp4'){
     requests.push(pathname);return route.fulfill({contentType:clip.type,body:clip.bytes});
    }
    const component=prefix+'/component/fixture/';
    if(pathname.startsWith(component)){
     const name=pathname.slice(component.length);
     if(['index.html','survey.js','survey.css'].includes(name))return route.fulfill({
      contentType:name.endsWith('.js')?'application/javascript':name.endsWith('.css')?'text/css':'text/html',
      body:fs.readFileSync('survey/frontend/'+name)});
    }
    return route.fulfill({status:404,body:'Wrong backend route'});
   });
   await page.goto('http://127.0.0.1:8516'+prefix+'/');
   const app=page.frameLocator('#app');const video=app.locator('video');
   await video.waitFor();
   assert.equal(await video.getAttribute('src'),'http://127.0.0.1:8516'+prefix+'/media/fixture.mp4');
   await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).waitFor();
   await video.evaluate(el=>new Promise((resolve,reject)=>{
    if(el.error)reject(new Error('Video failed'));else if(el.readyState>=2)resolve();else{
     el.addEventListener('loadeddata',resolve,{once:true});
     el.addEventListener('error',()=>reject(new Error('Video failed')),{once:true});
    }
   }));
   assert.equal(await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).isEnabled(),true);
   assert(requests.length>0,'Clip fetched through expected backend route');
   await page.close();
  }
  console.log('PASS: playable media at root, base path, Cloud-style proxy prefix, and combined proxy/base path with origin-only referrers.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
