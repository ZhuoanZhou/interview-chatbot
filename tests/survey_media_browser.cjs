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
  const p=await browser.newPage({viewport:{width:1366,height:768}});
  const errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://127.0.0.1:8514/study');
  const frame=p.locator('iframe[title*="fixed_communication_survey"]');
  const app=p.frameLocator('iframe[title*="fixed_communication_survey"]');
  const video=p.locator('.st-key-survey-demo video[data-testid="stVideo"]');
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
  assert.equal(await app.locator('video').count(),0,'Video must be rendered by st.video outside the survey iframe');
  await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).waitFor();
  await video.evaluate(async el=>{el.dataset.original='yes';el.muted=true;await el.play()});
  await video.evaluate(el=>el.pause());
  await video.evaluate(el=>new Promise(resolve=>{el.addEventListener('seeked',resolve,{once:true});el.currentTime=.2}));
  await video.evaluate(el=>new Promise(resolve=>{el.addEventListener('ended',resolve,{once:true});el.play()}));
  const events=await app.locator('body').evaluate(()=>JSON.parse(sessionStorage.getItem(Object.keys(sessionStorage).find(key=>key.startsWith('communication-survey:')))).events);
  for(const kind of ['play','pause','seeked','ended'])assert(events.some(e=>e.type==='video_'+kind&&Number.isFinite(e.browser_time_origin_ms)),kind+' is logged with the native clock origin');
  await p.getByTestId('stExpandSidebarButton').click();
  await p.getByRole('button',{name:'Rerun host',exact:true}).click();
  await p.waitForTimeout(600);
  assert.equal(await video.getAttribute('data-original'),'yes','Host reruns preserve the player');
  assert.equal((await p.request.get(url,{headers:{Range:'bytes=-32'}})).status(),206,'Media survives host rerun');
  const ready=await video.evaluate(el=>({ready:el.readyState,error:el.error}));
  assert(ready.ready>=2&&!ready.error);
  await p.getByTestId('stSidebarCollapseButton').click();
  for(const size of [{width:1366,height:768},{width:1280,height:720},{width:390,height:844},{width:844,height:390},{width:390,height:420}]){
   await p.setViewportSize(size);await p.waitForTimeout(200);
   const playerBox=await video.boundingBox(), frameBox=await frame.boundingBox();
   assert(playerBox.height>=60&&playerBox.y+playerBox.height<=frameBox.y+1,'Native player and survey must not overlap');
   assert(frameBox.y+frameBox.height<=size.height+1,'Survey controls fit viewport');
   for(const name of ['Back','Skip demonstration','I have watched the demonstration','Save and take a break','End my survey now']){
    const control=await app.getByRole('button',{name,exact:true}).boundingBox();
    assert(control.y>=0&&control.y+control.height<=size.height+1,`${name} outside ${JSON.stringify(size)}`);
   }
   fs.mkdirSync('tmp/survey-qa',{recursive:true});
   await p.screenshot({path:`tmp/survey-qa/native-video-${size.width}-${size.height}.png`});
  }
  await app.getByRole('button',{name:'Back',exact:true}).click();
  await app.getByRole('heading',{name:/Would you like/}).waitFor();
  assert.equal(await video.isVisible(),false,'Native video disappears on navigation');
  await app.getByRole('button',{name:'Next',exact:true}).click();
  await video.waitFor();
  await app.getByRole('button',{name:'Save and take a break',exact:true}).click();
  await app.getByRole('heading',{name:'Your survey is paused',exact:true}).waitFor();
  assert.equal(await video.isVisible(),false,'Native video disappears while paused');
  assert.deepEqual(errors,[]);
  console.log('PASS: native st.video, byte-range requests, playback/seek/end logs, host reruns, navigation/pause, and desktop/mobile layouts.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
