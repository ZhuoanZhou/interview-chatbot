/* Layout checks use fabricated answers only; preview server runs on port 8511. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const schema=JSON.parse(fs.readFileSync('survey/schema.json','utf8'));
const harness=`<!doctype html><style>body{margin:0;padding:56px 16px 8px}iframe{display:block;width:100%;height:calc(100dvh - 64px);border:0}</style><iframe id="app"></iframe><script>
let args;
addEventListener('message',e=>{if(e.data.type==='streamlit:componentReady')document.querySelector('iframe').contentWindow.postMessage({type:'streamlit:render',args},'*')});
window.start=(schema,pageId,answers)=>{args={schema,record:{revision:0,state:{page:pageId,status:'active',answers}},session_key:'layout',preview:true};document.querySelector('iframe').src='/survey/frontend/index.html'};
</script>`;
(async()=>{
 const b=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const p=await b.newPage({viewport:{width:1366,height:768}});
  const errors=[];p.on('pageerror',e=>errors.push(e.message));
  fs.mkdirSync('tmp/survey-qa',{recursive:true});
  await p.goto('http://127.0.0.1:8511');
  await p.getByRole('button',{name:'Start a new survey',exact:true}).click();
  const frame=p.locator('iframe[title*="fixed_communication_survey"]');
  const live=p.frameLocator('iframe[title*="fixed_communication_survey"]');
  await live.getByRole('button',{name:'Start survey',exact:true}).click();
  await live.getByLabel('Other',{exact:true}).check();
  const box=await frame.boundingBox();
  // The participant-ID panel is open by default; the survey fills the space beside it.
  const main=await p.getByTestId('stMain').boundingBox();
  assert(box.width>=main.width-64,'Streamlit must use the available desktop width');
  assert(box.y+box.height<=769,'Frame fits visible screen');
  await p.screenshot({path:'tmp/survey-qa/wide-live.png'});
  await p.route('http://127.0.0.1:8512/**',route=>{
   const url=new URL(route.request().url());
   if(url.pathname==='/harness')return route.fulfill({contentType:'text/html',body:harness});
   const file=path.join(process.cwd(),url.pathname);
   return route.fulfill({contentType:file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(file)});
  });
  async function start(id){
   await p.goto('http://127.0.0.1:8512/harness');
   const answers={};
   for(const q of schema.pages){
    const other=q.other_option||'Other';
    if(q.options?.includes(other))answers[q.id]={status:'answered',choices:[other],other:{other:'Sample answer'}};
    if(q.fields)answers[q.id]={status:'answered',groups:Object.fromEntries(q.fields.map(f=>[f.id,'Other'])),other:Object.fromEntries(q.fields.map(f=>[f.id,'Sample answer']))};
   }
   await p.evaluate(([s,id,a])=>{sessionStorage.clear();start(s,id,a)},[schema,id,answers]);
   const app=p.frameLocator('#app');await app.locator('#question-title').waitFor();return app;
  }
  const overflow=[];
  for(const size of [{width:1280,height:720},{width:1366,height:768},{width:1920,height:1080}]){
   await p.setViewportSize(size);
   for(const q of schema.pages){
    const app=await start(q.id);
    const metrics=await app.locator('#page').evaluate(el=>({height:el.clientHeight,content:el.scrollHeight}));
    if(metrics.content>metrics.height+2)overflow.push({size,page:q.id,...metrics});
    assert.equal(await app.locator('#scroll-more').count(),0);
    assert(await app.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    for(const container of await app.locator('.answer-options:has(.other-row)').all()){
     const labels=await container.locator('.choice span').allTextContents();
     assert(['Other','It depends on something else','Do something else'].includes(labels.at(-1)));
     const otherBox=container.locator('.other-row textarea');
     assert(await otherBox.isVisible());
     const other=await container.locator('.other-row').boundingBox();
     const label=await container.locator('.other-row>.choice').boundingBox();
     const textbox=await otherBox.boundingBox();
     if(await container.locator(':scope > .options.fit-rows').count()){
      // Part 2: "Do something else" is the last cell of the answer grid, its text box on the same line.
      assert(await container.locator(':scope > .options.fit-rows > .other-row:last-child').count()===1,'Do something else is last');
      assert(textbox.x>=label.x+label.width-2,'Text box follows its label on the same line');
     }else{
      const grid=await container.locator(':scope > .options').boundingBox();
      assert(other.y>=grid.y+grid.height,'Other follows all other choices');
      assert(textbox.y>=label.y+label.height,'Other text box is stacked below its label');
      assert(Math.abs(textbox.x-label.x)<2,'Other label and text box share a left edge');
     }
    }
    if(size.width===1366&&['stop_trying','s1_action','e1_edit','edit_intro','feature_1','situation_1'].includes(q.id))await p.screenshot({path:`tmp/survey-qa/wide-${q.id}.png`});
   }
  }
  for(const size of [{width:390,height:844},{width:844,height:390},{width:390,height:420}]){
   await p.setViewportSize(size);const app=await start('stop_trying');
   const question=await app.locator('#page').boundingBox(), controls=await app.locator('#controls').boundingBox();
   assert(question.height>40&&question.y+question.height<=controls.y+1);
   for(const name of ['Next','Skip','Save and take a break','End my survey now']){
    const box=await app.getByRole('button',{name,exact:true}).boundingBox();
    assert(box.y>=0&&box.y+box.height<=size.height+1,`${name} outside viewport`);
   }
   await app.locator('#page').focus();await p.keyboard.press('End');
   await app.locator('#page').evaluate(el=>el.scrollTop=el.scrollHeight);
   assert(await app.locator('#page').evaluate(el=>el.scrollTop>0),'Small-screen overflow stays reachable');
   const after=await app.locator('#controls').boundingBox();assert.equal(after.y,controls.y);
   assert(await app.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await p.screenshot({path:`tmp/survey-qa/wide-${size.width}-${size.height}.png`});
  }
  assert.deepEqual(errors,[]);
  assert.deepEqual(overflow,[],'Desktop survey content, including Other fields, must fit');
  console.log('PASS: full-width Streamlit host, every survey page with expanded Other fields at three desktop sizes, and accessible overflow/fixed controls on small screens.');
 }finally{await b.close()}
})().catch(e=>{console.error(e);process.exit(1)});
