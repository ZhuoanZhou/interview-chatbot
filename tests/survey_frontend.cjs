/* Isolated frontend tests: no real accounts, credentials, participants, or video. */
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const schema=JSON.parse(fs.readFileSync('survey/schema.json','utf8'));

const harness=`<!doctype html><iframe id="app" title="survey" style="width:900px;height:1200px;border:0"></iframe>
<script>
window.batches=[];window.failSaves=false;window.delay=40;window.args=null;
const frame=document.getElementById('app');
window.render=()=>frame.contentWindow.postMessage({type:'streamlit:render',args:window.args},'*');
window.addEventListener('message',e=>{
 if(e.data.type==='streamlit:componentReady')render();
 if(e.data.type==='streamlit:setComponentValue'){
  const packet=e.data.value;
  if(window.failSaves){window.args.error='Simulated connection loss';render();return;}
  setTimeout(()=>{
   if(!batches.some(b=>b.batch_id===packet.batch_id))batches.push(packet);
   args.record={...args.record,revision:packet.base_revision+1,batch_id:packet.batch_id,state:packet.state};
   Object.assign(args,{ack:packet.batch_id,revision:packet.base_revision+1,saved_at:new Date().toISOString(),error:''});render();
  },window.delay);
 }
});
window.start=(schema,record)=>{args={schema,record,session_key:'fixture',preview:true,demo_data:btoa('fixture'),demo_error:''};frame.src='/survey/frontend/index.html';};
</script>`;

(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage();
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('http://127.0.0.1:8512/**',route=>{
  const url=new URL(route.request().url());
  if(url.pathname==='/harness')return route.fulfill({contentType:'text/html',body:harness});
  const file=path.join(process.cwd(),url.pathname);
  const type=file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html';
  return route.fulfill({contentType:type,body:fs.readFileSync(file)});
 });
 async function start(pageId,answers={}){
  await page.goto('http://127.0.0.1:8512/harness');
  await page.evaluate(([schema,pageId,answers])=>{
   sessionStorage.clear();
   start(schema,{revision:0,schema_version:schema.version,state:{page:pageId,status:'active',answers}});
  },[schema,pageId,answers]);
  const app=page.frameLocator('#app');
  await app.locator('#question-title').waitFor();return app;
 }
 // Every Other in the schema must reveal optional input, also for radio and grouped fields.
 let app=await start('aac');
 await app.getByLabel('Other',{exact:true}).check();
 await app.getByRole('textbox').fill('communication board');
 await app.getByLabel('No',{exact:true}).check();
 assert.equal(await app.getByRole('textbox').count(),0);
 await app.getByLabel('Other',{exact:true}).check();
 assert.equal(await app.getByRole('textbox').inputValue(),'');
 await app.getByRole('button',{name:'Clear answer',exact:true}).click();
 assert.equal(await app.getByRole('textbox').count(),0);
 // Real browser insertion/deletion continues while acknowledgements arrive.
 app=await start('closing');
 await page.evaluate(()=>{window.delay=350;});
 const input=app.getByRole('textbox');
 await input.pressSequentially('abc',{delay:120});
 await page.waitForTimeout(1700);
 assert.equal(await input.evaluate(el=>el===document.activeElement),true);
 await input.press('Backspace');
 // Inject an InputEvent to verify assistive/paste metadata handling, not physical-key timing.
 await input.evaluate(el=>{el.value+='xy';el.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertFromPaste',data:'xy'}));});
 await page.waitForTimeout(3500);
 let batches=await page.evaluate(()=>batches);
 assert(batches.flatMap(b=>b.events).some(e=>e.input_type==='insertFromPaste'&&e.inserted==='xy'));
 assert(batches.flatMap(b=>b.events).some(e=>e.type==='text_input'&&e.deleted==='c'));
 // Connection failure: retain edits and replay the same immutable batch after iframe refresh.
 await page.evaluate(()=>{window.failSaves=true;});
 await input.pressSequentially('pending');
 await page.waitForTimeout(3300);
 await app.getByText(/Your latest changes have not been saved yet/).waitFor();
 await page.evaluate(()=>{window.failSaves=false;document.getElementById('app').contentWindow.location.reload();});
 await app.getByRole('textbox').waitFor();
 assert.equal(await app.getByRole('textbox').inputValue(),'abxypending');
 await page.waitForTimeout(2000);
 batches=await page.evaluate(()=>batches);
 assert.equal(batches.at(-1).state.answers.closing.text,'abxypending');
 assert.equal(new Set(batches.map(b=>b.batch_id)).size,batches.length);
 // The complete demo branch preserves both rating tables and retry branching.
 app=await start('demo_consent');
 await app.getByLabel('Yes',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).click();
 for(let i=1;i<=6;i++){
  await app.getByText('Parts of the system · '+i+' of 6',{exact:true}).waitFor();
  await app.getByLabel('Somewhat useful',{exact:true}).check();
  await app.getByRole('button',{name:'Next',exact:true}).click();
 }
 await app.getByLabel('Usually easier',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Other',{exact:true}).check();
 await app.getByRole('textbox').fill('point');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Change another word and try a new version again',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Two more',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 for(let i=1;i<=7;i++){
  await app.getByText('Situations · '+i+' of 7',{exact:true}).waitFor();
  await app.getByLabel('N/A',{exact:true}).check();
  await app.getByRole('button',{name:'Next',exact:true}).click();
 }
 await app.getByRole('textbox').fill('done');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Submit survey',exact:true}).click();
 await app.getByText('Your responses have been saved. You can now close this tab.').waitFor();
 batches=await page.evaluate(()=>batches);
 const final=batches.at(-1).state;
 assert.equal(final.answers.retry_count.choices[0],'Two more');
 assert.equal(final.answers.feature_6.choices[0],'Somewhat useful');
 assert.equal(final.answers.situation_7.choices[0],'N/A');
 assert.equal(final.status,'submitted');
 // Ending early requires confirmation and preserves answers when cancelled.
 app=await start('closing');
 await app.getByRole('textbox').fill('Please keep this answer');
 for (const width of [900, 360]) {
  await page.locator('#app').evaluate((el,width)=>{el.style.width=width+'px';},width);
  const pause=await app.getByRole('button',{name:'Save and take a break',exact:true}).boundingBox();
  const end=await app.getByRole('button',{name:'End my survey now',exact:true}).boundingBox();
  const footer=await app.locator('#footer').boundingBox();
  assert(pause.height>=52 && end.height>=52);
  assert(Math.abs(pause.y-end.y)<1,'Secondary actions share one row');
  assert(end.x>=pause.x+pause.width+10);
  assert(end.x+end.width<=footer.x+footer.width+1,'Actions fit without horizontal scrolling');
 }
 await app.getByRole('button',{name:'End my survey now',exact:true}).click();
 await app.getByText(/You will not be able to return to answer more questions/).waitFor();
 assert.equal(await app.getByRole('heading',{name:'End your survey now?'}).evaluate(el=>el===document.activeElement),true);
 await app.getByRole('button',{name:'Keep going',exact:true}).click();
 assert.equal(await app.getByRole('textbox').inputValue(),'Please keep this answer');
 await app.getByRole('button',{name:'Save and take a break',exact:true}).click();
 await app.getByText('You can now close this tab and return later using your participant ID.',{exact:true}).waitFor();
 await app.getByRole('button',{name:'Continue survey',exact:true}).click();
 await app.getByRole('button',{name:'End my survey now',exact:true}).click();
 await app.getByRole('button',{name:'End and submit survey',exact:true}).click();
 await app.getByText('Your responses have been saved. You can now close this tab.').waitFor();
 batches=await page.evaluate(()=>batches);
 assert.equal(batches.at(-1).state.status,'ended');
 assert.equal(batches.at(-1).state.answers.closing.text,'Please keep this answer');
 assert.deepEqual(errors,[]);
 console.log('PASS: demo ratings, Other fields, input focus, paste/deletion, failed-save recovery, pending-event refresh, desktop/mobile action buttons, pause and confirmed early submission.');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
