/* Isolated frontend tests: no real accounts, credentials, participants, or video. */
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const schema=JSON.parse(fs.readFileSync('survey/schema.json','utf8'));

const harness=`<!doctype html><iframe id="app" title="survey" style="width:900px;height:1200px;border:0"></iframe>
<script>
window.batches=[];window.attempts=[];window.failSaves=false;window.delay=40;window.args=null;
const frame=document.getElementById('app');
window.render=()=>frame.contentWindow.postMessage({type:'streamlit:render',args:window.args},'*');
window.addEventListener('message',e=>{
 if(e.data.type==='streamlit:componentReady')render();
 if(e.data.type==='streamlit:setComponentValue'){
  const packet=e.data.value;window.attempts.push(packet);
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
 // Editing, idle time, visibility changes and draft refresh must not upload.
 await page.waitForTimeout(3300);
 assert.equal(await page.evaluate(()=>attempts.length),0,'Choices and clearing stay local');
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 assert.equal((await page.evaluate(()=>batches[0].state.answers.aac)).status,'skipped');
 app=await start('closing');
 let input=app.getByRole('textbox');
 await input.pressSequentially('abc',{delay:120});
 await input.press('Backspace');
 await input.evaluate(el=>{el.value+='xy';el.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertFromPaste',data:'xy'}));});
 const draftEvents=await app.locator('body').evaluate(()=>JSON.parse(sessionStorage.getItem('communication-survey:fixture')).events);
 await app.locator('body').evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
 await page.waitForTimeout(3500);
 assert.equal(await page.evaluate(()=>attempts.length),0,'Typing and tab visibility must not upload');
 await page.evaluate(()=>document.getElementById('app').contentWindow.location.reload());
 await app.getByRole('textbox').waitFor();
 assert.equal(await app.getByRole('textbox').inputValue(),'abxy');
 await page.waitForTimeout(3300);
 assert.equal(await page.evaluate(()=>attempts.length),0,'Refreshing an unsaved draft must not upload');
 // Rapid navigation queues snapshots; subsequent typing is not included on ACK.
 await page.evaluate(()=>{window.delay=800;});
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Back',exact:true}).click();
 input=app.getByRole('textbox');
 await input.click();
 await input.press('End');
 await input.pressSequentially('pending',{delay:20});
 await page.waitForFunction(()=>batches.length===2);
 await page.waitForTimeout(3300);
 assert.equal(await input.evaluate(el=>el===document.activeElement),true);
 let batches=await page.evaluate(()=>batches);
 assert.equal(batches.length,2);
 assert.equal(batches[0].state.answers.closing.text,'abxy');
 assert.equal(batches[1].state.answers.closing.text,'abxy','Queued Back save excludes later typing');
 assert.equal(batches[1].base_revision,batches[0].base_revision+1);
 const savedEvents=batches.flatMap(b=>b.events);
 for(const event of draftEvents) assert.deepEqual(savedEvents.find(e=>e.event_id===event.event_id),event,'Original event and timestamp preserved');
 assert(savedEvents.some(e=>e.input_type==='insertFromPaste'&&e.inserted==='xy'));
 assert(savedEvents.some(e=>e.type==='text_input'&&e.deleted==='c'));
 // A failed requested save retries unchanged after refresh; newer drafts stay local.
 await page.evaluate(()=>{window.failSaves=true;});
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText(/Your latest changes have not been saved yet/).waitFor();
 const failedId=await page.evaluate(()=>attempts.at(-1).batch_id);
 await page.evaluate(()=>{window.failSaves=false;document.getElementById('app').contentWindow.location.reload();});
 await page.waitForFunction(()=>batches.length===3);
 batches=await page.evaluate(()=>batches);
 assert.equal(batches.at(-1).batch_id,failedId);
 assert.equal(batches.at(-1).state.answers.closing.text,'abxypending');
 assert.equal(new Set(batches.map(b=>b.batch_id)).size,batches.length);
 // Submit is usable even with unsent focus/session events, and saves before completion.
 await app.getByRole('button',{name:'Submit survey',exact:true}).click();
 await app.getByText('Your responses have been saved. You can now close this tab.').waitFor();
 assert.equal(await page.evaluate(()=>batches.at(-1).state.status),'submitted');
 // Pending navigation snapshots survive refresh without absorbing newer edits.
 app=await start('closing');
 await app.getByRole('textbox').fill('requested answer');
 await page.evaluate(()=>{window.failSaves=true;});
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText(/Your latest changes have not been saved yet/).waitFor();
 await app.getByRole('button',{name:'Back',exact:true}).click();
 await app.getByRole('textbox').fill('new unsaved answer');
 await page.evaluate(()=>{window.failSaves=false;document.getElementById('app').contentWindow.location.reload();});
 await page.waitForFunction(()=>batches.length===2);
 assert.equal(await app.getByRole('textbox').inputValue(),'new unsaved answer');
 assert.equal(await page.evaluate(()=>batches.at(-1).state.answers.closing.text),'requested answer');
 // Automatic retry completes an explicit save without another participant click.
 await page.evaluate(()=>{window.failSaves=true;});
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText(/Your latest changes have not been saved yet/).waitFor();
 const retryId=await page.evaluate(()=>attempts.at(-1).batch_id);
 await page.evaluate(()=>{window.failSaves=false;});
 await page.waitForFunction(()=>batches.length===3,null,{timeout:20000});
 assert.equal(await page.evaluate(()=>batches.at(-1).batch_id),retryId);
 assert.equal(await page.evaluate(()=>batches.at(-1).state.answers.closing.text),'new unsaved answer');
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
 console.log('PASS: action-only uploads, original timestamps, draft/queue refresh recovery, sequential navigation snapshots, automatic save retries, demo ratings, desktop/mobile buttons, pause and final submission.');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
