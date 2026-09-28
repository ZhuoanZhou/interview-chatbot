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
window.start=(schema,record)=>{args={schema,record,session_key:'fixture',preview:true,demo_url:'/media/fixture.mp4',demo_error:''};frame.src='/survey/frontend/index.html';};
</script>`;

(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const clip=await require('./survey_video_fixture.cjs')(browser);
 const page=await browser.newPage();
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('http://127.0.0.1:8512/**',route=>{
  const url=new URL(route.request().url());
  if(url.pathname==='/harness')return route.fulfill({contentType:'text/html',body:harness});
  if(url.pathname==='/media/fixture.mp4')return route.fulfill({contentType:clip.type,body:clip.bytes});
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
 // Other input is always visible; typing selects Other and deselection clears it.
 let app=await start('aac');
 assert.equal(await app.getByRole('textbox').count(),1);
 await app.getByRole('textbox').fill('communication board');
 assert(await app.getByLabel('Other',{exact:true}).isChecked());
 await app.getByLabel('No',{exact:true}).check();
 assert.equal(await app.getByRole('textbox').inputValue(),'');
 await app.getByLabel('Other',{exact:true}).check();
 assert.equal(await app.getByRole('textbox').inputValue(),'');
 await app.getByRole('button',{name:'Clear answer',exact:true}).click();
 assert.equal(await app.getByRole('textbox').count(),1);
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
 app=await start('demo_video',{demo_consent:{status:'answered',choices:['Yes']}});
 await app.locator('video').evaluate(video=>new Promise(resolve=>{if(video.readyState>=2)resolve();else video.addEventListener('loadeddata',resolve,{once:true})}));
 assert.equal(await app.locator('video').getAttribute('src'),'http://127.0.0.1:8512/media/fixture.mp4');
 await app.locator('video').evaluate(video=>{video.dataset.original='true';video.dispatchEvent(new Event('play'))});
 await page.evaluate(()=>render());
 assert.equal(await app.locator('video').getAttribute('data-original'),'true','Save acknowledgements must not restart playback');
 await app.getByRole('button',{name:'Skip demonstration',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 assert((await page.evaluate(()=>batches[0].events)).some(e=>e.type==='video_play'));
 // A failed media request stays skippable and cannot count as watching.
 await page.route('**/media/fixture.mp4',route=>route.fulfill({status:503,body:'Unavailable'}));
 app=await start('demo_video');
 await app.getByText(/The video could not be loaded/).waitFor();
 assert.equal(await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).isEnabled(),false);
 assert.equal(await app.getByRole('button',{name:'Skip demonstration',exact:true}).isEnabled(),true);
 await page.unroute('**/media/fixture.mp4');
 app=await start('demo_consent');
 await app.getByLabel('Yes',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'I have watched the demonstration',exact:true}).click();
 // The five examples return with an editable transcript, before the ratings.
 await app.getByRole('heading',{name:'Example 1 — At a pharmacy'}).waitFor();
 assert(await app.getByText('Example 1 of 5',{exact:true}).isVisible());
 assert(await app.getByText('What you meant to say',{exact:true}).isVisible());
 let transcript=app.getByRole('textbox',{name:'Transcript'});
 assert.equal(await transcript.inputValue(),'I’m picking up the prescription for Mark Line.');
 await transcript.click();await transcript.press('End');
 for(let i=0;i<'Mark Line.'.length;i++)await transcript.press('Backspace');
 await transcript.pressSequentially('Mara Klein.',{delay:20});
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 assert.equal(await app.getByRole('textbox',{name:'Transcript'}).inputValue(),'Can you tell me there the letter of us?');
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 await app.getByRole('textbox',{name:'Transcript'}).fill('No peanuts');
 await app.getByRole('button',{name:'Reset text',exact:true}).click();
 assert.equal(await app.getByRole('textbox',{name:'Transcript'}).inputValue(),'Some peanuts, please, I’m a little sick.');
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 assert(await app.getByText('Example 5 of 5',{exact:true}).isVisible());
 await app.getByRole('button',{name:'Back',exact:true}).click();
 await app.getByRole('button',{name:'Back',exact:true}).click();
 await app.getByRole('button',{name:'Back',exact:true}).click();
 await app.getByRole('button',{name:'Back',exact:true}).click();
 assert.equal(await app.getByRole('textbox',{name:'Transcript'}).inputValue(),'I’m picking up the prescription for Mara Klein.','Edits persist after Back');
 for(let i=0;i<4;i++)await app.getByRole('button',{name:'Next example',exact:true}).click();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 for(let i=1;i<=6;i++){
  await app.locator(`input[name="feature_${i}"][value="Somewhat useful"]`).check();
 }
 assert.equal(await app.locator('.rating-row').count(),6);
 assert.equal(await app.locator('.rating-row input:checked').count(),6);
 await app.getByRole('button',{name:'Next',exact:true}).click();
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
  await app.locator(`input[name="situation_${i}"][value="Not sure"]`).check();
 }
 assert.equal(await app.locator('.rating-row').count(),7);
 assert.equal(await app.locator('.rating-row input:checked').count(),7);
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('textbox').fill('done');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Submit survey',exact:true}).click();
 await app.getByText('Your responses have been saved. You can now close this tab.').waitFor();
 batches=await page.evaluate(()=>batches);
 const final=batches.at(-1).state;
 assert.deepEqual(final.answers.e1_edit,{status:'answered',text:'I’m picking up the prescription for Mara Klein.',original:'I’m picking up the prescription for Mark Line.',edited:true});
 assert.equal(final.answers.e2_edit.edited,false);assert.equal(final.answers.e2_edit.status,'answered');
 // Example 3 was skipped, then passed with Next on the way forward again: kept as it is.
 assert.equal(final.answers.e3_edit.edited,false);
 assert(batches.flatMap(b=>b.events).some(e=>e.field_id==='e3_edit'&&e.type==='skip'));
 assert.equal(final.answers.e4_edit.text,'Some peanuts, please, I’m a little sick.');assert.equal(final.answers.e4_edit.edited,false);
 assert.equal(final.answers.e5_edit.status,'answered');
 const demoEvents=batches.flatMap(b=>b.events);
 assert(demoEvents.some(e=>e.field_id==='e1_edit'&&e.type==='keydown'&&e.key==='Backspace'&&Number.isFinite(e.elapsed_ms)));
 assert(demoEvents.some(e=>e.field_id==='e1_edit'&&e.type==='text_input'&&e.deleted==='.'));
 assert(demoEvents.some(e=>e.field_id==='e4_edit'&&e.type==='transcript_reset'&&e.previous.text==='No peanuts'));
 assert(demoEvents.some(e=>e.field_id==='e2_edit'&&e.type==='transcript_unchanged'));
 assert.equal(final.answers.retry_count.choices[0],'Two more');
 assert.equal(final.answers.feature_6.choices[0],'Somewhat useful');
 assert.equal(final.answers.situation_7.choices[0],'Not sure');
 assert.equal(final.status,'submitted');
 // Resume from an old item, retain historical values, and navigate the whole set.
 const watched={demo_consent:{status:'answered',choices:['Yes']},demo_video:{status:'answered',choices:['watched']}};
 app=await start('feature_4',{...watched,feature_4:{status:'answered',choices:['N/A']}});
 assert.equal(await app.locator('.rating-row').count(),6);
 await app.getByText('Previously answered: N/A. Choose a rating to change it.',{exact:true}).waitFor();
 await app.locator('input[name="feature_1"][value="Very useful"]').check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 let resumed=await page.evaluate(()=>batches[0].state.answers);
 assert.equal(resumed.feature_4.choices[0],'N/A');
 assert.equal(resumed.feature_1.choices[0],'Very useful');
 assert.equal(resumed.feature_2.status,'unanswered');
 await app.getByRole('button',{name:'Back',exact:true}).click();
 assert.equal(await app.locator('.rating-row').count(),6);
 await app.locator('input[name="feature_4"][value="Not sure"]').check();
 assert.equal(await app.locator('.legacy-rating').count(),0);
 await app.getByRole('button',{name:'Clear answer',exact:true}).click();
 assert.equal(await app.locator('.rating-row input:checked').count(),0);
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 await page.waitForFunction(()=>batches.length===3);
 resumed=await page.evaluate(()=>batches.at(-1).state.answers);
 for(let i=1;i<=6;i++)assert.equal(resumed['feature_'+i].status,'skipped');
 // A session saved on a removed follow-up screen resumes on its example; old answers are kept.
 app=await start('s2_words',{s2_action:{status:'answered',choices:['Change the text and show it']},s2_words:{status:'answered',text:'elevator'}});
 await app.getByText('Example 2 — Asking for directions',{exact:true}).waitFor();
 assert(await app.getByLabel('Change the text and show it',{exact:true}).isChecked());
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText('Example 3 — Talking at home',{exact:true}).waitFor();
 await page.waitForFunction(()=>batches.length===1);
 assert.equal(await page.evaluate(()=>batches[0].state.answers.s2_words.text),'elevator');
 // Typing Other in one story group must not change the other group.
 app=await start('story');
 await app.locator('fieldset').nth(0).getByRole('textbox').fill('a neighbour');
 await app.locator('fieldset').nth(1).getByRole('textbox').fill('the park');
 assert.equal(await app.locator('input:checked').count(),2);
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 const story=await page.evaluate(()=>batches[0].state.answers.story);
 assert.deepEqual(story.groups,{person:'Other',place:'Other'});
 assert.deepEqual(story.other,{person:'a neighbour',place:'the park'});
 // Typing Other respects exclusive answers on multiple-choice questions.
 app=await start('text_input');
 await app.getByLabel('I do not enter text',{exact:true}).check();
 await app.getByRole('textbox').fill('eye tracking');
 assert.equal(await app.getByLabel('I do not enter text',{exact:true}).isChecked(),false);
 assert(await app.getByLabel('Other',{exact:true}).isChecked());
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
