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
 // Q1 ratings, then the other Part 3 questions; the optional exercise comes last.
 for(let i=1;i<=6;i++){
  await app.locator(`input[name="feature_${i}"][value="Somewhat useful"]`).check();
 }
 assert.equal(await app.locator('.rating-row').count(),6);
 assert.equal(await app.locator('.rating-row input:checked').count(),6);
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Typing the word myself',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Other',{exact:true}).check();
 await app.getByRole('textbox').fill('point');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Correct another word and use Re-check again',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByLabel('Two more times',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 await app.getByRole('button',{name:'Skip',exact:true}).click();
 for(let i=1;i<=7;i++){
  await app.locator(`input[name="situation_${i}"][value="Not sure"]`).check();
 }
 assert.equal(await app.locator('.rating-row').count(),7);
 assert.equal(await app.locator('.rating-row input:checked').count(),7);
 await app.getByRole('button',{name:'Next',exact:true}).click();
 // Optional exercise at the end of Part 3.
 await app.getByRole('heading',{name:'Try the example situations'}).waitFor();
 assert(await app.getByText(/It does not include speech-to-text or the “Re-check” function shown in the video/).isVisible());
 await app.getByLabel('Yes, I’d like to try',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 // Example 1: edit the transcript directly, then Next.
 await app.getByRole('heading',{name:'Example 1 — At a pharmacy'}).waitFor();
 assert(await app.getByText('Example 1 of 5',{exact:true}).isVisible());
 assert.equal(await app.getByText('What would you do first in this situation?').count(),0,'No question step after the demo');
 let transcript=app.getByRole('textbox',{name:'Transcript'});
 assert.equal(await transcript.inputValue(),'I’m picking up the prescription for Mark Line.');
 await transcript.click();await transcript.press('End');
 for(let i=0;i<'Mark Line.'.length;i++)await transcript.press('Backspace');
 await transcript.pressSequentially('Mara Klein.',{delay:20});
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 // Example 2: Keep as is moves straight on.
 await app.getByRole('heading',{name:'Example 2 — Asking for directions'}).waitFor();
 await app.getByRole('button',{name:'Keep as is',exact:true}).click();
 // Example 3: Say it again moves straight on.
 await app.getByRole('heading',{name:'Example 3 — Talking at home'}).waitFor();
 await app.getByRole('button',{name:'Say it again',exact:true}).click();
 // Example 4: Delete all, type, Reset, then Delete all and type again.
 await app.getByRole('heading',{name:'Example 4 — Ordering food'}).waitFor();
 transcript=app.getByRole('textbox',{name:'Transcript'});
 await app.getByRole('button',{name:'Delete all',exact:true}).click();
 assert.equal(await transcript.inputValue(),'');
 assert.equal(await transcript.evaluate(el=>el===document.activeElement),true,'Delete all keeps focus in the text box');
 await transcript.pressSequentially('No nuts',{delay:20});
 await app.getByRole('button',{name:'Reset',exact:true}).click();
 assert.equal(await transcript.inputValue(),'Some peanuts, please, I’m a little sick.');
 await app.getByRole('button',{name:'Delete all',exact:true}).click();
 await transcript.pressSequentially('No peanuts please',{delay:20});
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 // Example 5: start editing, then Abandon: moves on to the closing question.
 await app.getByRole('heading',{name:'Example 5 — Talking with a friend'}).waitFor();
 await app.getByRole('textbox',{name:'Transcript'}).fill('I’m proud');
 await app.getByRole('button',{name:'Abandon',exact:true}).click();
 await app.getByText(/Is there anything else you want us to know/).waitFor();
 // Going back shows the recorded decision.
 await app.getByRole('button',{name:'Back',exact:true}).click();
 assert.equal(await app.getByRole('button',{name:'Abandon',exact:true}).getAttribute('aria-pressed'),'true');
 assert.equal(await app.getByRole('button',{name:'Keep as is',exact:true}).getAttribute('aria-pressed'),'false');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('textbox').fill('done');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByRole('button',{name:'Submit survey',exact:true}).click();
 await app.getByText('Your responses have been saved. You can now close this tab.').waitFor();
 batches=await page.evaluate(()=>batches);
 const final=batches.at(-1).state;
 const P='Some peanuts, please, I’m a little sick.';
 assert.deepEqual(final.answers.e1_edit,{status:'answered',decision:'edited',text:'I’m picking up the prescription for Mara Klein.',original:'I’m picking up the prescription for Mark Line.',edited:true});
 assert.deepEqual(final.answers.e2_edit,{status:'answered',decision:'kept',text:'Can you tell me there the letter of us?',original:'Can you tell me there the letter of us?',edited:false});
 assert.equal(final.answers.e3_edit.decision,'say_again');assert.equal(final.answers.e3_edit.edited,false);
 assert.equal(final.answers.e4_edit.decision,'edited');assert.equal(final.answers.e4_edit.text,'No peanuts please');
 assert.equal(final.answers.e5_edit.decision,'abandoned');assert.equal(final.answers.e5_edit.text,'I’m proud');assert.equal(final.answers.e5_edit.edited,true);
 for(let n=1;n<=5;n++)assert.equal(final.answers[`e${n}_action`],undefined,'No question step');
 const demoEvents=batches.flatMap(b=>b.events);
 assert(demoEvents.some(e=>e.field_id==='e1_edit'&&e.type==='keydown'&&e.key==='Backspace'&&Number.isFinite(e.elapsed_ms)));
 assert(demoEvents.some(e=>e.field_id==='e1_edit'&&e.type==='text_input'&&e.deleted==='.'));
 assert(demoEvents.some(e=>e.field_id==='e4_edit'&&e.type==='delete_all'&&e.previous_text===P));
 assert(demoEvents.some(e=>e.field_id==='e4_edit'&&e.type==='text_input'&&e.input_type==='deleteAllButton'&&e.deleted===P));
 assert(demoEvents.some(e=>e.field_id==='e4_edit'&&e.type==='transcript_reset'&&e.previous_text==='No nuts'));
 assert(demoEvents.some(e=>e.field_id==='e4_edit'&&e.type==='text_input'&&e.input_type==='resetButton'&&e.inserted===P));
 for(const [n,d] of [[2,'kept'],[3,'say_again'],[5,'abandoned']])
  assert(demoEvents.some(e=>e.field_id===`e${n}_edit`&&e.type==='decision'&&e.decision===d&&Number.isFinite(e.elapsed_ms)));
 assert.equal(final.answers.retry_count.choices[0],'Two more times');
 assert.equal(final.answers.candidates_compare.choices[0],'Typing the word myself');
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
 app=await start('s2_words',{s2_action:{status:'answered',choices:['Change parts of the text']},s2_words:{status:'answered',text:'elevator'}});
 await app.getByText('Example 2 — Asking for directions',{exact:true}).waitFor();
 assert(await app.getByLabel('Change parts of the text',{exact:true}).isChecked());
 assert.deepEqual(await app.locator('.option-group-label').allTextContents(),
  ['Continue the conversation','Use my voice again','Use text','Use another way to communicate','Stop trying']);
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText('Example 3 — Talking at home',{exact:true}).waitFor();
 await page.waitForFunction(()=>batches.length===1);
 assert.equal(await page.evaluate(()=>batches[0].state.answers.s2_words.text),'elevator');
 // Next without editing or choosing is unanswered; a session saved on the removed
 // question step resumes on its edit screen.
 const watchedOnly={demo_consent:{status:'answered',choices:['Yes']},demo_video:{status:'answered',choices:['watched']},
  edit_intro:{status:'answered',choices:['Yes, I’d like to try']}};
 app=await start('e1_action',{...watchedOnly,e1_action:{status:'answered',choices:['Change the text']}});
 await app.getByRole('heading',{name:'Example 1 — At a pharmacy'}).waitFor();
 await app.getByRole('textbox',{name:'Transcript'}).fill('changed');
 await app.getByRole('button',{name:'Reset',exact:true}).click();
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 assert.equal(await page.evaluate(()=>batches[0].state.answers.e1_edit.status),'unanswered');
 assert.equal(await page.evaluate(()=>batches[0].state.answers.e1_action.choices[0]),'Change the text','Old answers are kept');
 // The other decisions also record and move straight on.
 app=await start('e1_edit',watchedOnly);
 for(const [n,name] of [[1,'Switch to my AAC'],[2,'Ask for help'],[3,'Not sure']]){
  await app.getByText(`Example ${n} of 5`,{exact:true}).waitFor();
  await app.getByRole('button',{name,exact:true}).click();
 }
 await app.getByText('Example 4 of 5',{exact:true}).waitFor();
 await page.waitForFunction(()=>batches.length===3);
 const decided=await page.evaluate(()=>batches.at(-1).state.answers);
 assert.deepEqual([1,2,3].map(n=>decided[`e${n}_edit`].decision),['switch_aac','ask_help','not_sure']);
 // Word candidates: click a word for six suggestions (three above, three below).
 app=await start('e1_edit',watchedOnly);
 const ta=app.getByRole('textbox',{name:'Transcript'});
 const clickWord=word=>ta.evaluate((el,word)=>{const i=el.value.indexOf(word)+1;el.focus();el.setSelectionRange(i,i);el.dispatchEvent(new MouseEvent('click',{bubbles:true}));},word);
 const shown=()=>app.locator('.candidate-row:not(.hidden) .candidate').allTextContents();
 await clickWord('Mark');
 assert.deepEqual(await shown(),['Mike','Mara','Marc','Mary','Matt','Martin']);
 assert.equal(await app.locator('.candidate-row.above:not(.hidden) .candidate').count(),3);
 const above=await app.locator('.candidate-row.above').boundingBox(), below=await app.locator('.candidate-row.below').boundingBox(), word=await app.locator('.word-highlight').boundingBox();
 assert(above.y+above.height<=word.y+2 && below.y>=word.y+word.height-2,'Rows sit above and below the word');
 await app.getByRole('button',{name:'Mara',exact:true}).click();
 assert.equal(await ta.inputValue(),'I’m picking up the prescription for Mara Line.');
 assert.equal(await ta.evaluate(el=>el===el.ownerDocument.activeElement),true,'Text box keeps focus');
 assert.equal(await app.locator('.candidate-row:not(.hidden)').count(),0);
 await clickWord('Mara');
 assert.deepEqual(await shown(),['Mike','Mark','Marc','Mary','Matt','Martin'],'Original swapped into the chosen slot');
 // Typing still works, closes the suggestions, and a retyped word keeps its slot's candidates.
 await ta.evaluate(el=>{const i=el.value.indexOf('Line');el.setSelectionRange(i,i+4);});
 await ta.pressSequentially('Klien',{delay:20});
 assert.equal(await app.locator('.candidate-row:not(.hidden)').count(),0);
 await clickWord('Klien');
 assert.deepEqual(await shown(),['Lane','Lyon','Klein','Lynn','Lime','Link']);
 await app.getByRole('button',{name:'Klein',exact:true}).click();
 assert.equal(await ta.inputValue(),'I’m picking up the prescription for Mara Klein.','Punctuation kept');
 // Punctuation and apostrophes: the first word, clicked with the real mouse.
 const box=await ta.boundingBox();
 await ta.click({position:{x:26,y:Math.min(28,box.height/2)}});
 assert.deepEqual(await shown(),['I’ll','I’ve','I’d','we’re','you’re','he’s']);
 await ta.press('Escape');
 assert.equal(await app.locator('.candidate-row:not(.hidden)').count(),0);
 await app.getByRole('button',{name:'Next example',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 const cand=await page.evaluate(()=>batches[0]);
 assert.equal(cand.state.answers.e1_edit.text,'I’m picking up the prescription for Mara Klein.');
 const ce=cand.events;
 assert(ce.some(e=>e.type==='candidates_shown'&&e.position===7&&e.word==='Mark'&&e.options.length===6));
 assert(ce.some(e=>e.type==='candidate_selected'&&e.from==='Mark'&&e.to==='Mara'&&e.option_index===1&&Number.isFinite(e.elapsed_ms)));
 assert(ce.some(e=>e.type==='text_input'&&e.input_type==='candidateSelected'&&e.deleted==='k'&&e.inserted==='a'));
 assert(ce.some(e=>e.type==='candidates_closed'&&e.reason==='typing'));
 assert(ce.some(e=>e.type==='candidates_closed'&&e.reason==='escape'));
 assert(ce.some(e=>e.type==='text_input'&&e.inserted==='Klien'||e.type==='text_input'&&e.value_after?.includes('Klien')));
 // "No, skip the examples" goes straight to the closing question.
 app=await start('edit_intro',{...watchedOnly,edit_intro:undefined});
 await app.getByLabel('No, skip the examples',{exact:true}).check();
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await app.getByText(/Is there anything else you want us to know/).waitFor();
 // Stop the exercise: remaining unanswered examples are skipped; answered ones stay.
 app=await start('e2_edit',{...watchedOnly,e1_edit:{status:'answered',decision:'kept',text:'x',original:'x',edited:false},
  e3_edit:{status:'answered',decision:'say_again',text:'y',original:'y',edited:false}});
 await app.getByRole('button',{name:'Stop the exercise',exact:true}).click();
 await app.getByText(/Is there anything else you want us to know/).waitFor();
 await page.waitForFunction(()=>batches.length===1);
 const stopped=await page.evaluate(()=>batches[0]);
 assert.equal(stopped.state.answers.e1_edit.decision,'kept');
 assert.equal(stopped.state.answers.e3_edit.decision,'say_again');
 for(const n of [2,4,5])assert.deepEqual(stopped.state.answers[`e${n}_edit`],{status:'skipped',stopped:true});
 assert(stopped.events.some(e=>e.type==='exercise_stopped'&&e.field_id==='e2_edit'&&e.skipped.join()==='e2_edit,e4_edit,e5_edit'));
 // A question can name its own free-text choice (retry: "It depends on something else").
 app=await start('retry_count',{...watched,failed_repair:{status:'answered',choices:['Correct another word and use Re-check again']}});
 await app.getByRole('textbox').fill('how busy it is');
 assert(await app.getByLabel('It depends on something else',{exact:true}).isChecked());
 await app.getByLabel('One more time',{exact:true}).check();
 assert.equal(await app.getByRole('textbox').inputValue(),'','Deselecting clears its text');
 await app.getByRole('textbox').fill('the setting');
 await app.getByRole('button',{name:'Next',exact:true}).click();
 await page.waitForFunction(()=>batches.length===1);
 const retry=await page.evaluate(()=>batches[0].state.answers.retry_count);
 assert.deepEqual([retry.status,retry.choices,retry.other],['answered',['It depends on something else'],{other:'the setting'}]);
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
