/* Run against the local SURVEY_PREVIEW server on port 8511. Uses fake data only. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const b=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const p=await b.newPage({viewport:{width:1280,height:900}});
  await p.goto('http://127.0.0.1:8511');
  await p.getByText('Preview only — uses local test files and makes no Google Drive or AI calls.',{exact:true}).waitFor();
  fs.mkdirSync('tmp/survey-qa',{recursive:true});
  await p.getByRole('button',{name:'Start a new survey',exact:true}).click();
  const frame=p.locator('iframe[title*="fixed_communication_survey"]');
  const app=p.frameLocator('iframe[title*="fixed_communication_survey"]');
  await app.getByRole('button',{name:'Start survey',exact:true}).click();
  await app.getByLabel('Other',{exact:true}).check();
  await app.getByRole('textbox').fill('A sample answer');
  for(const size of [{width:1280,height:900},{width:390,height:844},{width:844,height:390},{width:390,height:420}]){
   await p.setViewportSize(size);await p.waitForTimeout(400);
   const box=await frame.boundingBox();
   assert(box.y>=0 && box.y+box.height<=size.height+1,JSON.stringify({size,box}));
   const controls=await app.locator('#controls').boundingBox();
   const question=await app.locator('#page').boundingBox();
   assert(question.height>40);
   assert(question.y+question.height<=controls.y+1,'Navigation must not overlap answers');
   for(const name of ['Next','Back','Skip','Save and take a break','End my survey now']){
    const button=await app.getByRole('button',{name,exact:true}).boundingBox();
    assert(button.y>=0&&button.y+button.height<=size.height+1,`${name} outside ${JSON.stringify(size)}`);
   }
   await app.locator('#page').evaluate(el=>el.scrollTop=el.scrollHeight);
   const after=await app.locator('#controls').boundingBox();
   assert(Math.abs(after.y-controls.y)<1,'Question scrolling must not move navigation');
   assert(await app.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await p.screenshot({path:`tmp/survey-qa/fixed-nav-${size.width}-${size.height}.png`,fullPage:true});
  }
  await app.getByRole('button',{name:'Next',exact:true}).click();
  await p.waitForTimeout(100);
  assert.equal(await app.locator('#page').evaluate(el=>el.scrollTop),0);
  console.log('PASS: all five actions stay inside desktop/mobile/short viewports; question scroll does not move controls or overlap answers; navigation resets scroll.');
 }finally{await b.close()}
})().catch(e=>{console.error(e);process.exit(1)});
