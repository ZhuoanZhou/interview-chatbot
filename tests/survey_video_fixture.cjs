/* Generate a tiny playable test clip locally; never fetch the study video. */
module.exports = async function videoFixture(browser) {
 const page = await browser.newPage();
 try {
  const fixture = await page.evaluate(async () => {
   const canvas = document.createElement('canvas'); canvas.width=160; canvas.height=90;
   const ctx=canvas.getContext('2d');
   const stream=canvas.captureStream(10);
   const type=MediaRecorder.isTypeSupported('video/mp4')?'video/mp4':'video/webm';
   const recorder=new MediaRecorder(stream,{mimeType:type});const chunks=[];
   recorder.ondataavailable=e=>chunks.push(e.data);
   const stopped=new Promise(resolve=>recorder.onstop=resolve);
   recorder.start();
   const draw=setInterval(()=>{ctx.fillStyle='teal';ctx.fillRect(0,0,160,90)},50);
   await new Promise(resolve=>setTimeout(resolve,600));recorder.stop();await stopped;
   clearInterval(draw);stream.getTracks().forEach(track=>track.stop());
   return {type,bytes:Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()))};
  });
  return {type:fixture.type,bytes:Buffer.from(fixture.bytes)};
 } finally {await page.close()}
};
