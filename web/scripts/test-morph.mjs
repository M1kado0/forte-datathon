// Browser interactions use synthetic files; never read or upload dataset material.
// Successful results come from the running API. Its stub status is reported explicitly.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const base=process.env.APP_URL ?? 'http://127.0.0.1:5173';
const browser=await chromium.launch(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {channel:'chrome'});
const settle=page=>page.waitForFunction(()=>document.querySelector('.cinema').dataset.moving==='false');
try {
 for (const width of process.env.API_ONLY ? [] : process.env.VIEWPORT_WIDTHS?.split(',').map(Number) ?? [1440,1280,768,390]) {
  const page=await browser.newPage({viewport:{width,height:900},hasTouch:width===390});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const fontRequests=[];page.on('request',r=>{if(/Hina/i.test(r.url()))fontRequests.push(r.url());});
  await page.goto(base);await page.locator('.eye-loader').waitFor({state:'detached'});
  await page.evaluate(()=>document.fonts.ready);
  assert.match(await page.locator('.site-header nav a').first().evaluate(el=>getComputedStyle(el).fontFamily),/Open Sans/);
  assert.equal(await page.locator('.site-header').innerText().then(t=>t.includes('Datathon')),false);
  assert.equal(await page.locator('.upload-target').isVisible(),false);
  // Track the actual element throughout the morph (no duplicated upload state).
  const surface=await page.locator('.morph-surface').elementHandle();
  const form=await page.locator('.check-form').elementHandle();
  const before=await surface.boundingBox();
  await page.locator('.cinema-panel[data-position="active"] .slide-cta').focus();
  const scrollBefore=await page.evaluate(()=>scrollY);
  await page.keyboard.press('Enter');
  await page.waitForFunction(()=>document.querySelector('.cinema').dataset.moving==='true');
  assert.equal(await page.locator('.upload-panel').evaluate(el=>el.inert),true);
  await settle(page);
  assert.equal(await page.evaluate(()=>document.activeElement.className),'upload-target');
  assert.equal(await page.evaluate(()=>scrollY),scrollBefore);
  assert.equal(await surface.evaluate(el=>el===document.querySelector('.morph-surface')),true);
  assert.ok((await surface.boundingBox()).width>before.width);
  const input=page.locator('input[type=file]');
  const submit=page.locator('button[type=submit]');
  await input.setInputFiles({name:'invalid.mp4',mimeType:'video/mp4',buffer:Buffer.from('synthetic')});
  assert.equal(await submit.isDisabled(),true);
  assert.match(await page.locator('.upload-error').innerText(),/только .mov/);
  await input.setInputFiles({name:'retained.MOV',mimeType:'application/octet-stream',buffer:Buffer.from('synthetic')});
  assert.equal(await submit.isEnabled(),true);
  await page.locator('.back-to-carousel').click();await settle(page);
  assert.equal(await page.evaluate(()=>document.activeElement.classList.contains('slide-cta')),true);
  assert.equal(await page.locator('.upload-target').isVisible(),false);
  await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();await settle(page);
  assert.equal(await form.evaluate(el=>el===document.querySelector('.check-form')),true);
  assert.equal(await page.locator('.file-info p').innerText(),'retained.MOV');
  assert.equal(await page.locator('input[type=file]').count(),1);
  await page.keyboard.press('Escape');await settle(page);
  assert.equal(await page.locator('.cinema').getAttribute('data-expanded'),'false');
  if(width===390) {
   const box=await page.locator('.cinema-stage').boundingBox();
   const cdp=await page.context().newCDPSession(page);const y=box.y+box.height/2;
   await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:290,y}]});
   await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:190,y}]});
   await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:90,y}]});
   await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
   await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video').getAttribute('src')==='/demo/team.webm');
   await cdp.detach();
  }
  await page.locator('.site-header a[href="#analysis"]').click();await settle(page);
  await page.waitForFunction(()=>Math.abs(document.querySelector('#analysis').getBoundingClientRect().top-(parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height'))+24))<3).catch(async error => { console.log('Anchor failure', width, await page.evaluate(()=>({y:scrollY,top:document.querySelector('#analysis').getBoundingClientRect().top,hash:location.hash,focus:document.activeElement.className}))); throw error; });
  assert.equal(await page.locator('.file-info p').innerText(),'retained.MOV');
  await page.locator('.brand-link').click();await settle(page);
  await page.emulateMedia({reducedMotion:'reduce'});await page.goto(base+'/#analysis');await settle(page);
  assert.equal(await page.locator('.eye-loader:visible').count(),0);
  assert.equal(await page.locator('.upload-target').isVisible(),true);
  await page.locator('.back-to-carousel').click();await settle(page);
  assert.equal(await page.locator('.cinema-panel[data-position="active"] video').evaluate(v=>v.paused),true);
  assert.equal(await page.locator('.cinema-controls').count(),0);
  // Hash navigation preserves the selected scene; return with the retained keyboard control.
  if (await page.locator('.cinema-panel[data-position="active"] video').getAttribute('src') !== '/demo/counting.webm') {
    await page.locator('#analysis').press('ArrowLeft');
    await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video').getAttribute('src')==='/demo/counting.webm');
  }
  await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();await settle(page);
  assert.equal(await page.locator('.upload-panel').evaluate(el=>el.inert),false);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.deepEqual(fontRequests,[]);assert.deepEqual(errors,[]);
  console.log(width+'px: morph continuity, no scroll, keyboard focus, file persistence, validation, return/Escape, anchors, reduced motion, no Hina, overflow — PASS');
  await page.close();
 }
 const page=await browser.newPage({viewport:{width:1280,height:900},reducedMotion:'reduce'});
 await page.goto(base+'/#analysis');await settle(page);
 const health=await (await page.request.get(base+'/api/health')).json();
 console.log('Backend: '+JSON.stringify(health));
 await page.locator('input[type=file]').setInputFiles({name:'api-flow.mov',mimeType:'video/quicktime',buffer:Buffer.from('synthetic API transport check')});
 const post=page.waitForResponse(r=>r.url().endsWith('/api/ui/analyze')&&r.request().method()==='POST');
 await page.locator('button[type=submit]').click();
 const response=await post;assert.equal(response.status(),200);
 // Chromium omits native file bytes from Request.postData(). A 200 from this
 // endpoint also proves that its required multipart "file" field was supplied.
 assert.match(response.request().headers()['content-type'],/multipart\/form-data; boundary=/);
 const {job_id}=await response.json();assert.ok(job_id);
 await page.waitForFunction(()=>document.querySelector('.results-section h2'));
 assert.equal(await page.locator('.demo-notice').isVisible(),health.model_loaded === false);
 assert.equal(await page.locator('.results-section').getAttribute('aria-busy'),'false');
 await page.locator('.back-to-carousel').click();await settle(page);
 await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();await settle(page);
 assert.equal(await page.locator('.results-section h2').count(),1);
 // Network failures only: never fabricate a successful analysis result.
 await page.context().setOffline(true);
 await page.locator('button[type=submit]').click();
 await page.getByRole('alert').waitFor();
 assert.match(await page.getByRole('alert').innerText(),/не ответил|соединение/);
 assert.equal(await page.locator('.file-info p').innerText(),'api-flow.mov');
 await page.context().setOffline(false);
 await page.getByRole('alert').getByRole('button').click();
 await page.locator('.results-section h2').waitFor();
 console.log('Actual local API: multipart upload, polling, result preservation, offline error/retry — PASS (model accuracy not tested)');
 await page.close();
}finally{await browser.close();}

