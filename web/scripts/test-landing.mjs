import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import {mkdirSync} from 'node:fs';
const base=process.env.APP_URL ?? 'http://127.0.0.1:5173';
const out='../reports/screens/light-confirm';mkdirSync(out,{recursive:true});
const browser=await chromium.launch(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {channel:'chrome'});
try {
 for(const width of [1440,768,390,320]) {
  const page=await browser.newPage({viewport:{width,height:900},hasTouch:width===390}); const errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text()+' '+m.location().url);});
  await page.goto(base);await page.locator('.eye-loader').waitFor({state:'detached'});await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>{const v=document.querySelector('.cinema-panel[data-position="active"] video');return v && v.videoWidth>0 && !v.paused;});
  await page.screenshot({path:`${out}/${width}-hero.png`});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.equal(await page.locator('.cinema-controls, .partners-caption').count(),0);
  assert.equal(await page.getByRole('button', {name:'Приостановить анимацию'}).count(),1);
  assert.equal(await page.locator('.partners').count(),1);
  assert.equal(await page.locator('#home .carousel-meta + .partners').count(),1);
  for(const selector of ['#hero-title','.about-title']) {
    const font=await page.locator(selector).first().evaluate(el=>({family:getComputedStyle(el).fontFamily,weight:getComputedStyle(el).fontWeight}));
    assert.match(font.family,/Playfair Display/);assert.equal(font.weight,'400');
  }
  const teamFont=await page.locator('.team-row h3').first().evaluate(el=>({family:getComputedStyle(el).fontFamily,weight:getComputedStyle(el).fontWeight}));
  assert.match(teamFont.family,/Handjet/);assert.equal(teamFont.weight,'500');
  for(const selector of ['.site-header nav a','.hero-intro p','.team-role','.team-row p','.slide-cta']) {
    assert.match(await page.locator(selector).first().evaluate(el=>getComputedStyle(el).fontFamily),/Open Sans/);
  }
  assert.equal(await page.locator('.scroll-depth').evaluate(el=>getComputedStyle(el).opacity),'0');
  // Slow down/up and rapid reversals: sample actual scroll-linked values each frame.
  const samples=await page.evaluate(async()=>{
    const values=[];
    for(const y of [0,20,40,60,100,180,360,100,20,0]) {
      window.scrollTo({top:y,behavior:'instant'});
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      values.push({y:scrollY,depth:Number(document.querySelector('.scroll-depth').style.getPropertyValue('--scroll-depth'))});
    }
    return values;
  });
  assert.ok(samples[1].depth>0 && samples[1].depth<samples[2].depth);
  assert.equal(samples[5].depth,1);assert.equal(samples.at(-1).depth,0);
  await page.evaluate(()=>window.scrollTo({top:100,behavior:'instant'}));
  await page.waitForFunction(()=>Number(document.querySelector('.scroll-depth').style.getPropertyValue('--scroll-depth'))>0);
  await page.screenshot({path:`${out}/${width}-scroll.png`});
  assert.equal(await page.locator('.scroll-depth').evaluate(el=>getComputedStyle(el).pointerEvents),'none');
  assert.equal(await page.locator('.site-header').evaluate(el=>getComputedStyle(el).filter),'none');
  await page.locator('#analysis').press('ArrowRight');
  await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video')?.getAttribute('src')==='/demo/team.webm');
  await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video')?.videoWidth>0);
  await page.waitForTimeout(700); // Existing slide transition lock is 650 ms.
  await page.locator('#analysis').press('ArrowRight');
  assert.equal(await page.locator('.cinema-panel[data-position="active"] video').getAttribute('src'),'/demo/payment.webm');
  if(width===390) {
    await page.waitForTimeout(700);
    await page.locator('.cinema-stage').scrollIntoViewIfNeeded();
    const box=await page.locator('.cinema-stage').boundingBox();
    const cdp=await page.context().newCDPSession(page);
    const y=box.y+box.height/2;
    await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:290,y}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:210,y}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:100,y}]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video')?.getAttribute('src')==='/demo/counting.webm');
    await cdp.detach();
  }
  await page.locator('.site-header a[href="#analysis"]').click();
  await page.waitForFunction(()=>Math.abs(document.querySelector('#analysis').getBoundingClientRect().top-(parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height'))+24))<3);
  await page.waitForFunction(()=>document.querySelector('.cinema-panel[data-position="active"] video')?.paused);
  await page.screenshot({path:`${out}/${width}-analysis.png`});
  await page.locator('.site-header a[href="#about"]').click();
  await page.waitForFunction(()=>document.querySelector('.site-header a[href="#about"]').getAttribute('aria-current')==='location');
  await page.waitForFunction(()=>Math.abs(document.querySelector('#about').getBoundingClientRect().top-(parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height'))+24))<3);
  await page.evaluate(()=>Promise.all(document.getAnimations().filter(a=>a.effect?.getTiming().iterations!==Infinity).map(a=>a.finished)));
  await page.screenshot({path:`${out}/${width}-about.png`});
  assert.equal(await page.locator('.team-row').count(),4);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.emulateMedia({reducedMotion:'reduce'});await page.goto(base);
  await page.locator('.cinema-panel[data-position="active"] video').waitFor();
  assert.equal(await page.locator('.eye-loader:visible').count(),0);
  assert.equal(await page.locator('.cinema-panel[data-position="active"] video').evaluate(v=>v.paused),true);
  assert.equal(await page.locator('.marquee-track').evaluate(e=>getComputedStyle(e).animationName),'none');
  assert.equal(await page.locator('.scroll-depth').evaluate(e=>getComputedStyle(e).display),'none');
  assert.deepEqual(errors,[]);
  console.log(`${width}px: carousel, anchors, marquee placement, Playfair scope, bidirectional scroll blur, reduced motion, overflow, console — PASS`);
  await page.close();
 }
} finally {await browser.close();}

