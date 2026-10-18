// Public stock footage only; never read dataset frames or upload private material.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';

const base = process.env.APP_URL ?? 'http://127.0.0.1:5173';
const out = '../reports/screens/iteration-3';
mkdirSync(out, { recursive: true });
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : { channel: 'chrome' });
const capture = (page, name) => process.env.SKIP_SCREENSHOTS ? Promise.resolve() : page.screenshot({ path: `${out}/${name}.png` });
const scrollTo = (page, selector, offset = 170) => page.locator(selector).first().evaluate((el, gap) => window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - gap, behavior: 'instant' }), offset);
const settleText = page => page.evaluate(() => Promise.all(Array.from(document.querySelectorAll('[data-reveal]')).flatMap(el => el.getAnimations()).map(a => a.finished.catch(() => {}))));
try {
  for (const width of process.env.VIEWPORT_WIDTHS?.split(',').map(Number) ?? [1440, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, hasTouch: width <= 600, isMobile: width <= 600 });
    const errors = [];
    page.on('pageerror', error => { errors.push(error.message); console.error(error.message); });
    await page.addInitScript(() => {
      window.revealStarts = [];
      const animate = Element.prototype.animate;
      Element.prototype.animate = function (frames, options) {
        if (this.hasAttribute('data-reveal')) window.revealStarts.push({ text: this.textContent, delay: options.delay });
        return animate.call(this, frames, options);
      };
    });
    await page.goto(base);
    await page.locator('.eye-loader').waitFor({ state: 'detached' });
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('.site-header img, .brand-link').count(), 0);
    assert.equal(await page.locator('.partner-forte').first().getAttribute('src'), '/forte-logo.svg');
    assert.equal(await page.locator('.cinema').getAttribute('tabindex'), '0');
    assert.equal(await page.locator('.depth-ribbon button').count(), 0);
    assert.equal(await page.locator('.cinema-panel').count(), 3);
    assert.equal(await page.locator('.slide-cta').count(), 3);
    const navSize = await page.locator('.site-header nav').boundingBox();
    const styles = await page.evaluate(() => {
      const nav = getComputedStyle(document.querySelector('.site-header nav'));
      const frame = getComputedStyle(document.querySelector('.ribbon-frame'));
      const edge = getComputedStyle(document.querySelector('.depth-ribbon'), '::before');
      return { glass: nav.backdropFilter, background: nav.backgroundColor, shadow: frame.boxShadow, transition: frame.transitionProperty, edgeBlur: edge.backdropFilter, edgeMask: edge.maskImage, pointer: edge.pointerEvents, center: getComputedStyle(document.querySelector('.morph-surface')).filter };
    });
    assert.equal(styles.glass, 'none');
    assert.equal(styles.background, 'rgb(255, 255, 255)');
    assert.notEqual(styles.shadow, 'none');
    assert.equal(styles.transition, 'box-shadow');
    assert.match(styles.edgeBlur, /blur/); assert.match(styles.edgeMask, /gradient/);
    assert.equal(styles.pointer, 'none'); assert.equal(styles.center, 'none');
    // A compact hero can put About in the initial viewport. Already-visible
    // copy intentionally skips the scroll entrance animation.
    const aboutPending = await page.locator('.about-title').getAttribute('data-reveal-pending') === '';
    if (!aboutPending) assert.equal(await page.locator('.about-title').evaluate(el => getComputedStyle(el).opacity), '1');
    assert.equal(await page.evaluate(() => window.revealStarts.some(r => r.text.includes('За кадром'))), false);
    assert.match(await page.locator('body').evaluate(el => getComputedStyle(el).backgroundImage), /radial-gradient/);
    assert.equal(await page.locator('#about').evaluate(el => getComputedStyle(el).backgroundImage), 'none');
    const aboutBox = await page.locator('#about').boundingBox();
    assert.equal(aboutBox.x, 0); assert.equal(aboutBox.width, width);
    const copyBefore = await page.locator('.hero-intro p').boundingBox();
    await page.waitForFunction(() => document.querySelector('.hero-title').dataset.animating === 'true');
    assert.equal(await page.locator('.slogan-pause').count(), 0);
    assert.match(await page.locator('.hero-title').evaluate(el => getComputedStyle(el).fontFamily), /Google Sans/);
    assert.equal(await page.locator('.hero-title').evaluate(el => getComputedStyle(el).fontWeight), '400');
    assert.equal(await page.locator('.slogan-letter').first().evaluate(el => getComputedStyle(el).animationPlayState), 'running');
    assert.deepEqual(await page.locator('.hero-intro p').boundingBox(), copyBefore);
    const titleBox = await page.locator('.hero-title').boundingBox();
    assert.ok(titleBox.x >= 0 && titleBox.x + titleBox.width <= width);
    await page.mouse.move(0, 0);
    assert.equal(await page.locator('.ribbon-frame img').evaluateAll(images => new Set(images.map(image => image.src)).size), 8);
    await page.waitForFunction(() => document.querySelector('.partner-NU').naturalWidth > 0);
    const logo = await page.locator('.partner-NU').first().evaluate(el => {
      const box = el.getBoundingClientRect(), slot = el.parentElement.getBoundingClientRect();
      return { ratio: box.width / box.height, naturalRatio: el.naturalWidth / el.naturalHeight, paddingTop: box.top - slot.top, paddingBottom: slot.bottom - box.bottom, fitsX: box.left >= slot.left && box.right <= slot.right, fit: getComputedStyle(el).objectFit };
    });
    assert.ok(Math.abs(logo.ratio - logo.naturalRatio) < .002);
    assert.ok(logo.paddingTop >= 8 && logo.paddingBottom >= 8 && logo.fitsX);
    assert.equal(logo.fit, 'contain');
    await capture(page, `${width}-hero`);

    // Scroll into text without anchor focus, and observe the actual in-flight reveal.
    await scrollTo(page, '.about-intro');
    if (aboutPending) {
      await page.waitForFunction(() => document.querySelector('.about-title').getAnimations().length > 0);
      const inFlight = await page.locator('.about-title').evaluate(el => ({ opacity: Number(getComputedStyle(el).opacity), transform: getComputedStyle(el).transform }));
      assert.ok(inFlight.opacity < 1); assert.notEqual(inFlight.transform, 'none');
    }
    await settleText(page);
    assert.equal(await page.locator('.about-title').evaluate(el => getComputedStyle(el).opacity), '1');
    await capture(page, `${width}-about`);
    await scrollTo(page, '.team-row');
    await page.waitForFunction(() => !document.querySelector('.team-row h3').hasAttribute('data-reveal-pending'));
    await settleText(page);
    const githubLinks = await page.locator('.team-row h3 a').evaluateAll(links => links.map(link => ({ href: link.href, target: link.target, rel: link.rel, label: link.getAttribute('aria-label') })));
    assert.deepEqual(githubLinks.map(link => link.href), ['https://github.com/a100miles', 'https://github.com/M1kado0', 'https://github.com/ytyourstruly', 'https://github.com/trulyaldi']);
    assert.ok(githubLinks.every(link => link.target === '_blank' && link.rel.includes('noopener') && link.label?.includes('GitHub')));
    const teamNameType = await page.locator('.team-row h3').first().evaluate(el => ({ family: getComputedStyle(el).fontFamily, weight: getComputedStyle(el).fontWeight, style: getComputedStyle(el).fontStyle }));
    assert.match(teamNameType.family, /Handjet/); assert.equal(teamNameType.weight, '500'); assert.equal(teamNameType.style, 'normal');
    await page.locator('.team-row h3 a').first().focus();
    assert.equal(await page.locator('.about-section').evaluate(el => getComputedStyle(el, '::before').content), 'none');
    await page.locator('.team-row h3 a').first().evaluate(link => link.blur());
    assert.equal(await page.locator('.about-particle').count(), 12);
    assert.equal(await page.locator('.about-particles').evaluate(el => getComputedStyle(el).pointerEvents), 'none');
    assert.equal(await page.locator('.about-particles').evaluate(el => getComputedStyle(el).opacity), '1');
    if (!await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)) {
      assert.equal(await page.locator('.about-particle').first().evaluate(el => getComputedStyle(el).animationName), 'about-particle-float');
    }
    if (await page.evaluate(() => matchMedia('(hover: hover) and (pointer: fine)').matches)) {
      const row = page.locator('.team-row').first();
      await row.hover();
      await page.waitForFunction(() => Math.abs(Number(getComputedStyle(document.querySelector('.team-row')).scale) - 1.018) < .001);
      const particleStyle = await page.locator('.about-particle').first().evaluate(el => ({ radius: getComputedStyle(el).borderRadius, timing: getComputedStyle(el).animationTimingFunction, shadow: getComputedStyle(el).boxShadow }));
      assert.equal(particleStyle.radius, '0px'); assert.match(particleStyle.timing, /steps\(9/); assert.match(particleStyle.shadow, /inset/);
      await capture(page, `${width}-team-hover`);
      await page.locator('.site-header nav').hover();
      await page.waitForFunction(() => Number(getComputedStyle(document.querySelector('.team-row')).scale) === 1);
    }
    await capture(page, `${width}-team`);
    await scrollTo(page, '.about-intro');
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => window.revealStarts.filter(r => r.text.includes('За кадром')).length), aboutPending ? 1 : 0);
    assert.equal(await page.locator('.about-title').evaluate(el => el.getAnimations().filter(animation => animation.playState === 'running').length), 0);

    // All About copy and the solid navigation retain accessible contrast.
    const contrast = await page.evaluate(() => {
      const rgb = value => value.match(/[\d.]+/g).map(Number);
      const luminance = channels => channels.slice(0, 3).map(c => c / 255).map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4).reduce((v, c, i) => v + c * [.2126, .7152, .0722][i], 0);
      const ratio = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
      const background = rgb(getComputedStyle(document.querySelector('#about')).backgroundColor);
      const text = ['.about-title', '.about-title em', '.about-copy p', '.about-note', '.team-row h3', '.team-role', '.team-row p', '.project-principle span', '.project-principle p'].map(selector => ratio(rgb(getComputedStyle(document.querySelector(selector)).color), background));
      const navBackground = rgb(getComputedStyle(document.querySelector('.site-header nav')).backgroundColor);
      const nav = ratio(rgb(getComputedStyle(document.querySelector('.site-header nav a')).color), navBackground);
      return { text, nav };
    });
    assert.ok(contrast.text.every(value => value >= 4.5)); assert.ok(contrast.nav >= 4.5);
    await scrollTo(page, '.about-intro', -20);
    await capture(page, `${width}-navigation`);
    const nextNavSize = await page.locator('.site-header nav').boundingBox();
    assert.equal(navSize.width, nextNavSize.width); assert.equal(navSize.height, nextNavSize.height);

    await page.locator('.site-header .nav-cta').click();
    await page.waitForFunction(() => document.querySelector('.cinema').dataset.moving === 'false');
    assert.equal(await page.locator('.upload-target').isVisible(), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    const sponsorNames = await page.locator('.partners img[alt]:not([alt=""])').evaluateAll(images => images.map(image => image.alt));
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.waitForFunction(() => document.querySelectorAll('[data-reveal-pending]').length === 0);
    await page.reload();
    assert.equal(await page.locator('[data-reveal-pending]').count(), 0);
    assert.notEqual(await page.locator('.about-particles').evaluate(el => getComputedStyle(el).display), 'none');
    assert.equal(await page.locator('.about-particles').evaluate(el => getComputedStyle(el).opacity), '1');
    assert.equal(await page.locator('.about-particle').first().evaluate(el => getComputedStyle(el).animationName), 'none');
    assert.equal(await page.evaluate(() => window.revealStarts.length), 0);
    assert.deepEqual(await page.locator('.partner-slot:visible img').evaluateAll(images => images.map(image => image.alt)), sponsorNames);
    await scrollTo(page, '.partners', 280);
    await capture(page, `${width}-logos-reduced`);
    await page.locator('.hero-title').hover();
    assert.equal(await page.locator('.hero-title').evaluate(el => getComputedStyle(el).scale), 'none');
    assert.equal(await page.locator('.slogan-letter').first().evaluate(el => getComputedStyle(el).animationName), 'none');
    assert.equal(await page.locator('.slogan-pause').count(), 0);
    assert.deepEqual(errors, []);
    console.log(`${width}px: dots, NU proportions, Google Sans 400, letter wave without pause button, eight images, white navigation, full dark About, reveals, reduced motion, no overflow — PASS (minimum About contrast ${Math.min(...contrast.text).toFixed(2)}:1)`);
    await page.close();
  }

  for (const failure of ['missing', 'constructor']) {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(mode => {
      if (mode === 'missing') delete window.IntersectionObserver;
      else window.IntersectionObserver = class { constructor() { throw new Error('Simulated observer initialization failure'); } };
    }, failure);
    await page.goto(base); await page.locator('.eye-loader').waitFor({ state: 'detached' });
    assert.equal(await page.locator('[data-reveal-pending]').count(), 0);
    assert.equal(await page.locator('.about-title').evaluate(el => getComputedStyle(el).opacity), '1');
    await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();
    await page.waitForFunction(() => document.querySelector('.cinema').dataset.moving === 'false');
    assert.equal(await page.locator('.upload-target').isVisible(), true);
    assert.deepEqual(errors, []);
    console.log(`Observer ${failure}: visible content and working upload — PASS`);
    await page.close();
  }

  // Emulate ignoring the enhancement rules, as an engine without backdrop-filter would.
  const fallback = await browser.newPage();
  await fallback.goto(base);
  await fallback.locator('.site-header nav').waitFor();
  const solid = await fallback.evaluate(() => {
    const strip = sheet => {
      for (let i = sheet.cssRules.length - 1; i >= 0; i--) {
        const rule = sheet.cssRules[i];
        if (rule instanceof CSSSupportsRule && rule.conditionText.includes('backdrop-filter')) sheet.deleteRule(i);
        else if (rule.cssRules) strip(rule);
      }
    };
    Array.from(document.styleSheets).forEach(strip);
    const nav = getComputedStyle(document.querySelector('.site-header nav'));
    return { blur: nav.backdropFilter, background: nav.backgroundColor, edge: getComputedStyle(document.querySelector('.depth-ribbon'), '::before').content };
  });
  assert.equal(solid.blur, 'none'); assert.equal(solid.background, 'rgb(255, 255, 255)'); assert.equal(solid.edge, 'none');
  console.log('No-backdrop CSS fallback: solid navigation, unblurred carousel — PASS');
  await fallback.close();
} finally { await browser.close(); }
