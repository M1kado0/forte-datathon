// Browser-only fixtures: generated geometric JPEGs, never dataset material.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';

const base = process.env.APP_URL ?? 'http://127.0.0.1:8000';
const browser = await chromium.launch(process.env.CHROMIUM_PATH
  ? { executablePath: process.env.CHROMIUM_PATH } : { channel: 'chrome' });
const out = '../reports/screens/result-article';
if (process.env.CAPTURE) mkdirSync(out, { recursive: true });
const rule = {
  id: 'synthetic', name: 'Синтетический пример: пересчёт наличных',
  citation_suffix: 'Тестовый документ, тестовое утверждение, п. 42',
  rules: [{ title: 'Тестовый документ для проверки интерфейса', approved_by: 'Тестовое утверждение',
    punkt: 'п. 42', citation: 'Тестовый документ, тестовое утверждение, п. 42', url: 'https://example.com/source' }],
};

async function setup(width) {
  const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const fixture = { rules: [rule], verdict: 'violation', violations: [] };
  await page.route('**/api/ui/rules', route => route.fulfill({ json: fixture.rules }));
  await page.route('**/api/ui/analyze', route => route.fulfill({ json: { job_id: 'fixture' } }));
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ json: {
    job_id: 'fixture', status: 'done', verdict: fixture.verdict, violations: fixture.violations,
  } }));
  await page.goto(base + '/#analysis');
  const frames = await page.evaluate(() => [40, 42.5, 44].map((timestamp_sec, index) => {
    const canvas = document.createElement('canvas'); canvas.width = 1600; canvas.height = 1000;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = ['#ebe4e8', '#e0d9dd', '#f3e5e0'][index]; ctx.fillRect(0, 0, 1600, 1000);
    ctx.strokeStyle = '#a31551'; ctx.lineWidth = 5; ctx.strokeRect(150 + index * 180, 160, 600, 420);
    ctx.fillStyle = '#4a0221'; ctx.font = '40px sans-serif';
    ctx.fillText(`Синтетический кадр ${index + 1}`, 140, 760);
    ctx.font = '24px sans-serif'; ctx.fillText('Изображение только для проверки интерфейса', 140, 815);
    return { timestamp_sec, image_base64: canvas.toDataURL('image/jpeg').split(',')[1] };
  }));
  fixture.violations = [{ type: 'synthetic', timecode_sec: 42.5,
    description: `Синтетическое наблюдение: проверяем расположение текста, кадров и нормативного источника. ${rule.citation_suffix}`,
    evidence_frames: frames }];
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic.mov', mimeType: 'video/quicktime', buffer: Buffer.from('synthetic') });
  const submit = async () => {
    await page.locator('button[type=submit]').click();
    await page.locator('.result-summary').waitFor();
  };
  await submit();
  return { page, fixture, submit, errors, frames };
}

try {
  for (const width of [1440, 390]) {
    const { page, fixture, submit, errors, frames } = await setup(width);
    const article = page.locator('.result-article').first();
    const main = article.locator('.evidence-open');
    const thumbs = article.locator('.evidence-thumbnails button');
    assert.equal(await thumbs.count(), 3);
    assert.equal(await thumbs.nth(1).getAttribute('aria-pressed'), 'true');
    assert.match(await article.locator('.result-timecode').innerText(), /00:42,5/);
    assert.equal(await article.locator('.result-description').innerText().then(text => text.includes(rule.citation_suffix)), false);
    assert.equal(await article.locator('.source-document').innerText(), rule.rules[0].title);
    assert.equal(await article.locator('.source-point').innerText(), 'п. 42');
    assert.equal(await article.getByRole('link').getAttribute('href'), rule.rules[0].url);
    assert.equal(await article.locator('h3').evaluate(el => getComputedStyle(el).fontWeight), '500');
    assert.match(await article.locator('h3').evaluate(el => getComputedStyle(el).fontFamily), /Handjet/);
    assert.match(await article.locator('.result-description').evaluate(el => getComputedStyle(el).fontFamily), /Open Sans/);
    assert.match(await article.locator('.source-document').evaluate(el => getComputedStyle(el).fontFamily), /Open Sans/);
    assert.equal(await article.locator('.result-particles .about-particle').count(), 12);
    assert.equal(await article.locator('.result-particles').getAttribute('aria-hidden'), 'true');
    assert.equal(await article.locator('.result-particles').evaluate(el => getComputedStyle(el).pointerEvents), 'none');
    await article.scrollIntoViewIfNeeded();
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.waitForFunction(() => document.querySelector('.result-article .result-particles').dataset.paused === 'false');
    assert.equal(await article.locator('.about-particle').first().evaluate(el => getComputedStyle(el).animationName), 'about-particle-float');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await article.locator('.about-particle').first().evaluate(el => getComputedStyle(el).animationName), 'none');
    const contrast = await article.evaluate(el => {
      const luminance = color => color.match(/[\d.]+/g).slice(0, 3).map(Number).map(c => c / 255)
        .map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4)
        .reduce((sum, c, index) => sum + c * [.2126, .7152, .0722][index], 0);
      const base = getComputedStyle(el).backgroundColor;
      return ['h3', '.result-description', '.result-detail h4', '.source-document', '.source-authority', '.source-point', '.result-sources a', '.result-timecode', '.evidence-open-label', 'figcaption', '.evidence-thumbnails button[aria-pressed="true"]'].map(selector => {
        const style = getComputedStyle(el.querySelector(selector));
        const background = luminance(style.backgroundColor === 'rgba(0, 0, 0, 0)' ? base : style.backgroundColor);
        const foreground = luminance(style.color);
        return (Math.max(background, foreground) + .05) / (Math.min(background, foreground) + .05);
      });
    });
    assert.ok(contrast.every(ratio => ratio >= 4.5), `Text contrast: ${contrast.join(', ')}`);
    const surfaceBox = await page.locator('.morph-surface').boundingBox();
    const articleBox = await article.boundingBox();
    assert.ok(Math.abs(surfaceBox.width - articleBox.width) < 1);
    assert.ok(Math.abs(surfaceBox.y + surfaceBox.height - articleBox.y - articleBox.height) < 1);
    const geometry = await article.evaluate(el => {
      const evidence = el.querySelector('.result-evidence').getBoundingClientRect();
      const text = el.querySelector('.result-observation').getBoundingClientRect();
      return { evidenceWidth: evidence.width, textWidth: text.width, evidenceBottom: evidence.bottom, textTop: text.top };
    });
    if (width > 800) assert.ok(geometry.evidenceWidth > geometry.textWidth);
    else assert.ok(geometry.evidenceBottom <= geometry.textTop);
    if (process.env.CAPTURE) await article.screenshot({ path: `${out}/${width}-article.png` });

    await thumbs.nth(2).click();
    await page.waitForFunction(() => document.querySelector('.evidence-open img').alt.includes('44 с'));
    await article.locator('.result-timecode').click();
    await page.waitForFunction(() => document.querySelector('.evidence-open img').alt.includes('42.5 с'));
    assert.equal(await main.evaluate(el => el === document.activeElement), true);
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog');
    await dialog.waitFor();
    assert.equal(await dialog.evaluate(el => el.contains(document.activeElement)), true);
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('dialog h2').textContent.includes('Кадр 3'));
    assert.equal(await dialog.getByRole('button', { name: 'Следующий кадр', exact: true }).isDisabled(), true);
    await page.keyboard.press('ArrowLeft');
    await dialog.getByRole('button', { name: 'Исходный размер', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.evidence-viewer').dataset.zoomed === 'true');
    assert.ok(await dialog.locator('.evidence-viewer').evaluate(el => el.scrollWidth > el.clientWidth));
    await dialog.getByRole('button', { name: 'Вписать в окно', exact: true }).click();
    if (process.env.CAPTURE) await page.screenshot({ path: `${out}/${width}-viewer.png` });
    // Native modal keeps keyboard focus inside, even when cycling past the last control.
    for (let i = 0; i < 7; i++) { await page.keyboard.press('Tab'); assert.equal(await dialog.evaluate(el => el.contains(document.activeElement)), true); }
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await main.evaluate(el => el === document.activeElement), true);
    assert.equal(await page.locator('.cinema').getAttribute('data-expanded'), 'true');
    assert.notEqual(await page.evaluate(() => document.body.style.overflow), 'hidden');
    await main.click(); await dialog.waitFor();
    await dialog.getByRole('button', { name: 'Закрыть просмотр кадра' }).click();
    await dialog.waitFor({ state: 'detached' });

    // Independent event selection, a single-frame gallery and a missing-frame fallback.
    fixture.violations.push({ ...fixture.violations[0], evidence_frames: [frames[0]] });
    await submit();
    assert.equal(await page.locator('.result-article').count(), 2);
    assert.equal(await page.locator('.result-article').nth(1).locator('.evidence-thumbnails').count(), 0);
    await page.locator('.result-article').first().locator('.evidence-thumbnails button').nth(2).click();
    assert.match(await page.locator('.result-article').nth(1).locator('.evidence-open img').getAttribute('alt'), /40 с/);
    fixture.violations = [{ ...fixture.violations[0], timecode_sec: 59.96, evidence_frames: [] }];
    await submit();
    assert.match(await page.locator('.result-timecode').innerText(), /01:00,0/);
    assert.equal(await page.locator('button.result-timecode').count(), 0);
    assert.equal(await page.locator('.evidence-empty').isVisible(), true);
    fixture.violations[0].evidence_frames = [{ timestamp_sec: 0, image_base64: 'invalid' }];
    await submit(); await page.locator('.evidence-image-error').waitFor();
    fixture.verdict = 'no_violation'; fixture.violations = [];
    await submit(); assert.equal(await page.locator('.result-article').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(`${width}px: gallery, nearest timecode, sources, viewer/zoom/keyboard/focus, responsive order, multiple/missing/broken frames, clean result — PASS`);
    await page.close();
  }
} finally { await browser.close(); }
