// Manual browser regression checks. All API responses and evidence are synthetic.
// Never read dataset material or assert model quality here.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const base = process.env.APP_URL ?? "http://127.0.0.1:5173";
const browser = await chromium.launch(process.env.CHROMIUM_PATH
  ? { executablePath: process.env.CHROMIUM_PATH } : { channel: "chrome" });
let failures = 0;
const rule = { id: "test_type", name: "Синтетический тип", citation_suffix: "", rules: [] };
async function check(name, run, options = {}) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce", ...options });
  page.setDefaultTimeout(2500);
  await page.route("**/api/health", route => route.fulfill({ json: { status: "ok", version: "test", model_loaded: true } }));
  await page.route("**/api/ui/rules", route => route.fulfill({ json: [rule] }));
  try { await run(page); console.log(`PASS: ${name}`); }
  catch (error) { failures++; console.error(`FAIL: ${name}: ${error.message}`); }
  finally { await page.close(); }
}
async function result(page) {
  const jpeg = await page.evaluate(() => {
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 8;
    return canvas.toDataURL("image/jpeg").split(",")[1];
  });
  await page.route("**/api/ui/analyze", route => route.fulfill({ json: { job_id: "synthetic" } }));
  await page.route("**/api/ui/jobs/*", route => route.fulfill({ json: {
    job_id: "synthetic", status: "done", verdict: "violation", violations: [{
      type: rule.id, timecode_sec: 42.5, description: "Синтетическое наблюдение для проверки интерфейса.",
      evidence_frames: [{ timestamp_sec: 42.5, image_base64: jpeg }],
    }],
  } }));
  await page.locator('input[type="file"]').setInputFiles({ name: "synthetic.mov", mimeType: "video/quicktime", buffer: Buffer.from("synthetic") });
  await page.getByRole("button", { name: "Проверить видео", exact: true }).click();
  await page.locator(".results-section h2").waitFor();
}
try {
  for (const loaded of [true, false, "unavailable"]) await check(`health ${loaded}: result remains available; demo notice only for stub`, async page => {
    await page.route("**/api/health", route => route.fulfill(loaded === "unavailable"
      ? { status: 503, json: {} } : { json: { status: "ok", version: "test", model_loaded: loaded } }));
    await page.goto(`${base}/#analysis`);
    await result(page);
    if (loaded === false) await page.locator(".demo-notice").waitFor();
    assert.equal(await page.locator(".demo-notice").count(), loaded === false ? 1 : 0);
  });
  for (const mode of ["failed", "pending"]) await check(`rules ${mode}: evidence survives and citation can recover`, async page => {
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    await page.route("**/api/ui/rules", async route => {
      if (mode === "pending") await gate;
      await route.fulfill(mode === "failed" ? { status: 503, json: {} } : { json: [rule] });
    });
    try {
      await page.goto(`${base}/#analysis`);
      await result(page);
      assert.equal(await page.locator(".result-article").count(), 1, "Violation hidden by catalog state");
      assert.match(await page.locator(".result-article").innerText(), /00:42,5/);
      assert.equal(await page.locator('img[alt^="Кадр-доказательство"]').count(), 1);
      if (mode === "failed") {
        await page.unroute("**/api/ui/rules");
        await page.route("**/api/ui/rules", route => route.fulfill({ json: [rule] }));
        await page.getByRole("alert").getByRole("button", { name: "Повторить" }).click();
      } else release();
      await page.getByRole("heading", { name: rule.name }).waitFor();
    } finally { release(); }
  });
  await check("loader dismissed within one second", async page => {
    await page.clock.install();
    await page.goto(base);
    await page.locator(".eye-loader").waitFor({ state: "attached" });
    await page.clock.fastForward(1000);
    assert.equal(await page.locator(".eye-loader").count(), 0);
  }, { reducedMotion: "no-preference" });
  await check("mouse drag switches slides; CTA still opens upload", async page => {
    await page.goto(base);
    const box = await page.locator(".morph-surface").boundingBox();
    await page.mouse.move(box.x + box.width - 30, box.y + 90);
    await page.mouse.down();
    await page.mouse.move(box.x + 30, box.y + 90, { steps: 5 });
    await page.mouse.up();
    assert.equal(await page.locator('.cinema-panel[data-position="active"] video').getAttribute("src"), "/demo/team.webm");
    await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();
    await page.locator(".upload-target").waitFor();
    assert.equal(await page.locator('[aria-roledescription="карусель"]').count(), 1);
  });
} finally { await browser.close(); }
process.exitCode = failures ? 1 : 0;
