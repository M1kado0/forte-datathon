// Synthetic upload only: never capture private dataset frames for agent review.
// Run against a frontend server; all API responses are mocked (APP_URL overrides the origin).
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";

const base = process.env.APP_URL ?? "http://localhost:8000";
if (process.env.TEST_VIDEO) throw new Error("Screenshots accept synthetic uploads only; do not capture dataset frames.");
const out = "../reports/screens/pr-40";
const metadata = {
  id: "synthetic_count", name: "Синтетический пример: пересчёт наличных",
  citation_suffix: "Тестовый документ, п. 42",
  rules: [{ title: "Тестовый документ для проверки интерфейса", approved_by: "Тестовое утверждение",
    punkt: "п. 42", citation: "Тестовый документ, п. 42", url: "https://example.com/source" }],
};
const catalog = [metadata, { id: "synthetic_general", name: "Синтетическое наблюдение", citation_suffix: "", rules: [
  { ...metadata.rules[0], punkt: null, citation: "Тестовый документ" },
] }];
mkdirSync(out, { recursive: true });
async function capture(page, name) {
  // Full-page captures must reveal below-the-fold copy through normal scrolling.
  for (const element of await page.locator("[data-reveal]").all()) await element.scrollIntoViewIfNeeded();
  await page.evaluate(async () => {
    await Promise.allSettled(document.getAnimations()
      .filter(animation => animation.effect?.getTiming().iterations !== Infinity)
      .map(animation => animation.finished));
    window.scrollTo({ top: 0, behavior: "instant" });
  });
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
}
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
try {
  for (const [name, width, height] of [["1440", 1440, 900], ["1280", 1280, 800], ["768", 768, 1024], ["390", 390, 844]]) {
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let mode = "initial";
    let jpeg;
    await page.route("**/api/health", route => route.fulfill({ json: { status: "ok", version: "synthetic", model_loaded: false } }));
    await page.route("**/api/ui/rules", route => route.fulfill({ json: catalog }));
    await page.route("**/api/ui/analyze", route => route.fulfill({ json: { job_id: "fixture" } }));
    await page.route("**/api/ui/jobs/*", route => route.fulfill({ json: mode === "clean"
      ? { job_id: "fixture", status: "done", verdict: "no_violation", violations: [] }
      : ["initial", "violation"].includes(mode) ? { job_id: "fixture", status: "done", verdict: "violation", violations: [{
        type: mode === "initial" ? "synthetic_general" : metadata.id, timecode_sec: 42.5,
        description: `Синтетический пример для проверки интерфейса. ${mode === "initial" ? "" : metadata.citation_suffix}`,
        evidence_frames: (mode === "initial" ? [42.5] : [42.5, 43, 44]).map(timestamp_sec => ({ timestamp_sec, image_base64: jpeg })),
      }] } : { job_id: "fixture", status: mode, verdict: null, violations: [] } }));
    await page.goto(base);
    jpeg = await page.evaluate(() => {
      const canvas = document.createElement("canvas"); canvas.width = 960; canvas.height = 600;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ebe4e8"; ctx.fillRect(0, 0, 960, 600);
      ctx.strokeStyle = "#a31551"; ctx.lineWidth = 5; ctx.strokeRect(120, 100, 400, 260);
      ctx.fillStyle = "#4a0221"; ctx.font = "28px sans-serif";
      ctx.fillText("Синтетический кадр для проверки интерфейса", 80, 480);
      return canvas.toDataURL("image/jpeg").split(",")[1];
    });
    await page.locator(".eye-loader").waitFor({ state: "detached" });
    await page.evaluate(() => document.fonts.ready);
    await capture(page, `${name}-landing`);
    await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();
    await page.getByText("Выберите видео, чтобы начать проверку.").waitFor();
    await page.waitForFunction(() => !document.querySelector(".upload-panel").inert);
    await capture(page, `${name}-empty`);
    await page.locator('input[type="file"]').setInputFiles({ name: "synthetic.mov", mimeType: "video/quicktime", buffer: Buffer.from("synthetic fixture") });
    await page.getByRole("button", { name: "Проверить видео", exact: true }).click();
    await page.getByRole("heading", { name: "Обнаружены признаки возможных нарушений" }).waitFor();
    await page.getByRole("link", { name: /акт в целом/ }).waitFor();
    await page.locator(".demo-notice").waitFor();
    await capture(page, `${name}-result`);
    assert.equal(await page.locator("img[alt^='Кадр-доказательство']").count(), 1);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    mode = "processing";
    await page.getByRole("button", { name: "Проверить видео", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "Обрабатываем видео" }).waitFor();
    await capture(page, `${name}-processing`);
    mode = "failed";
    await page.getByRole("alert").waitFor();
    await capture(page, `${name}-error`);
    mode = "clean";
    await page.getByRole("button", { name: "Повторить", exact: true }).click();
    await page.getByRole("heading", { name: "Нарушение не обнаружено", exact: true }).waitFor();
    await page.getByText("Результат — подсказка для проверки человеком, а не заключение.").waitFor();
    await capture(page, `${name}-clean`);
    assert.equal(await page.locator("article").count(), 0);
    mode = "violation";
    await page.getByRole("button", { name: "Проверить видео", exact: true }).click();
    await page.getByRole("heading", { name: metadata.name, exact: true }).waitFor();
    await page.getByRole("link", { name: /п\. 42/ }).waitFor();
    assert.equal(await page.locator("article p").filter({ hasText: metadata.citation_suffix }).count(), 0);
    assert.equal(await page.locator("img[alt^='Кадр-доказательство']").count(), 1);
    assert.equal(await page.locator(".evidence-thumbnails button").count(), 3);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await capture(page, `${name}-three-frames`);
    await page.route("**/api/ui/rules", route => route.fulfill({ status: 503, json: { detail: "test failure" } }));
    await page.reload();
    await page.locator(".eye-loader").waitFor({ state: "detached" });
    await page.locator('.cinema-panel[data-position="active"] .slide-cta').click();
    await page.waitForFunction(() => !document.querySelector(".upload-panel").inert);
    await page.locator('input[type="file"]').setInputFiles({ name: "synthetic.mov", mimeType: "video/quicktime", buffer: Buffer.from("synthetic fixture") });
    await page.getByRole("button", { name: "Проверить видео", exact: true }).click();
    await page.getByRole("alert").filter({ hasText: "Не удалось загрузить нормативные источники" }).waitFor();
    assert.equal(await page.locator(".result-article").count(), 1, "Catalog failure must not hide violations");
    assert.equal(await page.locator("img[alt^='Кадр-доказательство']").count(), 1);
    assert.equal(await page.locator(".evidence-thumbnails button").count(), 3);
    await capture(page, `${name}-rules-unavailable`);
    await page.unroute("**/api/ui/rules");
    await page.route("**/api/ui/rules", route => route.fulfill({ json: catalog }));
    await page.getByRole("button", { name: "Повторить", exact: true }).click();
    await page.getByRole("heading", { name: metadata.name, exact: true }).waitFor();
    await page.getByRole("link", { name: /п\. 42/ }).waitFor();
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`${width}px: synthetic upload, result, processing, failed, retry, clean, overflow — PASS`);
  }
} finally { await browser.close(); }
console.log(`Screenshots: ${out}`);
