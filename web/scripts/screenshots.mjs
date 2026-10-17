// Playwright screenshots at 1440 and 390 px. Needs the app running: `make app` (port 8000).
// First time on a laptop: `npx playwright install chromium`.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.APP_URL ?? "http://localhost:8000";
const OUT = "../reports/screens";
const DEMO_ID = process.env.DEMO_ID ?? "KZ-100077";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {},
);
for (const [name, width, height] of [["desktop", 1440, 900], ["mobile", 390, 844]]) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 });
  await page.goto(BASE);
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${OUT}/${name}-1-empty.png`, fullPage: true });

  await page.fill("#cid", "NO-SUCH-ID");
  await page.click("button[type=submit]");
  await page.waitForSelector("[role=alert]");
  await page.screenshot({ path: `${OUT}/${name}-2-error.png`, fullPage: true });

  await page.fill("#cid", DEMO_ID);
  await page.click("button[type=submit]");
  await page.waitForSelector("text=Почему такой результат");
  await page.screenshot({ path: `${OUT}/${name}-3-result.png`, fullPage: true });

  // what-if flip: drag the slider to the minimum
  await page.fill("input[type=range]", "0");
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${OUT}/${name}-3b-whatif.png`, fullPage: true });

  await page.goto(`${BASE}/#insights`);
  await page.waitForSelector("text=Вывод");
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}/${name}-4-insights.png`, fullPage: true });
  await page.close();
}
await browser.close();
console.log(`saved to ${OUT}`);
