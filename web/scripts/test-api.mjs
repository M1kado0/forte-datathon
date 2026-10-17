import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";

const source = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { api } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const originalFetch = globalThis.fetch;
const requests = [];
globalThis.fetch = async (url, options) => {
  requests.push({ url, options });
  return { ok: true, json: async () => ({ job_id: "fixture" }) };
};
try {
  await api.analyze(new File(["synthetic"], "sample.mov"));
  await api.rules();
  await api.job("fixture");
  assert.equal(requests[0].options.signal, undefined, "Uploads must not have the polling timeout");
  assert.equal(requests[0].options.body.get("file").name, "sample.mov");
  assert.equal(requests[1].options?.signal, undefined);
  assert.ok(requests[2].options.signal instanceof AbortSignal, "Job polls retain a timeout");
  console.log("API regression: upload has no timeout; job polling retains timeout — PASS");
} finally { globalThis.fetch = originalFetch; }
