import { useEffect, useRef, useState } from "react";
import { api, type Job, type RuleType } from "@/api";
import { Button, ErrorBox, Skeleton } from "@/components/ui";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rules, setRules] = useState<RuleType[]>([]);
  const [error, setError] = useState("");
  const [logoAvailable, setLogoAvailable] = useState(true);
  const active = useRef(0);
  useEffect(() => () => { active.current += 1; }, []);
  useEffect(() => { void api.rules().then(setRules).catch(() => {}); }, []);

  async function analyze() {
    if (!file || busy) return;
    const run = ++active.current;
    setBusy(true);
    setUploading(true);
    setError("");
    setJob(null);
    if (!rules.length) void api.rules().then(setRules).catch(() => {});
    try {
      const { job_id } = await api.analyze(file);
      if (active.current !== run) return;
      setUploading(false);
      const deadline = Date.now() + 300_000;
      while (active.current === run) {
        const result = await api.job(job_id);
        if (active.current !== run) return;
        setJob(result);
        if (result.status === "failed") throw new Error("Не удалось обработать видео. Повторите проверку.");
        if (result.status === "done") return;
        if (Date.now() >= deadline) throw new Error("Время ожидания истекло. Повторите проверку.");
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (cause) {
      if (active.current === run) setError(cause instanceof Error ? cause.message : "Не удалось связаться с сервисом. Повторите проверку.");
    } finally {
      if (active.current === run) { setBusy(false); setUploading(false); }
    }
  }

  return (
    <main className="mx-auto w-full max-w-5xl px-5 py-10 sm:px-8 sm:py-16">
      {logoAvailable && <img src="/forte-logo.svg" alt="ForteBank" onError={() => setLogoAvailable(false)} className="mb-12 h-10 w-auto" />}
      <div className="bg-hero mb-8 rounded-card px-5 py-8 sm:px-8 sm:py-12">
        <h1 className="max-w-xl text-3xl font-extrabold text-primary-foreground sm:text-4xl">Проверьте кассовую дисциплину</h1>
      </div>
      <form onSubmit={event => { event.preventDefault(); void analyze(); }} className="space-y-5">
        <label className="block font-semibold">Загрузите видео .mov
          <input type="file" accept=".mov,video/quicktime" disabled={busy} onChange={event => {
            setFile(event.target.files?.[0] ?? null); setJob(null); setError("");
          }} className="mt-3 block w-full rounded-btn border bg-background p-4 text-sm file:mr-4 file:font-semibold" />
        </label>
        <Button type="submit" className="w-full sm:w-auto" disabled={!file || busy}>{uploading ? "Загружаем видео…" : busy ? "Проверяем видео…" : "Проверить видео"}</Button>
      </form>
      <section aria-live="polite" aria-busy={busy} className="mt-10 space-y-5">
        <p className="text-sm text-muted-foreground">Демонстрационная заглушка: результат и сплошной кадр не связаны с содержимым видео.</p>
        {!file && <p>Выберите видео, чтобы начать проверку.</p>}
        {busy && <><p role="status">{uploading ? "Загружаем видео…" : "Обрабатываем видео…"}</p><Skeleton className="h-20 w-full rounded-btn" /></>}
        {error && <ErrorBox message={error} onRetry={() => { void analyze(); }} />}
        {job?.status === "done" && <>
          <h2 className="text-2xl font-bold text-forte-dark">{job.verdict === "violation" ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</h2>
          <p>Результат — подсказка для проверки человеком, а не заключение.</p>
          {job.violations.map((violation, index) => {
            const metadata = rules.find(rule => rule.id === violation.type || rule.name === violation.type);
            return <article key={index} className="space-y-4 rounded-card bg-card p-5 sm:p-8">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h3 className="min-w-0 break-words text-xl font-bold text-forte-dark">{metadata?.name ?? (/^[\x00-\x7F]+$/.test(violation.type) ? "Тип требует уточнения" : violation.type)}</h3>
                <p className="shrink-0 font-semibold">Таймкод: {violation.timecode_sec} с</p>
              </div>
              <p className="max-w-prose break-words leading-relaxed">{violation.description}</p>
              {metadata ? <ul className="space-y-3 text-sm">
                {metadata.rules.map((rule, ruleIndex) => <li key={ruleIndex}>
                  <a href={rule.url} target="_blank" rel="noopener noreferrer" className="break-words font-semibold text-primary underline underline-offset-4">{rule.title}, утв. {rule.approved_by}{rule.punkt ? `, ${rule.punkt}` : " (акт в целом; конкретный пункт не указан)"}</a>
                </li>)}
              </ul> : <p className="text-sm text-muted-foreground">Нормативный источник не получен. Проверьте ссылку в описании.</p>}
              <div className="grid gap-4 sm:grid-cols-2">
                {violation.evidence_frames.map((frame, frameIndex) => <figure key={frameIndex} className="min-w-0">
                  <a href={`data:image/jpeg;base64,${frame.image_base64}`} download={`evidence-${index + 1}-${frameIndex + 1}.jpg`} aria-label={`Скачать кадр ${frame.timestamp_sec} с`} className="block rounded-btn focus-visible:outline-primary">
                    <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Кадр-доказательство, ${frame.timestamp_sec} с`} className="max-h-96 w-full rounded-btn bg-muted object-contain" />
                  </a>
                  <figcaption className="mt-2 text-sm text-muted-foreground">{frame.timestamp_sec} с · Нажмите, чтобы скачать кадр</figcaption>
                </figure>)}
              </div>
            </article>;
          })}
        </>}
      </section>
    </main>
  );
}
