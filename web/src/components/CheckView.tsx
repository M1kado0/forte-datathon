import { useEffect, useRef, useState } from "react";
import { api, type Job, type RuleType } from "@/api";
import { Button, ErrorBox, Skeleton } from "@/components/ui";
import { VideoUpload, validateVideo } from "@/components/VideoUpload";

export function CheckView() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rules, setRules] = useState<RuleType[]>([]);
  const [rulesLoading, setRulesLoading] = useState(true);
  const [rulesError, setRulesError] = useState("");
  const [error, setError] = useState("");
  const active = useRef(0);
  useEffect(() => () => { active.current += 1; }, []);
  async function loadRules() {
    setRulesLoading(true);
    setRulesError("");
    try { setRules(await api.rules()); }
    catch { setRulesError("Не удалось загрузить нормативные источники. Повторите загрузку."); }
    finally { setRulesLoading(false); }
  }
  useEffect(() => { void loadRules(); }, []);

  async function analyze() {
    if (!file || busy) return;
    const validation = validateVideo(file);
    if (validation) { setError(validation); return; }
    const run = ++active.current;
    setBusy(true);
    setUploading(true);
    setError("");
    setJob(null);
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
    <section className="analysis-section" aria-labelledby="analysis-title">
      <div className="check-layout">
        <div className="check-heading">
          <h2 id="analysis-title">Проверьте <br />кассовую <br /><span>дисциплину.</span></h2>
          <p className="intro-copy">От видеозаписи — к наблюдению,<br className="desktop-break" /> таймкоду и нормативному источнику.</p>
        </div>
        <form onSubmit={event => { event.preventDefault(); void analyze(); }} className="check-form">
          <VideoUpload file={file} disabled={busy} onChange={selected => { setFile(selected); setJob(null); setError(""); }} />
          <Button type="submit" className="analyze-button" disabled={!file || busy}>
            <span>{uploading ? "Загружаем видео…" : busy ? "Проверяем видео…" : "Проверить видео"}</span>
            <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M4 12h15m-6-6 6 6-6 6" /></svg>
          </Button>
          <p className="file-guidance">{!file ? "Выберите видео, чтобы начать проверку." : busy ? "Дождитесь результата проверки." : "Видео выбрано. Можно начать проверку."}</p>
        </form>
      </div>
      <section aria-live="polite" aria-busy={busy} className="results-section space-y-5">
        <p className="demo-notice"><span>Демонстрационный режим</span>Результат и сплошной кадр не связаны с содержимым видео.</p>
        {busy && <><p role="status">{uploading ? "Загружаем видео…" : "Обрабатываем видео…"}</p><Skeleton className="h-20 w-full rounded-btn" /></>}
        {error && <ErrorBox message={error} onRetry={() => { void analyze(); }} />}
        {job?.status === "done" && <>
          <h2 className="text-2xl font-bold text-forte-dark">{job.verdict === "violation" ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</h2>
          <p>Результат — подсказка для проверки человеком, а не заключение.</p>
          {job.violations.length > 0 && rulesLoading && <p role="status">Загружаем нормативные источники…</p>}
          {job.violations.length > 0 && rulesError && <ErrorBox message={rulesError} onRetry={() => { void loadRules(); }} />}
          {!rulesLoading && !rulesError && job.violations.map((violation, index) => {
            const metadata = rules.find(rule => rule.id === violation.type || rule.name === violation.type);
            const suffix = metadata?.citation_suffix ? ` ${metadata.citation_suffix}` : "";
            const description = suffix && violation.description.endsWith(suffix)
              ? violation.description.slice(0, -suffix.length) : violation.description;
            return <article key={index} className="result-article space-y-4">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h3 className="min-w-0 break-words text-xl font-bold text-forte-dark">{metadata?.name ?? violation.type}</h3>
                <p className="shrink-0 font-semibold">Таймкод: {violation.timecode_sec} с</p>
              </div>
              <p className="max-w-prose break-words leading-relaxed">{description}</p>
              {metadata ? <ul className="space-y-3 text-sm">
                {metadata.rules.map((rule, ruleIndex) => <li key={ruleIndex}>
                  <a href={rule.url} target="_blank" rel="noopener noreferrer" className="break-words font-semibold text-primary underline underline-offset-4">{rule.citation}{rule.punkt ? "" : " (акт в целом; конкретный пункт не указан)"}</a>
                </li>)}
              </ul> : <p className="text-sm text-muted-foreground">Нормативный источник не получен. Проверьте ссылку в описании.</p>}
              <div className="grid gap-4 sm:grid-cols-2">
                {violation.evidence_frames.map((frame, frameIndex) => <figure key={frameIndex} className="min-w-0">
                  <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Кадр-доказательство, ${frame.timestamp_sec} с`} className="max-h-96 w-full rounded-btn bg-muted object-contain" />
                  <figcaption className="mt-2 text-sm text-muted-foreground">{frame.timestamp_sec} с</figcaption>
                </figure>)}
              </div>
            </article>;
          })}
        </>}
      </section>
    </section>
  );
}
