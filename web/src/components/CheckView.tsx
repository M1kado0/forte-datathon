import { useEffect, useRef, useState } from "react";
import { api, type Job, type RuleType } from "@/api";
import { Button, ErrorBox, Skeleton } from "@/components/ui";
import { VideoUpload, validateVideo } from "@/components/VideoUpload";
import { ResultParticles } from "@/components/ResultParticles";
import { ResultArticle } from "@/components/ResultArticle";

export function CheckView() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rules, setRules] = useState<RuleType[]>([]);
  const [rulesLoading, setRulesLoading] = useState(true);
  const [rulesError, setRulesError] = useState("");
  const [error, setError] = useState("");
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
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
  useEffect(() => {
    let disposed = false;
    // Health is advisory: an unavailable endpoint must not hide results.
    void api.health().then(
      health => { if (!disposed) setModelLoaded(health.model_loaded); },
      () => { if (!disposed) setModelLoaded(null); },
    );
    return () => { disposed = true; };
  }, [job?.status]);

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
      <section aria-live="polite" aria-busy={busy} className="results-section">
        {modelLoaded === false && <p className="demo-notice"><strong>Демонстрационный режим</strong>Результат и кадры не связаны с содержимым видео.</p>}
        {busy && <div className="result-pending">
          <p role="status">{uploading ? "Загружаем видео…" : "Обрабатываем видео…"}</p>
          <div className="result-skeleton"><div><Skeleton className="h-8 w-4/5" /><Skeleton className="mt-5 h-4 w-full" /><Skeleton className="mt-3 h-4 w-2/3" /></div><Skeleton className="h-40 w-full" /></div>
        </div>}
        {error && <ErrorBox message={error} onRetry={() => { void analyze(); }} />}
        {job?.status === "done" && <>
          <header className="result-summary" data-verdict={job.verdict}>
            <ResultParticles />
            <div className="result-status">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                {job.verdict === "violation" ? <path d="M12 7v6m0 3v1" /> : <path d="m7 12 3 3 7-7" />}
              </svg>
              <span>Проверка завершена</span>
              {file && <span className="result-filename" title={file.name}>{file.name}</span>}
            </div>
            <h2>{job.verdict === "violation" ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</h2>
            <p className="result-disclaimer">Результат — подсказка для проверки человеком, а не заключение.</p>
          </header>
          {job.violations.length > 0 && rulesLoading && <p role="status">Загружаем нормативные источники…</p>}
          {job.violations.length > 0 && rulesError && <ErrorBox message={rulesError} onRetry={() => { void loadRules(); }} />}
          {job.violations.map((violation, index) => <ResultArticle key={`${job.job_id}-${index}`} violation={violation}
            metadata={rules.find(rule => rule.id === violation.type || rule.name === violation.type)} />)}
        </>}
      </section>
    </section>
  );
}
