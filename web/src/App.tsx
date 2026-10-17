import { useEffect, useRef, useState } from "react";
import { api, type Job } from "@/api";
import { Button, ErrorBox, Skeleton } from "@/components/ui";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef(0);
  useEffect(() => () => { active.current += 1; }, []);

  async function analyze() {
    if (!file || busy) return;
    const run = ++active.current;
    setBusy(true);
    setError("");
    setJob(null);
    try {
      const { job_id } = await api.analyze(file);
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
      if (active.current === run) setError(cause instanceof Error ? cause.message : "Повторите проверку.");
    } finally {
      if (active.current === run) setBusy(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-5 py-10 sm:px-8 sm:py-16">
      <p className="mb-12 text-xl font-bold text-forte-dark">ForteBank</p>
      <h1 className="mb-8 text-3xl font-extrabold text-forte-dark sm:text-4xl">Проверьте кассовую дисциплину</h1>
      <form onSubmit={event => { event.preventDefault(); void analyze(); }} className="space-y-5">
        <label className="block font-semibold">Загрузите видео .mov
          <input type="file" accept=".mov,video/quicktime" disabled={busy} onChange={event => {
            setFile(event.target.files?.[0] ?? null); setJob(null); setError("");
          }} className="mt-3 block w-full rounded-btn border bg-background p-4 text-sm file:mr-4 file:font-semibold" />
        </label>
        <Button type="submit" disabled={!file || busy}>{busy ? "Проверяем видео…" : "Проверить видео"}</Button>
      </form>
      <section aria-live="polite" aria-busy={busy} className="mt-10 space-y-5">
        <p className="text-sm text-muted-foreground">Демонстрационная заглушка: результат и сплошной кадр не связаны с содержимым видео.</p>
        {!file && <p>Выберите видео, чтобы начать проверку.</p>}
        {busy && <><p role="status">Обрабатываем видео…</p><Skeleton className="h-20 w-full rounded-btn" /></>}
        {error && <ErrorBox message={error} />}
        {job?.status === "done" && <>
          <h2 className="text-2xl font-bold text-forte-dark">{job.verdict === "violation" ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</h2>
          <p>Результат — подсказка для проверки человеком, а не заключение.</p>
          {job.violations.map((violation, index) => <article key={index} className="space-y-3 border-t pt-5">
            <h3 className="font-bold">{violation.type}</h3>
            <p>Таймкод: {violation.timecode_sec} с</p><p className="break-words">{violation.description}</p>
            {violation.evidence_frames.map((frame, frameIndex) => <img key={frameIndex} src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Демонстрационный кадр, ${frame.timestamp_sec} с`} className="h-32 w-32 border object-contain" />)}
          </article>)}
        </>}
      </section>
    </main>
  );
}
