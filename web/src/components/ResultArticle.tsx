import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { EvidenceFrame, RuleType, Violation } from "@/api";
import { ResultParticles } from "@/components/ResultParticles";

function formatTimecode(seconds: number) {
  const tenths = Math.round(seconds * 10);
  return `${String(Math.floor(tenths / 600)).padStart(2, "0")}:${String(Math.floor(tenths / 10) % 60).padStart(2, "0")},${tenths % 10}`;
}

function nearestFrame(violation: Violation) {
  return violation.evidence_frames.reduce((closest, frame, index, frames) =>
    Math.abs(frame.timestamp_sec - violation.timecode_sec) < Math.abs(frames[closest].timestamp_sec - violation.timecode_sec) ? index : closest, 0);
}

function EvidenceImage({ frame, thumbnail = false }: { frame: EvidenceFrame; thumbnail?: boolean }) {
  const [failed, setFailed] = useState(false);
  return failed ? <div className={thumbnail ? "evidence-thumbnail-error" : "evidence-image-error"} role="img" aria-label="Кадр недоступен">{thumbnail ? "Нет кадра" : "Не удалось показать кадр. Выберите другой кадр или повторите проверку."}</div>
    : <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt={thumbnail ? "" : `Кадр-доказательство, ${frame.timestamp_sec} с`} onError={() => setFailed(true)} />;
}

function FrameViewer({ frames, selected, onSelect, onClose }: {
  frames: EvidenceFrame[];
  selected: number;
  onSelect: (index: number) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const [zoomed, setZoomed] = useState(false);
  const frame = frames[selected];
  useEffect(() => {
    const element = dialog.current!;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    element.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      element.close();
      document.body.style.overflow = overflow;
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);
  function select(index: number) {
    setZoomed(false);
    onSelect(index);
  }
  return createPortal(<dialog ref={dialog} className="evidence-dialog" aria-labelledby={titleId}
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) onClose(); }}
    onKeyDown={event => {
      // A portal still bubbles through React: Escape must not close the upload panel.
      event.stopPropagation();
      if (event.key === "Tab") {
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex="0"]'));
        const first = controls[0], last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
      if (!zoomed && event.key === "ArrowLeft" && selected > 0) { event.preventDefault(); select(selected - 1); }
      if (!zoomed && event.key === "ArrowRight" && selected < frames.length - 1) { event.preventDefault(); select(selected + 1); }
    }}>
    <div className="evidence-dialog-heading">
      <h2 id={titleId}>Кадр {selected + 1} из {frames.length} <span>· {formatTimecode(frame.timestamp_sec)}</span></h2>
      <button type="button" onClick={onClose} className="evidence-dialog-close" aria-label="Закрыть просмотр кадра">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
      </button>
    </div>
    <div key={selected} className="evidence-viewer" data-zoomed={zoomed} tabIndex={0} aria-label="Изображение кадра; в исходном размере доступна прокрутка">
      <EvidenceImage key={selected} frame={frame} />
    </div>
    <div className="evidence-dialog-controls">
      <button type="button" aria-pressed={zoomed} onClick={() => setZoomed(value => !value)}>{zoomed ? "Вписать в окно" : "Исходный размер"}</button>
      {frames.length > 1 && <div className="evidence-step-controls">
        <button type="button" disabled={selected === 0} onClick={() => select(selected - 1)} aria-label="Предыдущий кадр"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M20 12H4m6-6-6 6 6 6" /></svg></button>
        <button type="button" disabled={selected === frames.length - 1} onClick={() => select(selected + 1)} aria-label="Следующий кадр"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M4 12h16m-6-6 6 6-6 6" /></svg></button>
      </div>}
    </div>
  </dialog>, document.body);
}

export function ResultArticle({ violation, metadata }: { violation: Violation; metadata?: RuleType }) {
  const headingId = useId();
  const [selected, setSelected] = useState(() => nearestFrame(violation));
  const [viewerOpen, setViewerOpen] = useState(false);
  const mainFrame = useRef<HTMLButtonElement>(null);
  const frames = violation.evidence_frames;
  const frame = frames[selected];
  const suffix = metadata?.citation_suffix ? ` ${metadata.citation_suffix}` : "";
  const description = suffix && violation.description.endsWith(suffix)
    ? violation.description.slice(0, -suffix.length) : violation.description;
  const eventTime = formatTimecode(violation.timecode_sec);
  const timeLabel = <><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg><span>Таймкод <strong>{eventTime}</strong></span></>;

  return <article className="result-article" aria-labelledby={headingId}>
    <ResultParticles />
    <div className="result-article-heading">
      <h3 id={headingId}>{metadata?.name ?? violation.type}</h3>
      {frames.length > 0 ? <button type="button" className="result-timecode" title={`${violation.timecode_sec} с`} aria-label={`Показать ближайший кадр к таймкоду ${eventTime}`} onClick={() => {
        setSelected(nearestFrame(violation));
        mainFrame.current?.focus({ preventScroll: true });
        mainFrame.current?.scrollIntoView({ block: "nearest", behavior: "instant" });
      }}>{timeLabel}</button> : <p className="result-timecode" title={`${violation.timecode_sec} с`}>{timeLabel}</p>}
    </div>
    <div className="result-detail">
      <div className="result-evidence">
        {frame ? <div className="evidence-gallery">
          <figure>
            <button type="button" ref={mainFrame} className="evidence-open" onClick={() => setViewerOpen(true)} aria-label={`Открыть кадр ${selected + 1}, ${formatTimecode(frame.timestamp_sec)}, в полном размере`}>
              <EvidenceImage key={selected} frame={frame} />
              <span className="evidence-open-label"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" /></svg>Открыть кадр</span>
            </button>
            <figcaption aria-live="polite"><span>Кадр {selected + 1} из {frames.length}</span><span>{formatTimecode(frame.timestamp_sec)}</span></figcaption>
          </figure>
          {frames.length > 1 && <div className="evidence-thumbnails" role="group" aria-label="Выбрать кадр-доказательство">
            {frames.map((item, index) => <button type="button" key={index} aria-pressed={index === selected} aria-label={`Показать кадр ${index + 1}, ${formatTimecode(item.timestamp_sec)}`} onClick={() => setSelected(index)}>
              <EvidenceImage frame={item} thumbnail />
              <span>{formatTimecode(item.timestamp_sec)}</span>
            </button>)}
          </div>}
        </div> : <p className="evidence-empty">Кадры не получены. Повторите проверку видео.</p>}
      </div>
      <div className="result-observation">
        <h4>Наблюдение</h4>
        <p className="result-description">{description}</p>
        <div className="result-sources">
          <h4>Нормативный источник</h4>
          {metadata && metadata.rules.length > 0 ? <ul>{metadata.rules.map((rule, index) => <li key={index}>
            <p className="source-document">{rule.title || rule.citation}</p>
            {rule.approved_by && <p className="source-authority">Утверждён: {rule.approved_by}</p>}
            <p className="source-point">{rule.punkt ?? "Акт в целом; конкретный пункт не указан"}</p>
            <a href={rule.url} target="_blank" rel="noopener noreferrer" aria-label={`Открыть источник: ${rule.citation}${rule.punkt ? "" : " (акт в целом; конкретный пункт не указан)"}`}>
              Открыть источник <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7" /></svg>
            </a>
          </li>)}</ul> : <p>Нормативный источник не получен. Проверьте ссылку в описании.</p>}
        </div>
      </div>
    </div>
    {viewerOpen && frame && <FrameViewer frames={frames} selected={selected} onSelect={setSelected} onClose={() => setViewerOpen(false)} />}
  </article>;
}
