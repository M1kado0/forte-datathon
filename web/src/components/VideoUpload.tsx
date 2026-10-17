import { useRef, useState, type DragEvent } from "react";

// The server validates the filename suffix, not MIME type; no size cap is defined.
export function validateVideo(file: File): string {
  return /\.mov$/i.test(file.name) ? "" : "Поддерживается только .mov. Выберите видео в формате QuickTime.";
}

export function VideoUpload({ file, disabled, onChange }: {
  file: File | null; disabled: boolean; onChange: (file: File | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  function select(files: File[]) {
    if (disabled || files.length === 0) return;
    const issue = files.length > 1 ? "Выберите одно видео для проверки." : validateVideo(files[0]);
    setError(issue);
    if (!issue) onChange(files[0]);
    if (input.current) input.current.value = "";
  }
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault(); dragDepth.current = 0; setDragging(false);
    select(Array.from(event.dataTransfer.files));
  }
  return <div className="upload-shell">
    <div className="upload-zone" data-dragging={dragging && !disabled} data-disabled={disabled}
      onDragEnter={event => { event.preventDefault(); if (!disabled) { dragDepth.current++; setDragging(true); } }}
      onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = disabled ? "none" : "copy"; }}
      onDragLeave={event => { event.preventDefault(); if (--dragDepth.current <= 0) setDragging(false); }} onDrop={drop}>
      <input ref={input} id="video-file" className="sr-only" type="file" tabIndex={-1} accept=".mov,video/quicktime" disabled={disabled}
        aria-label="Загрузите видео .mov" aria-describedby="upload-hint" onChange={event => select(Array.from(event.target.files ?? []))} />
      <button type="button" className="upload-target" disabled={disabled} onClick={() => input.current?.click()} aria-describedby="upload-hint">
        <svg width="36" height="36" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <path d="M16 21V5m-6 6 6-6 6 6M6 21v5a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-5" />
        </svg>
        <span className="upload-title">{dragging ? "Отпустите видео здесь" : file ? "Заменить видео" : "Перетащите видео сюда"}</span>
        <span id="upload-hint" className="upload-hint">Формат .mov · QuickTime</span>
        <span className="file-picker-label">{file ? "Выбрать другое" : "Выбрать видео"}</span>
      </button>
      {file && <div className="selected-file">
        <svg width="24" height="28" viewBox="0 0 24 28" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true"><path d="M5 1h9l5 5v20H5ZM14 1v6h5M10 12l5 3-5 3Z" /></svg>
        <div className="file-info"><p>{file.name}</p><span>MOV · {new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(file.size / 1024 / 1024)} МБ</span></div>
        <button type="button" className="remove-file" disabled={disabled} aria-label="Удалить выбранное видео" onClick={() => { onChange(null); setError(""); }}>Удалить</button>
      </div>}
    </div>
    {error && <p role="alert" className="upload-error">{error}</p>}
  </div>;
}
