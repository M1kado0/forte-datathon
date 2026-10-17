import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent } from "react";
import { CheckView } from "@/components/CheckView";
import { SponsorMarquee } from "@/components/SponsorMarquee";
import { useMotionVisibility } from "@/components/useMotionVisibility";

// Context footage only. These clips have not been run through the analyzer.
const scenes = [
  { id: "counting", title: "Работа с наличными", description: "Движение денег, присутствие работников и контекст операции — отправная точка для проверки.", author: "Tima Miroshnichenko", url: "https://www.pexels.com/video/man-counting-money-6266425/" },
  { id: "team", title: "Внимание к контексту", description: "Кто находится у рабочего места? Один фрагмент помогает увидеть участников и последовательность действий.", author: "Tima Miroshnichenko", url: "https://www.pexels.com/video/people-counting-money-6266250/" },
  { id: "payment", title: "Момент передачи", description: "Работа кассира и клиента в кадре. Иллюстрация денежной операции, а не установленного нарушения.", author: "Kindel Media", url: "https://www.pexels.com/video/a-person-paying-at-a-cashier-8351658/" },
];
const railScenes = Array.from({ length: 14 }, (_, index) => scenes[index % scenes.length]);

/** Concave ribbon: small at the vanishing point, large at the edges.
 * Only compositor transforms change per frame; layout is measured on resize. */
function DepthRibbon({ running, reduced, selected, onSelect }: { running: boolean; reduced: boolean; selected: number; onSelect: (index: number) => void }) {
  const rail = useRef<HTMLDivElement>(null);
  const phase = useRef(0);
  const previous = useRef(selected);
  useEffect(() => {
    const element = rail.current;
    if (!element) return;
    let width = element.clientWidth, frame = 0, last = 0;
    let offset = selected - previous.current;
    if (offset === 2) offset = -1;
    if (offset === -2) offset = 1;
    previous.current = selected;
    let remaining = offset;
    if (reduced) { phase.current += offset; remaining = 0; }
    const tiles = Array.from(element.children) as HTMLElement[];
    const draw = (time: number) => {
      const dt = last ? Math.min(time - last, 40) : 0;
      last = time;
      const step = remaining * (1 - Math.exp(-dt / 140));
      remaining -= step;
      phase.current += step + (running ? dt / 1900 : 0);
      const center = Math.min(380, width * .29);
      tiles.forEach((tile, index) => {
        const t = ((index - phase.current) % 14 + 14) % 14 - 7;
        const depth = Math.abs(t) / 7;
        const x = Math.sign(t) * (center / 2 + 20 + Math.pow(Math.max(0, (Math.abs(t) - 1) / 6), 1.8) * width * .6);
        const scale = .23 + .95 * Math.pow(depth, 1.65);
        tile.style.transform = `translate(-50%, -50%) translateX(${x}px) perspective(900px) rotateY(${-Math.sign(t) * (18 + depth * 17)}deg) scale(${scale})`;
        tile.style.opacity = String(Math.min(1, Math.abs(t)));
        tile.style.zIndex = String(Math.floor(depth * 10));
      });
      if (running || Math.abs(remaining) > .001) frame = requestAnimationFrame(draw);
    };
    const resize = new ResizeObserver(() => { width = element.clientWidth; cancelAnimationFrame(frame); last = 0; frame = requestAnimationFrame(draw); });
    resize.observe(element);
    frame = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(frame); resize.disconnect(); };
  }, [running, reduced, selected]);
  return <div className="depth-ribbon" ref={rail} aria-hidden="true">
    {railScenes.map((scene, index) => <span className="ribbon-frame" key={index} data-selected={index % scenes.length === selected} onClick={() => onSelect(index % scenes.length)}>
      <img src={`/demo/${scene.id}.jpg`} alt="" width="1280" height="720" draggable={false} />
    </span>)}
  </div>;
}

export function HeroCarousel({ uploadOpen, onUploadChange }: { uploadOpen: boolean; onUploadChange: (open: boolean) => void }) {
  const [selected, setSelected] = useState(0);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [failed, setFailed] = useState(false);
  const [expanded, setExpanded] = useState(uploadOpen);
  const [moving, setMoving] = useState(false);
  const [slideBusy, setSlideBusy] = useState(false);
  const viewport = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const upload = useRef<HTMLDivElement>(null);
  const videos = useRef<(HTMLVideoElement | null)[]>([]);
  const ctas = useRef<(HTMLButtonElement | null)[]>([]);
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const snapshot = useRef<{ rect: DOMRect; height: number } | null>(null);
  const animations = useRef<Animation[]>([]);
  const lock = useRef(false);
  const focusAfter = useRef(false);
  const { reduced, canAnimate } = useMotionVisibility(viewport);
  const scene = scenes[selected];

  function go(index: number) {
    if (lock.current || slideBusy || uploadOpen) return;
    const next = (index + scenes.length) % scenes.length;
    if (next === selected) return;
    setSelected(next); setFailed(false); setSlideBusy(true);
  }
  useEffect(() => {
    if (!slideBusy) return;
    const timer = window.setTimeout(() => setSlideBusy(false), reduced ? 0 : 650);
    return () => clearTimeout(timer);
  }, [slideBusy, reduced]);

  function changeView(open: boolean) {
    if (lock.current || slideBusy) return;
    focusAfter.current = true;
    onUploadChange(open);
  }
  // Measure before the React layout change, then FLIP the same surface.
  // CheckView stays mounted, preserving file, job polling and results.
  useLayoutEffect(() => {
    if (uploadOpen === expanded || !surface.current || !stage.current) return;
    snapshot.current = { rect: surface.current.getBoundingClientRect(), height: stage.current.getBoundingClientRect().height };
    animations.current.forEach(animation => animation.cancel());
    lock.current = true;
    setMoving(true);
    setExpanded(uploadOpen);
  }, [uploadOpen, expanded]);

  useLayoutEffect(() => {
    const before = snapshot.current;
    if (!before || !surface.current || !stage.current || !upload.current) return;
    snapshot.current = null;
    const after = surface.current.getBoundingClientRect();
    const duration = reduced ? 0 : expanded ? 880 : 620;
    const easing = getComputedStyle(document.documentElement).getPropertyValue("--ease-morph").trim();
    const options = { duration, easing };
    const size = surface.current.animate([
      { transform: `translate(${before.rect.x - after.x}px, ${before.rect.y - after.y}px) scale(${before.rect.width / after.width}, ${before.rect.height / after.height})`, borderRadius: expanded ? "12px" : "16px" },
      { transform: "translate(0, 0) scale(1, 1)", borderRadius: expanded ? "16px" : "12px" },
    ], options);
    const flow = stage.current.animate([{ height: `${before.height}px` }, { height: `${stage.current.getBoundingClientRect().height}px` }], options);
    const content = upload.current.animate(expanded ? [
      { opacity: 0, transform: reduced ? "none" : "translateY(16px)", offset: 0 },
      { opacity: 0, transform: reduced ? "none" : "translateY(16px)", offset: .3 },
      { opacity: 1, transform: "none", offset: 1 },
    ] : [{ opacity: 0 }, { opacity: 0 }], options);
    animations.current = [size, flow, content];
    let cancelled = false;
    void size.finished.then(() => {
      if (cancelled) return;
      lock.current = false;
      setMoving(false);
    }).catch(() => { /* Replaced navigation or unmount. */ });
    const finish = () => animations.current.forEach(animation => { if (animation.playState !== "idle") animation.finish(); });
    window.addEventListener("resize", finish);
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    preference.addEventListener("change", finish);
    return () => { cancelled = true; window.removeEventListener("resize", finish); preference.removeEventListener("change", finish); animations.current.forEach(animation => animation.cancel()); };
  }, [expanded]);

  // Focus in the committed layout after React removes inert, before the next paint.
  useLayoutEffect(() => {
    if (moving || lock.current || !focusAfter.current) return;
    (expanded ? upload.current?.querySelector<HTMLButtonElement>(".upload-target") : ctas.current[selected])?.focus({ preventScroll: true });
    focusAfter.current = false;
  }, [moving, expanded, selected]);

  useEffect(() => {
    videos.current.forEach((player, index) => {
      if (!player) return;
      if (index === selected && !expanded && canAnimate) void player.play().catch(() => { /* Keep the poster when browser autoplay is unavailable. */ });
      else player.pause();
    });
  }, [selected, expanded, canAnimate]);

  function swipeEnd(event: PointerEvent<HTMLDivElement>) {
    if (!pointer.current) return;
    const dx = event.clientX - pointer.current.x, dy = event.clientY - pointer.current.y;
    if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy) * 1.5) go(selected + (dx < 0 ? 1 : -1));
    pointer.current = null;
  }

  return <section id="home" className="hero-section" aria-labelledby="hero-title" tabIndex={-1}>
    <div className="hero-intro section-shell">
      <h1 id="hero-title" className="display-title hero-title">Замечать <em>важное.</em></h1>
      <p>Видеоаналитика кассовых операций.<br />От наблюдения — к обоснованной проверке.</p>
    </div>
    <div id="analysis" className="cinema" ref={viewport} data-expanded={expanded} data-moving={moving} tabIndex={expanded ? -1 : 0} role="region" aria-roledescription="карусель" aria-label="Иллюстрации кассовых операций"
      onKeyDown={event => {
        if (expanded) { if (event.key === "Escape") { event.preventDefault(); changeView(false); } return; }
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); go(selected + (event.key === "ArrowRight" ? 1 : -1)); }
      }}>
      <div className="cinema-stage" ref={stage}
        onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
        onFocusCapture={() => setFocused(true)} onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false); }}
        onPointerDown={event => { if (!expanded && event.pointerType !== "mouse") pointer.current = { x: event.clientX, y: event.clientY }; }}
        onPointerUp={swipeEnd} onPointerCancel={() => { pointer.current = null; }}>
        <DepthRibbon running={canAnimate && !expanded && !hovered && !focused} reduced={reduced} selected={selected} onSelect={go} />
        <div className="morph-surface" ref={surface}>
          <div className="feature-films" role="region" aria-roledescription="карусель" aria-label="Иллюстрации кассовых операций" aria-hidden={expanded} {...{ inert: expanded || moving ? "" : undefined }}>
            {scenes.map((item, index) => <div key={item.id} className="cinema-panel" data-position={index === selected ? "active" : "inactive"} aria-hidden={index !== selected} {...{ inert: index !== selected ? "" : undefined }}>
              <img src={`/demo/${item.id}.jpg`} alt="" width="1280" height="720" draggable={false} />
              <video ref={element => { videos.current[index] = element; }} src={`/demo/${item.id}.webm`} poster={`/demo/${item.id}.jpg`} muted loop playsInline preload="none"
                aria-label={item.title} onError={() => { if (index === selected) setFailed(true); }} />
              <div className="cinema-label"><span>Иллюстрация</span><span>{String(index + 1).padStart(2, "0")} / 03</span></div>
              <button ref={element => { ctas.current[index] = element; }} className="primary-link slide-cta" type="button" onClick={() => changeView(true)} disabled={moving || slideBusy} aria-expanded={expanded} aria-controls="upload-panel">Проверить своё видео <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h16m-6-6 6 6-6 6" /></svg></button>
            </div>)}
          </div>
          <div id="upload-panel" className="upload-panel" ref={upload} aria-hidden={!expanded} {...{ inert: !expanded || moving ? "" : undefined }}>
            <button className="back-to-carousel" type="button" disabled={moving} onClick={() => changeView(false)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12H4m6-6-6 6 6 6" /></svg>Вернуться к роликам</button>
            <CheckView />
          </div>
        </div>
      </div>
      <div className="carousel-meta" aria-hidden={expanded} {...{ inert: expanded ? "" : undefined }}>
        <div className="cinema-details section-shell">
          <div className="scene-copy" aria-live="polite" aria-atomic="true"><h2 data-reveal>{scene.title}</h2><p data-reveal data-reveal-delay="80">{failed ? "Видео недоступно. Показан кадр иллюстрации; можно перейти к следующему ролику." : scene.description}</p></div>
        </div>
        <div className="footage-caption section-shell"><span>Постановочные кадры Pexels, не банковское CCTV.</span><a href={scene.url} target="_blank" rel="noopener noreferrer">{scene.author} / Pexels</a></div>
        <p className="hero-reassurance section-shell" data-reveal>Локальная обработка. Решение — за человеком.</p>
      </div>
      <SponsorMarquee />
    </div>
  </section>;
}

