import { useEffect, useRef } from "react";

/** A shallow, feathered backdrop above the content, below the sharp header.
 * Scroll updates touch only CSS variables, never React state or page layout. */
export function ScrollDepth() {
  const layer = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = layer.current;
    if (!element || !(CSS.supports("backdrop-filter", "blur(1px)") || CSS.supports("-webkit-backdrop-filter", "blur(1px)"))) return;
    const preference = matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;
    let previous = -1;
    const draw = () => {
      frame = 0;
      const position = preference.matches ? 0 : Math.min(1, Math.max(0, window.scrollY / 180));
      const progress = position * position * (3 - 2 * position);
      if (progress === previous) return;
      previous = progress;
      element.style.setProperty("--scroll-depth", String(progress));
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(draw); };
    const configure = () => {
      window.removeEventListener("scroll", schedule);
      if (!preference.matches) window.addEventListener("scroll", schedule, { passive: true });
      schedule();
    };
    configure();
    preference.addEventListener("change", configure);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      preference.removeEventListener("change", configure);
    };
  }, []);
  return <div className="scroll-depth" ref={layer} aria-hidden="true"><span /><span /><span /></div>;
}
