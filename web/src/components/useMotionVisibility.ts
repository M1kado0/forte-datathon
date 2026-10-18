import { useEffect, useState, type RefObject } from "react";

/** Keep nonessential playback asleep outside the viewport and in background tabs. */
export function useMotionVisibility(ref: RefObject<HTMLElement>) {
  const [inView, setInView] = useState(false);
  const [reduced, setReduced] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateMotion = () => setReduced(preference.matches);
    const updateVisibility = () => setPageVisible(!document.hidden);
    let observer: IntersectionObserver | undefined;
    try {
      observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { threshold: .15 });
      if (ref.current) observer.observe(ref.current);
    } catch { /* Leave decorative playback paused if visibility cannot be observed. */ }
    preference.addEventListener("change", updateMotion);
    document.addEventListener("visibilitychange", updateVisibility);
    return () => { observer?.disconnect(); preference.removeEventListener("change", updateMotion); document.removeEventListener("visibilitychange", updateVisibility); };
  }, [ref]);
  return { inView, reduced, pageVisible, canAnimate: inView && pageVisible && !reduced };
}
