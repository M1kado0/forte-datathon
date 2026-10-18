import { useEffect, useRef, useState } from "react";
import { EyeLoader } from "@/components/EyeLoader";
import { Header } from "@/components/Header";
import { HeroCarousel } from "@/components/HeroCarousel";
import { About } from "@/components/About";
import { ScrollDepth } from "@/components/ScrollDepth";

export default function App() {
  const [uploadOpen, setUploadOpen] = useState(() => window.location.hash === "#analysis");
  const [motionPaused, setMotionPaused] = useState(false);
  const navigation = useRef(0);
  function navigateTo(id: string) {
    const run = ++navigation.current;
    if (id !== "about") setUploadOpen(id === "analysis");
    if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
    // Native anchor scrolling can chase the changing height during the morph.
    // Wait for the committed layout, then use an explicit sticky-header offset.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const transitions = document.querySelector(".cinema-stage")?.getAnimations({ subtree: true })
        .filter(animation => animation.playState === "running" && animation.effect?.getTiming().iterations !== Infinity) ?? [];
      void Promise.allSettled(transitions.map(animation => animation.finished)).then(() => {
        if (run !== navigation.current) return;
        const target = document.getElementById(id);
        if (!target) return;
        const offset = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-height")) + 24;
        target.focus({ preventScroll: true });
        window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - offset, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
      });
    }));
  }
  useEffect(() => {
    const navigate = () => {
      if (location.hash === "#analysis") setUploadOpen(true);
      if (location.hash === "#home") setUploadOpen(false);
    };
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(() => {
    // Progressive enhancement: text is visible until an observer is ready.
    if (!("IntersectionObserver" in window) || !Element.prototype.animate) return;
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (preference.matches) return;
    const elements = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    const animations = new Set<Animation>();
    const easing = getComputedStyle(document.documentElement).getPropertyValue("--ease-reveal").trim();
    let observer: IntersectionObserver | undefined;
    const showAll = () => {
      observer?.disconnect();
      elements.forEach(element => element.removeAttribute("data-reveal-pending"));
      animations.forEach(animation => animation.cancel());
      animations.clear();
    };
    try {
      observer = new IntersectionObserver(entries => {
        for (const entry of entries) if (entry.isIntersecting) {
          const element = entry.target as HTMLElement;
          observer?.unobserve(element);
          element.removeAttribute("data-reveal-pending");
          if (preference.matches) continue;
          try {
            const animation = element.animate(
              [{ opacity: 0, transform: "translateY(18px)" }, { opacity: 1, transform: "translateY(0)" }],
              { duration: 720, delay: Number(element.dataset.revealDelay ?? 0), easing, fill: "backwards" },
            );
            animations.add(animation);
            void animation.finished.then(() => animations.delete(animation), () => animations.delete(animation));
          } catch { /* Keep the default visible text if animation is unavailable. */ }
        }
      }, { threshold: .12 });
      elements.forEach(element => {
        // Already-visible copy belongs to the existing introduction, not a second reveal.
        const rect = element.getBoundingClientRect();
        if (rect.top < innerHeight && rect.bottom > 0) return;
        observer!.observe(element);
        element.setAttribute("data-reveal-pending", "");
      });
    } catch { showAll(); }
    preference.addEventListener("change", showAll);
    // Keyboard navigation never lands on text that is still waiting to reveal.
    const showFocused = (event: FocusEvent) => {
      if (!(event.target instanceof HTMLElement)) return;
      const target = event.target;
      elements.filter(element => element.contains(target) || target.contains(element)).forEach(element => {
        observer?.unobserve(element);
        element.removeAttribute("data-reveal-pending");
        animations.forEach(animation => { if ((animation.effect as KeyframeEffect).target === element) animation.cancel(); });
      });
    };
    document.addEventListener("focusin", showFocused);
    return () => { showAll(); preference.removeEventListener("change", showAll); document.removeEventListener("focusin", showFocused); };
  }, []);
  return <>
    <EyeLoader />
    <ScrollDepth />
    <a className="skip-link" href="#analysis" onClick={event => { event.preventDefault(); navigateTo("analysis"); }}>Перейти к проверке</a>
    <Header uploadOpen={uploadOpen} onNavigate={navigateTo} />
    <main id="main"><HeroCarousel uploadOpen={uploadOpen} onUploadChange={setUploadOpen} paused={motionPaused} onPauseChange={setMotionPaused} /><About paused={motionPaused} /></main>
    <footer className="site-footer">
      <div className="footer-bottom section-shell">
        <span>Forte Datathon ’26</span>
        <nav aria-label="Навигация в подвале"><a href="#home" onClick={event => { event.preventDefault(); navigateTo("home"); }}>Наверх</a><a href="#analysis" onClick={event => { event.preventDefault(); navigateTo("analysis"); }}>Проверить видео</a><a href="#about" onClick={event => { event.preventDefault(); navigateTo("about"); }}>О проекте</a></nav>
        <a href="/demo/sources.txt" target="_blank" rel="noopener noreferrer">Источники видео</a>
      </div>
    </footer>
  </>;
}
