import { useEffect, useState } from "react";

const sections = [{ id: "home", label: "ГЛАВНАЯ" }, { id: "about", label: "О ПРОЕКТЕ" }, { id: "analysis", label: "АНАЛИЗ ВИДЕО" }];

export function Header({ uploadOpen, onNavigate }: { uploadOpen: boolean; onNavigate: (id: string) => void }) {
  const [active, setActive] = useState("home");
  useEffect(() => {
    let observer: IntersectionObserver | undefined;
    try {
      observer = new IntersectionObserver(entries => {
        for (const entry of entries) if (entry.isIntersecting) setActive(entry.target.id);
      }, { rootMargin: "-20% 0px -60% 0px" });
      ["home", "about"].forEach(id => { const element = document.getElementById(id); if (element) observer!.observe(element); });
    } catch { /* Anchors remain usable without section observation. */ }
    return () => observer?.disconnect();
  }, []);
  return <header className="site-header">
    <div className="header-inner section-shell">
      <a className="brand-link" href="#home" onClick={event => { event.preventDefault(); onNavigate("home"); }} aria-label="Forte Bank — главная"><img src="/forte-logo.svg" alt="ForteBank" width="104" height="31" /></a>
      <nav aria-label="Основная навигация">{sections.map(section => <a key={section.id} className={section.id === "analysis" ? "nav-cta" : undefined} href={`#${section.id}`} onClick={event => { event.preventDefault(); onNavigate(section.id); }} aria-current={(active === "about" ? "about" : uploadOpen ? "analysis" : "home") === section.id ? "location" : undefined}>{section.label}</a>)}</nav>
    </div>
  </header>;
}
