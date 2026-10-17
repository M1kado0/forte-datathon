import { useEffect, useState } from "react";

/** A short introduction, independent of network/model readiness. Never gates the form. */
export function EyeLoader() {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const dismiss = () => setVisible(false);
    if (motion.matches) dismiss();
    const timer = window.setTimeout(dismiss, 2600);
    motion.addEventListener("change", dismiss);
    window.addEventListener("keydown", dismiss, { once: true });
    window.addEventListener("pointerdown", dismiss, { once: true });
    return () => {
      window.clearTimeout(timer);
      motion.removeEventListener("change", dismiss);
      window.removeEventListener("keydown", dismiss);
      window.removeEventListener("pointerdown", dismiss);
    };
  }, []);
  if (!visible) return null;
  return <div className="eye-loader" aria-hidden="true">
    <svg className="eye-drawing" viewBox="0 0 320 180" fill="none">
      <defs><clipPath id="opening-eye"><path className="eyelid" d="M20 90 Q160 90 300 90 Q160 90 20 90Z" /></clipPath></defs>
      <g clipPath="url(#opening-eye)">
        <path fill="var(--background)" d="M0 0H320V180H0Z" />
        <g className="eye-iris">
          <circle cx="160" cy="90" r="59" fill="var(--forte-magenta)" />
          <circle cx="160" cy="90" r="27" fill="var(--forte-dark-magenta)" />
          <circle cx="176" cy="72" r="8" fill="var(--background)" />
        </g>
      </g>
      <path className="eyelid" d="M20 90 Q160 90 300 90 Q160 90 20 90Z" stroke="currentColor" strokeWidth="9" strokeLinejoin="round" />
    </svg>
  </div>;
}
