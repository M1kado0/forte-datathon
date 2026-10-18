import { useRef } from "react";
import { useMotionVisibility } from "@/components/useMotionVisibility";

export function ResultParticles() {
  const layer = useRef<HTMLSpanElement>(null);
  const { canAnimate } = useMotionVisibility(layer);
  return <span className="about-particles result-particles" ref={layer} data-paused={!canAnimate} aria-hidden="true">
    {Array.from({ length: 12 }, (_, index) => <i className="about-particle" key={index} />)}
  </span>;
}
