import { useRef } from "react";
import { useMotionVisibility } from "@/components/useMotionVisibility";

const sponsors = [
  { src: "/forte-logo.svg", name: "Forte Bank", className: "partner-forte" },
  { src: "/dataton26-logo.png", name: "Datathon", className: "partner-datathon" },
  { src: "/NU_Logo_ENG-01.png", name: "NU", className: "partner-NU" },
  { src: "/nu_impact_foundation-04.png", name: "NU Impact", className: "partner-nuImpact" },
];

export function SponsorMarquee() {
  const container = useRef<HTMLDivElement>(null);
  const { canAnimate } = useMotionVisibility(container);
  return <div className="partners" aria-label="Партнёры проекта" ref={container}>
    <div className="marquee-window">
      <div className="marquee-track" data-paused={!canAnimate}>
        {[0, 1].map(group => <div className="marquee-group" key={group} aria-hidden={group === 1 ? true : undefined}>
          {[0, 1, 2].flatMap(repeat => sponsors.map(sponsor => <div className="partner-slot" data-repeat={repeat > 0} key={`${repeat}-${sponsor.name}`}>
            <img src={sponsor.src} alt={group === 0 && repeat === 0 ? sponsor.name : ""} className={sponsor.className} />
          </div>))}
        </div>)}
      </div>
    </div>
  </div>;
}
