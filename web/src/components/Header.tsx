import { cn } from "@/lib/utils";

export type Tab = "check" | "insights";

const PRODUCT_NAME = "Риск-скоринг"; // P4 renames tomorrow

/**
 * Logo slot. Put the OFFICIAL file from forte.kz/docs/forte-logos.zip at web/public/forte-logo.svg.
 * Never recolour it; keep clear space (padding) around it.
 */
function Logo() {
  return (
    <img
      src="/forte-logo.svg"
      alt="ForteBank"
      className="h-7 w-auto"
      onError={(e) => {
        const el = e.currentTarget;
        el.replaceWith(Object.assign(document.createElement("span"), {
          className: "inline-flex h-7 shrink-0 items-center rounded-md border border-dashed border-border px-2 text-xs text-muted-foreground",
          textContent: "лого",
        }));
      }}
    />
  );
}

export function Header({ tab }: { tab: Tab }) {
  const link = (id: Tab, label: string) => (
    <a
      href={id === "check" ? "#" : "#insights"}
      aria-current={tab === id ? "page" : undefined}
      className={cn(
        "rounded-[var(--radius-btn)] px-2 py-2 text-sm font-semibold transition-colors sm:px-3",
        tab === id ? "text-primary bg-accent" : "text-muted-foreground hover:text-forte-dark",
      )}
    >
      {label}
    </a>
  );

  return (
    <header className="sticky top-0 z-10 bg-white/95 backdrop-blur-0 border-b">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-8">
        <Logo />
        <span className="hidden sm:block h-6 w-px bg-border" />
        <span className="whitespace-nowrap text-sm font-bold text-forte-dark sm:text-base">{PRODUCT_NAME}</span>
        <nav className="ml-auto flex items-center gap-0 sm:gap-1">
          {link("check", "Проверка")}
          {link("insights", "Аналитика")}
        </nav>
      </div>
    </header>
  );
}
