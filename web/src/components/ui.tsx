// Minimal primitives in shadcn style. Swap for `npx shadcn add button card input slider` tomorrow if needed.
import type { ButtonHTMLAttributes, HTMLAttributes, InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Button({ className, ...p }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "inline-flex h-12 items-center justify-center gap-2 rounded-[var(--radius-btn)] bg-primary px-6",
        "text-[15px] font-semibold text-primary-foreground transition-colors hover:bg-foreground",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      {...p}
    />
  );
}

export function Input({ className, ...p }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-12 w-full rounded-[var(--radius-btn)] border bg-white px-4 text-[15px]",
        "placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20",
        className,
      )}
      {...p}
    />
  );
}

export function Card({ className, ...p }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-[var(--radius-card)] bg-card p-6 shadow-[var(--shadow-card)]", className)}
      {...p}
    />
  );
}

export function Label({ className, ...p }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-xs font-semibold uppercase tracking-wide text-muted-foreground", className)} {...p} />;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden />;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-[var(--radius-btn)] border border-danger/30 bg-danger/5 px-4 py-3 text-sm">
      <span className="font-semibold text-danger">Не получилось.</span>
      <span className="text-foreground">{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="ml-auto font-semibold text-primary hover:underline">
          Повторить
        </button>
      )}
    </div>
  );
}
