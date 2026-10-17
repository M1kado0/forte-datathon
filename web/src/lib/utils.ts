import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** 1 250 000 ₸ — thin (narrow no-break) space thousands, as in the brand rules. */
export function formatKzt(x: number): string {
  return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ") + " ₸";
}

export function formatPct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}
