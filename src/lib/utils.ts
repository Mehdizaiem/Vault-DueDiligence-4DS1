import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
// utils.ts
export function cn1(...args: any[]) {
    return args.filter(Boolean).join(" ");
  }
