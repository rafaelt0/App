import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = { title: "Screener" };

export default function ScreenerLayout({ children }: { children: ReactNode }) {
  return children;
}
