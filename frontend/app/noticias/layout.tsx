import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = { title: "Notícias" };

export default function NoticiasLayout({ children }: { children: ReactNode }) {
  return children;
}
