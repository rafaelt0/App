import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = { title: "Simulação" };

export default function SimulacaoLayout({ children }: { children: ReactNode }) {
  return children;
}
