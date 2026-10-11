import type { Metadata } from "next";
import type { ReactNode } from "react";
import AppProvider from "../components/AppProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "B3Lab | Análise quantitativa de ações",
  description: "Fundamentos, portfólio, simulação, notícias e screener da B3.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="pt-BR"><body><AppProvider>{children}</AppProvider></body></html>;
}
