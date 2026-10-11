import type { Metadata } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import AppProvider from "../components/AppProvider";
import "./globals.css";

const ibmPlexSans = localFont({
  src: "./fonts/ibm-plex-sans-latin.woff2",
  weight: "400 700",
  style: "normal",
  display: "swap",
  variable: "--font-ibm-plex-sans",
});

export const metadata: Metadata = {
  title: "B3Lab | Análise quantitativa de ações",
  description: "Fundamentos, portfólio, simulação, notícias e screener da B3.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="pt-BR" className={ibmPlexSans.variable}><body><AppProvider>{children}</AppProvider></body></html>;
}
