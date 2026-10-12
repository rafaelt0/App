import type { Metadata } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import AppProvider from "../components/AppProvider";
import "./globals.css";
import "./motion.css";

const ibmPlexSans = localFont({
  src: "./fonts/ibm-plex-sans-latin.woff2",
  weight: "400 700",
  style: "normal",
  display: "swap",
  variable: "--font-ibm-plex-sans",
});

const ibmPlexMono = localFont({
  src: [
    { path: "./fonts/ibm-plex-mono-latin-400.woff2", weight: "400", style: "normal" },
    { path: "./fonts/ibm-plex-mono-latin-500.woff2", weight: "500", style: "normal" },
    { path: "./fonts/ibm-plex-mono-latin-600.woff2", weight: "600", style: "normal" },
  ],
  display: "swap",
  variable: "--font-ibm-plex-mono",
});

export const metadata: Metadata = {
  title: "B3Lab | Análise quantitativa de ações",
  description: "Fundamentos, portfólio, simulação, notícias e screener da B3.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="pt-BR" className={`${ibmPlexSans.variable} ${ibmPlexMono.variable}`}><body><AppProvider>{children}</AppProvider></body></html>;
}
