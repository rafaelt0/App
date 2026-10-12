"use client";

import { useState } from "react";
import styles from "./StockLogo.module.css";
import { stockLogoForTicker } from "../lib/stockLogos";

type StockLogoProps = {
  ticker: string;
  company: string;
  size?: number;
};

function companyInitials(company: string, ticker: string): string {
  const words = company.trim().split(/\s+/).filter(Boolean);
  const initials = words.slice(0, 2).map(word => word[0]).join("").toLocaleUpperCase("pt-BR");
  return initials || ticker.trim().slice(0, 2).toLocaleUpperCase("pt-BR") || "?";
}

export default function StockLogo({ ticker, company, size = 32 }: StockLogoProps) {
  const src = stockLogoForTicker(ticker);
  const [failedSource, setFailedSource] = useState<string>();
  const showImage = Boolean(src && failedSource !== src);

  return (
    <span
      className={styles.logo}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      {showImage && src
        ? <img className={styles.image} src={src} alt="" onError={() => setFailedSource(src)} />
        : <span className={styles.initials}>{companyInitials(company, ticker)}</span>}
    </span>
  );
}
