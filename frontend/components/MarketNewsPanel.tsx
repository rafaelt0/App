"use client";

import { useEffect, useId, useRef, useState } from "react";
import { loadMarketNews, type MarketNews } from "../lib/marketNews";
import { formatBrasilia, Icon, relativeAge } from "./ui";
import styles from "./MarketNewsPanel.module.css";

type FeedState = { kind: "loading" } | { kind: "ready"; data: MarketNews } | { kind: "error"; message: string };

export default function MarketNewsPanel() {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const request = useRef(0);
  const [open, setOpen] = useState(false);
  const [feed, setFeed] = useState<FeedState>({ kind: "loading" });

  useEffect(() => () => { request.current += 1; }, []);
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [open]);

  async function load() {
    const current = ++request.current;
    setFeed({ kind: "loading" });
    try {
      const data = await loadMarketNews();
      if (request.current === current) setFeed({ kind: "ready", data });
    } catch (error) {
      if (request.current === current) setFeed({ kind: "error", message: error instanceof Error ? error.message : "Não foi possível carregar as notícias." });
    }
  }

  return <>
    <button type="button" className={styles.launcher} aria-label="Abrir notícias do mercado" title="Notícias do mercado"
      aria-haspopup="dialog" aria-expanded={open} aria-controls={id} onClick={() => {
        dialog.current?.showModal(); setOpen(true); void load();
      }}><Icon name="news" /></button>
    <dialog ref={dialog} id={id} className={styles.panel} aria-labelledby={`${id}-title`} aria-describedby={`${id}-description`}
      onClose={() => setOpen(false)} onClick={event => {
        if (event.target !== dialog.current) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.current?.close();
      }}>
      <header className={styles.header}>
        <span className={styles.headerIcon}><Icon name="news" /></span>
        <div><h2 id={`${id}-title`}>Notícias do mercado</h2><p id={`${id}-description`}>Destaques do mercado brasileiro</p></div>
        <button type="button" className={styles.close} aria-label="Fechar notícias do mercado" onClick={() => dialog.current?.close()}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
        </button>
      </header>
      <div className={styles.toolbar}><span>Últimas {feed.kind === "ready" ? feed.data.window_hours : 48} horas · Por relevância</span>
        <button type="button" onClick={() => void load()} disabled={feed.kind === "loading"}>Atualizar</button>
      </div>
      <div className={styles.content} aria-busy={feed.kind === "loading"}>
        {feed.kind === "loading" && <p className={styles.message} role="status">Buscando notícias de mercado…</p>}
        {feed.kind === "error" && <div className={styles.message} role="alert"><Icon name="alert" /><p>{feed.message}</p><button type="button" onClick={() => void load()}>Tentar novamente</button></div>}
        {feed.kind === "ready" && (feed.data.articles.length ? <ol className={styles.list}>{feed.data.articles.map(article => <li key={article.id}>
          <article><div className={styles.meta}><span>{article.provider}</span><time dateTime={article.published} title={formatBrasilia(article.published)}>{relativeAge(article.published)}</time></div>
            <h3>{article.link ? <a href={article.link} target="_blank" rel="noopener noreferrer" aria-label={`${article.title} (abre em nova aba)`}>{article.title}<Icon name="arrow-right" /></a> : article.title}</h3>
            <span className={styles.importance} data-high={article.importance_label === "Alta"}>Importância {article.importance_label.toLocaleLowerCase("pt-BR")}</span>
          </article>
        </li>)}</ol> : <p className={styles.message} role="status">Nenhuma notícia encontrada nas últimas {feed.data.window_hours} horas.</p>)}
      </div>
      <footer className={styles.footer}>
        {feed.kind === "ready" && <p title={formatBrasilia(feed.data.as_of)}>Consulta {relativeAge(feed.data.as_of)} · Atualização a cada 5 min</p>}
        <p>Fonte: Google News e publicadores. Relevância por palavras-chave.</p>
      </footer>
    </dialog>
  </>;
}
