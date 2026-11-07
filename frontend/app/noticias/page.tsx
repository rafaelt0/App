"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import Chart from "../../components/Chart";
import { useApp } from "../../components/AppProvider";
import Link from "next/link";
import { formatBrasilia, formatPercent, formatValue, Loading, Metrics, Notice, Pane, relativeAge, Table } from "../../components/ui";
import { tickerColor } from "../../components/Shell";
import { ElapsedLoading } from "../../components/PortfolioParts";
import "./news.css";
import { api } from "../../lib/api";
import { filterArticles, initialFilters, intensityLevels, safeNewsLink, sortModes } from "../../lib/news";
import type { NewsArticle, NewsFilters, NewsSnapshot } from "../../lib/news";

/** Sample and article timestamps older than this ask the reader to refresh. */
const staleAfterMs = 15 * 60000;

const sentimentPaths = {positive: "M12 5l7 12H5z", negative: "M12 19L5 7h14z", neutral: "M6 12h12"} as const;
function SentimentIcon({tone}: {tone: keyof typeof sentimentPaths}) {
  return <svg width="12" height="12" viewBox="0 0 24 24" fill={tone === "neutral" ? "none" : "currentColor"} stroke="currentColor" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" aria-hidden="true"><path d={sentimentPaths[tone]} /></svg>;
}

/** Engine wording for the chip and summary line; the counts come from the API, one per article. */
function engineSplit(model: NewsSnapshot["model"], total: number): string {
  if (model.jev_count === total) return "Jev em todos os artigos";
  if (model.jev_count === 0) return "PLN Léxico (fallback) em todos os artigos";
  return `Jev em ${model.jev_count} de ${total} artigos · léxico em ${model.lexicon_count}`;
}
const filterNames: Record<keyof NewsFilters, string> = {ticker: "Ativo", sentiment: "Sentimento", intensity: "Intensidade", sort: "Ordenação"};

function ExternalIcon() {
  return <svg className="news-external-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></svg>;
}

const brasiliaTime = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "America/Sao_Paulo" });

function Article({article}: {article: NewsArticle}) {
  const link = safeNewsLink(article.link);
  const tone = article.sentiment === "Otimista" ? "positive" : article.sentiment === "Pessimista" ? "negative" : "neutral";
  return <article className="feed-row" aria-label={article.title}>
    <time dateTime={article.published} title={formatBrasilia(article.published)}>{brasiliaTime.format(new Date(article.published))}</time>
    <span className="feed-tickers">{article.tickers.join(" ")}</span>
    <span className={`feed-tone is-${tone}`}><SentimentIcon tone={tone} />{article.sentiment}</span>
    <span className={`feed-score is-${tone}`} title="Score de −1 a +1">{formatValue(article.score, "number", 2, "BRL", true)}<span className="feed-bar" aria-hidden="true"><span style={{ "--magnitude": Math.min(1, Math.abs(article.score)) } as CSSProperties} /></span></span>
    <div className="feed-body">
      <h3>{link ? <a href={link} target="_blank" rel="noopener noreferrer">{article.title}<ExternalIcon /><span className="sr-only"> (abre em nova aba)</span></a> : article.title}</h3>
      <p className="feed-meta">{article.provider || "Fonte não informada"} · {relativeAge(article.published)} · importância {article.importance_label.toLocaleLowerCase("pt-BR")} · intensidade {article.intensity.toLocaleLowerCase("pt-BR")} · {article.engine} · texto: {article.text_source}</p>
      {article.summary && !article.summary.startsWith(article.title) && <p className="feed-summary">{article.summary}</p>}
      <details className="feed-model">
        <summary>Algoritmo ({article.is_jev ? "Jev" : "Léxico"} · score {formatValue(article.score)})</summary>
        {article.is_jev ? <div className="feed-model-body">
          {([ ["POSITIVE", "Otimista", "positive"], ["NEGATIVE", "Pessimista", "negative"], ["NEUTRAL", "Neutro", "neutral"] ] as const).map(([label, name, state]) => {
            const score = article.scores.find(item => item.label === label)?.score ?? 0;
            return <div className="prob-row" key={label}><span>{name}</span><span className={`prob-track is-${state}`} aria-hidden="true"><span style={{width: `${score * 100}%`}} /></span><span>{formatPercent(score, 1)}</span></div>;
          })}
          <p>Score = POSITIVE − NEGATIVE; ≥ 0,20 otimista, ≤ −0,20 pessimista, demais neutro. Scores não somam 100% e não são probabilidades calibradas. Texto: {article.raw_text_length} palavras.</p>
        </div> : <div className="feed-model-body">
          <p>Termos positivos: {article.pos_terms.length ? article.pos_terms.map((term, index) => <span className="term is-positive" key={`${index}-${term}`}>{term}</span>) : "nenhum"}</p>
          <p>Termos negativos: {article.neg_terms.length ? article.neg_terms.map((term, index) => <span className="term is-negative" key={`${index}-${term}`}>{term}</span>) : "nenhum"}</p>
          <p>Score = (Pos − Neg) / (Pos + Neg) · {article.evidence_count ? `(${article.pos_terms.length} − ${article.neg_terms.length}) / ${article.evidence_count} = ${formatValue(article.score)}` : "sem termos encontrados: score 0 (sem evidência, não confiança na neutralidade)."}</p>
          <p>{article.evidence_count} evidências · {article.raw_text_length} palavras. Menos de dois termos implica evidência limitada, e a notícia fica fora da nota agregada. Frases, negação e repetições são consideradas.</p>
        </div>}
      </details>
    </div>
  </article>;
}

export default function NewsPage() {
  const {session, loading: bootstrapLoading, refreshSession} = useApp();
  const [snapshot, setSnapshot] = useState<NewsSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<NewsFilters>(initialFilters);
  const [limit, setLimit] = useState(15);
  const generation = useRef(0);
  const current = useRef(session);
  const refresh = useRef(refreshSession);
  current.current = session;
  refresh.current = refreshSession;
  const execute = useCallback(async (refresh = false) => {
    const context = current.current;
    if (!context?.portfolio_ready) return;
    const request = ++generation.current;
    setBusy(true); setError(null);
    setFilters(initialFilters); setLimit(15);
    try {
      const result = await api<NewsSnapshot>(`/api/news/${refresh ? "refresh" : "analyze"}`, {revision: context.revision});
      if (request === generation.current && current.current?.revision === context.revision && current.current.portfolio_ready) setSnapshot(result);
    } catch (cause) {
      if (request === generation.current) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }, []);
  useEffect(() => {
    if (bootstrapLoading || !session) return;
    setSnapshot(null); setError(null);
    if (session.portfolio_ready) { void execute(); }
    else if (session.restored) {
      const request = ++generation.current;
      setBusy(true);
      void api("/api/portfolio/restore", {target: "news", revision: session.revision}).then(async () => {
        if (request === generation.current) await refresh.current();
      }).catch(cause => {
        if (request === generation.current) setError(cause instanceof Error ? cause.message : String(cause));
      }).finally(() => { if (request === generation.current) setBusy(false); });
    } else setBusy(false);
    return () => { generation.current += 1; };
  }, [bootstrapLoading, session?.revision, session?.portfolio_ready, session?.restored, execute]);

  const changeFilter = (key: keyof NewsFilters, value: string) => {
    setFilters(previous => ({...previous, [key]: value})); setLimit(15);
  };
  const clearFilters = () => { setFilters(initialFilters); setLimit(15); };
  const articles = snapshot ? filterArticles(snapshot.articles, filters) : [];
  const visible = articles.slice(0, limit);
  const positive = articles.filter(article => article.sentiment === "Otimista").length;
  const negative = articles.filter(article => article.sentiment === "Pessimista").length;
  const failed = snapshot?.feeds.filter(feed => !feed.ok).map(feed => feed.ticker) ?? [];
  const allFailed = !!snapshot && failed.length === snapshot.feeds.length;
  const retryAt = snapshot?.model.retry_after ? snapshot.model.retry_after * 1000 : 0;
  const jevPaused = !!snapshot && snapshot.model.lexicon_count > 0 && retryAt > Date.now();
  const stale = !!snapshot && Date.now() - Date.parse(snapshot.as_of) > staleAfterMs;
  const activeFilters = (["ticker", "sentiment", "intensity"] as const).filter(key => filters[key] !== initialFilters[key]);
  const examples = [["10:42", "Exemplo: manchete otimista sobre resultado trimestral", "positive"], ["09:15", "Exemplo: manchete neutra sobre agenda da empresa", "neutral"], ["08:03", "Exemplo: manchete pessimista sobre revisão de guidance", "negative"]] as const;
  return <div className="desk desk-split">
    <Pane title="Carteira e filtros" className="pane-params" meta={snapshot ? `${snapshot.tickers.length} ativos` : undefined}>
      {bootstrapLoading ? <Loading>Restaurando sessão…</Loading> : <>
        {error && <Notice tone="negative">{error}</Notice>}
        {!session?.portfolio_ready && !busy && <>
          <p className="locked-title">{session?.selection.length ? "Atualize a carteira" : "Carteira não configurada"}</p>
          <p className="pane-lead">{session?.selection.length ? "A seleção precisa de uma análise atual. Carregue a carteira antes de consultar as notícias." : "Configure e carregue a carteira para acompanhar os ativos e seus pesos."}</p>
          <div className="param-run"><Link className="button-link primary" href="/portfolio"><kbd>F2</kbd> Ir para Carteira</Link></div>
          <h3 className="pane-sub">O que aparece aqui</h3>
          <ul className="plain-list"><li>Leitura por ativo: filtre por empresa, sentimento e intensidade.</li><li>Tom da carteira: síntese ponderada pelos pesos da alocação.</li><li>Fontes e cobertura: links, motor de análise e disponibilidade de cada feed.</li></ul>
        </>}
        {session?.portfolio_ready && <>
          {snapshot && <ul className="weights" aria-label="Carteira e pesos">{snapshot.tickers.map((ticker, index) => <li key={ticker}><i className="swatch" style={{ background: tickerColor(index) }} aria-hidden="true" /><b>{ticker}</b><span className="weights-track" aria-hidden="true"><span style={{ width: `${Math.min(1, snapshot.weights[ticker]) * 100}%` }} /></span><span className="weights-value">{formatPercent(snapshot.weights[ticker], 1)}</span></li>)}</ul>}
          {snapshot && <p className="param-hint">Amostra {formatBrasilia(snapshot.as_of)} · {relativeAge(snapshot.as_of)}</p>}
          {snapshot && <p className={`engine ${!snapshot.articles.length ? "is-none" : snapshot.model.lexicon_count ? "is-fallback" : ""}`}><span aria-hidden="true">●</span>{snapshot.articles.length ? engineSplit(snapshot.model, snapshot.articles.length) : "Sem notícias classificadas"}</p>}
          <div className="param-run"><button className="primary" onClick={() => void execute(true)} disabled={busy}>{busy ? "Atualizando…" : "Atualizar notícias"}</button></div>
          {snapshot && <>
            <h3 className="pane-sub">Filtros do feed</h3>
            <div className="param-group">
              <label>Ativo<select value={filters.ticker} onChange={event => changeFilter("ticker", event.target.value)}><option value="Todos">Todos os ativos</option>{snapshot.tickers.map(ticker => <option key={ticker} value={ticker}>{ticker} ({snapshot.articles.filter(article => article.tickers.includes(ticker)).length})</option>)}</select></label>
              <label>Ordenar por<select value={filters.sort} onChange={event => changeFilter("sort", event.target.value)}>{sortModes.map(sort => <option key={sort}>{sort}</option>)}</select></label>
              <label>Sentimento<select value={filters.sentiment} onChange={event => changeFilter("sentiment", event.target.value)}><option>Todos</option><option value="Otimista">Otimistas</option><option value="Neutro">Neutras</option><option value="Pessimista">Pessimistas</option></select></label>
              <label>Intensidade<select value={filters.intensity} onChange={event => changeFilter("intensity", event.target.value)}><option>Todos</option>{intensityLevels.filter(level => level !== "Evidência limitada" || snapshot.articles.some(article => !article.is_jev)).map(level => <option key={level}>{level}</option>)}</select></label>
            </div>
            {activeFilters.length > 0 && <div className="filter-chips" aria-label="Filtros ativos">
              {activeFilters.map(key => <button type="button" className="chip" key={key} aria-label={`Remover filtro ${filterNames[key]}: ${filters[key]}`} onClick={() => changeFilter(key, initialFilters[key])}>{filterNames[key]}: {filters[key]} <span aria-hidden="true">×</span></button>)}
              <button type="button" onClick={clearFilters}>Limpar filtros</button>
            </div>}
            <p className="param-hint">Filtros e “Ver mais” mudam só a lista; nota, gráfico e síntese não mudam.</p>
            <details className="disclosure"><summary>Fontes e metodologia</summary><div>
              <p>Até {snapshot.max_per_ticker} notícias únicas mais recentes por ativo nos últimos {snapshot.window_days} dias, antes dos filtros: {snapshot.articles.length} artigos únicos · {snapshot.summary.observation_count} observações artigo-ativo. Feeds indisponíveis: {failed.join(", ") || "nenhum"}.</p>
              <p>A cobertura depende do Google News e dos feeds acessíveis. Sem texto acessível usamos descrição RSS ou título. Importância é triagem por palavras-chave, não materialidade. No léxico, menos de dois termos implica evidência limitada; essas notícias não entram na nota agregada.</p>
            </div></details>
          </>}
        </>}
      </>}
    </Pane>
    <div className="board">
      {snapshot && (failed.length > 0 || jevPaused || stale || snapshot.feeds.some(feed => feed.ok && !feed.recent_count)) && <div className="span-12 board-notes">
        {snapshot.feeds.map(feed => !feed.ok ? <Notice key={feed.ticker} tone="warning">Feed indisponível para {feed.ticker}; nenhum item foi usado na análise.</Notice> : !feed.recent_count ? <Notice key={feed.ticker}>Nenhuma notícia recente (7 dias) para {feed.ticker}.</Notice> : null)}
        {jevPaused && <Notice>Análise avançada indisponível; nova tentativa após {formatBrasilia(new Date(retryAt).toISOString(), false)}.</Notice>}
        {stale && <Notice tone="warning">Amostra com mais de 15 minutos. Use “Atualizar notícias” para buscar os feeds de novo.</Notice>}
        {failed.length > 0 && <Notice tone="warning">Alguns feeds falharam. <button onClick={() => void execute(true)} disabled={busy}>Tentar feeds novamente</button></Notice>}
      </div>}
      {busy && <Pane title="Notícias da carteira" className="span-12"><ElapsedLoading>Buscando as notícias dos seus ativos e avaliando o tom de cada uma. Pode levar até um minuto.</ElapsedLoading></Pane>}
      {!snapshot && !busy && <>
        <Pane title="Tom da carteira" className="span-12" meta="aguardando carteira">
          <Metrics size="lg" items={["Nota do tom (/100)", "Cobertura do peso", "Otimistas", "Neutras", "Pessimistas"].map(label => ({ label, value: null }))} />
        </Pane>
        <Pane title="Feed de notícias" className="span-12" meta="aguardando carteira">
          <div className="example">{examples.map(([time, title, tone]) => <div className="feed-row is-example" key={time}><time>{time}</time><span className="feed-tickers">ABCD3</span><span className={`feed-tone is-${tone}`}><SentimentIcon tone={tone} />{tone === "positive" ? "Otimista" : tone === "negative" ? "Pessimista" : "Neutro"}</span><span className="feed-score">—</span><div className="feed-body"><h3>{title}</h3></div></div>)}
            <p>Cada linha traz horário, ativos, tom, score de −1 a +1 e o link para o publicador. As notícias reais aparecem depois de carregar a carteira.</p></div>
        </Pane>
      </>}
      {snapshot && <>
        <Pane title="Tom da carteira" className="span-8" meta={snapshot.summary.label}>
          <Metrics size="lg" items={[
            {label: "Nota do tom (/100)", value: snapshot.summary.score100, status: snapshot.summary.status, unit: "integer", help: snapshot.summary.label},
            {label: "Cobertura do peso", value: snapshot.summary.coverage, unit: "fraction", digits: 1},
            {label: "Otimistas", value: snapshot.summary.counts.positive, unit: "integer"},
            {label: "Neutras", value: snapshot.summary.counts.neutral, unit: "integer"},
            {label: "Pessimistas", value: snapshot.summary.counts.negative, unit: "integer"},
          ]} />
          {snapshot.model.jev_count > 0 && snapshot.model.lexicon_count > 0 && <p className="pane-foot">A nota combina Jev e léxico, cujas escalas não são diretamente comparáveis.</p>}
          <p className="pane-foot">Escala 0–100 do tom textual, não previsão de retorno ou risco. Cobertura é disponibilidade de peso, não confiança. Ativos sem notícias ficam fora; sem peso coberto a nota é —, não 50. Contagens de artigos únicos antes dos filtros.</p>
        </Pane>
        <Pane title="Observações por ativo" className="span-4">
          <Chart figure={snapshot.summary.chart} title="Observações artigo-ativo por ativo" height={220} />
          <details className="disclosure"><summary>Dados do gráfico</summary>
            <Table caption="Observações artigo-ativo por ativo" columns={[{key: "ticker", label: "Ativo"}, {key: "positive", label: "Otim.", unit: "integer"}, {key: "neutral", label: "Neutr.", unit: "integer"}, {key: "negative", label: "Pessim.", unit: "integer"}]} rows={snapshot.summary.distribution} />
          </details>
        </Pane>
        <Pane title="Feed de notícias" className="span-12" meta={<span role="status" aria-live="polite">{visible.length} de {articles.length} após filtros · ▲ {positive} · ■ {articles.length - positive - negative} · ▼ {negative}</span>} flush>
          {!articles.length && <Notice>{snapshot.articles.length ? "Nenhuma notícia corresponde aos filtros atuais." : allFailed ? "Feeds indisponíveis; não foi possível verificar notícias recentes." : "Nenhuma notícia recente encontrada nos feeds disponíveis."}</Notice>}
          {visible.length > 0 && <ul className="feed">{visible.map(article => <li key={article.id}><Article article={article} /></li>)}</ul>}
          {articles.length > limit ? <div className="feed-more"><button onClick={() => setLimit(previous => previous + 15)}>Ver mais {Math.min(15, articles.length - limit)} notícias ({articles.length - limit} restantes)</button></div> : articles.length > 15 ? <p className="pane-foot">Todas as {articles.length} notícias carregadas.</p> : null}
        </Pane>
        <Pane title="Tom por ativo" className="span-12" meta={`nota da carteira ${snapshot.summary.score100 === null ? "— (sem peso coberto)" : `${snapshot.summary.score100}/100`} · cobertura ${formatPercent(snapshot.summary.coverage, 1)} do peso`} flush>
          {!snapshot.articles.length && <Notice>{allFailed ? "Feeds indisponíveis; não foi possível verificar notícias recentes." : failed.length ? "Nenhuma notícia recente nos feeds acessíveis; alguns feeds falharam." : "Nenhuma notícia recente retornada pelos feeds consultados."}</Notice>}
          <Table caption="Tom por ativo" sortable columns={[{key: "ticker", label: "Ativo"}, {key: "state", label: "Feed"}, {key: "count", label: "Artigos", unit: "integer"}, {key: "positive", label: "Otim. (+)", unit: "integer"}, {key: "neutral", label: "Neutr. (0)", unit: "integer"}, {key: "negative", label: "Pessim. (−)", unit: "integer"}, {key: "mean", label: "Média −1 a +1", digits: 2, signed: true}, {key: "covered_share", label: "% do peso coberto", unit: "fraction", digits: 1}, {key: "contribution", label: "Contribuição vs 50", digits: 2, signed: true}]}
            rows={snapshot.summary.tone} />
          <details className="disclosure flush-disclosure"><summary>Como são calculadas a nota e a cobertura</summary><div>
            <p>Cada notícia recebe score entre −1 e +1. Jev usa probabilidade POSITIVE − NEGATIVE, com as três probabilidades somando 100%. O léxico usa (termos positivos − negativos) / total de termos encontrados; sem termos, score 0. As duas rotas não são medidas diretamente comparáveis.</p>
            <p>Para cada ativo calculamos a média simples das notícias, incluindo artigos compartilhados. Score coberto = Σ(peso × média do ativo) / Σ(pesos dos ativos com notícias). Cada peso entra uma vez, independentemente do número de notícias. Contribuição = 50 × peso/peso coberto × média; nota = int(50 + soma das contribuições), de 0 a 100. Ativos sem notícias são excluídos, não tratados como neutros; sem peso coberto não há nota.</p>
            <p>Cobertura = peso dos ativos com notícias / peso total, não confiança. Contagens e notas usam os dez artigos únicos mais recentes por ativo nos últimos sete dias UTC, sem datas futuras, antes dos filtros e paginação. Google e acesso aos publicadores podem limitar os resultados. Importância é triagem por palavras-chave.</p>
            <p>Atualizar notícias limpa os caches compartilhados de RSS e extração para todos os visitantes e libera nova tentativa de carregar o modelo; preserva o modelo já carregado e o cache de sentimento de uma hora, inclusive fallback de falhas individuais. Sem modelo, novas tentativas ocorrem após cinco minutos ou atualização explícita.</p>
          </div></details>
        </Pane>
      </>}
    </div>
  </div>;
}
