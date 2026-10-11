"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Chart from "../../components/Chart";
import { useApp } from "../../components/AppProvider";
import { EmptyState, Loading, Metrics, Notice, PageHeader, Section, Table } from "../../components/ui";
import "./news.css";
import { api } from "../../lib/api";
import { filterArticles, initialFilters, intensityLevels, safeNewsLink, sortModes } from "../../lib/news";
import type { NewsArticle, NewsFilters, NewsSnapshot } from "../../lib/news";

const decimal = new Intl.NumberFormat("pt-BR", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const signed = new Intl.NumberFormat("pt-BR", {minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "always"});
const percentage = new Intl.NumberFormat("pt-BR", {style: "percent", maximumFractionDigits: 1});

const sentimentPaths = {positive: "M12 5l7 12H5z", negative: "M12 19L5 7h14z", neutral: "M6 12h12"} as const;
function SentimentIcon({tone}: {tone: keyof typeof sentimentPaths}) {
  return <svg width="12" height="12" viewBox="0 0 24 24" fill={tone === "neutral" ? "none" : "currentColor"} stroke="currentColor" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" aria-hidden="true"><path d={sentimentPaths[tone]} /></svg>;
}

function Article({article}: {article: NewsArticle}) {
  const link = safeNewsLink(article.link);
  const tone = article.sentiment === "Otimista" ? "positive" : article.sentiment === "Pessimista" ? "negative" : "neutral";
  const importance = article.importance_rank === 3 ? "is-neutral" : article.importance_rank === 2 ? "is-warning" : "is-muted";
  return <article className="news-item" aria-label={article.title}>
    <div className="news-item-score">
      <span className="news-item-score-label">Score</span>
      <span className={`news-item-score-value is-${tone}`}>{signed.format(article.score)}</span>
    </div>
    <div className="news-item-body">
      <div className="news-item-top">
        {article.tickers.map(ticker => <span className="news-ticker" key={ticker}>{ticker}</span>)}
        <span className={`news-tag is-${tone}`}><SentimentIcon tone={tone} />{article.sentiment}</span>
        <span className={`news-tag ${importance}`}>Importância {article.importance_label}</span>
        <span className="news-tag">Intensidade: {article.intensity}</span>
      </div>
      <h3 className="news-item-title">{link ? <a href={link} target="_blank" rel="noopener noreferrer">{article.title}<span className="sr-only"> (abre em nova aba)</span></a> : article.title}</h3>
      <p className="news-item-meta">{article.provider || "Fonte não informada"} · {article.pub_time} · {article.engine} · Analisado: {article.text_source}</p>
      {article.summary && !article.summary.startsWith(article.title) && <p className="news-item-summary">{article.summary}</p>}
      <details className="news-model-details">
        <summary>Detalhes do algoritmo PLN ({article.is_jev ? "Jev" : "Léxico"} · score {decimal.format(article.score)})</summary>
        {article.is_jev ? <>
          <p className="news-model-heading">Probabilidades — Jev</p>
          {([ ["POSITIVE", "Otimista", "positive"], ["NEGATIVE", "Pessimista", "negative"], ["NEUTRAL", "Neutro", "neutral"] ] as const).map(([label, name, state]) => {
            const score = article.scores.find(item => item.label === label)?.score ?? 0;
            return <div className="news-model-row" key={label}>
              <div className="news-model-row-header"><span>{name} ({label})</span> <span className={`news-model-value is-${state}`}>{percentage.format(score)}</span></div>
              <div className="news-model-track" aria-hidden="true"><div className={`news-model-fill is-${state}`} style={{width: `${score * 100}%`}} /></div>
            </div>;
          })}
          <p className="news-model-note">Score = POSITIVE − NEGATIVE; ≥ 0,20 otimista, ≤ −0,20 pessimista, demais neutro. Scores não somam 100% e não são probabilidades calibradas. Tamanho do texto: {article.raw_text_length} palavras.</p>
        </> : <>
          <p className="news-model-label">Termos positivos encontrados</p>
          {article.pos_terms.length ? article.pos_terms.map((term, index) => <span className="news-term-chip is-positive" key={`${index}-${term}`}>{term} </span>) : <span>Nenhum</span>}
          <p className="news-model-label">Termos negativos encontrados</p>
          {article.neg_terms.length ? article.neg_terms.map((term, index) => <span className="news-term-chip is-negative" key={`${index}-${term}`}>{term} </span>) : <span>Nenhum</span>}
          <p className="news-model-formula">Score = (Pos − Neg) / (Pos + Neg)<br />
            {article.evidence_count ? `(${article.pos_terms.length} − ${article.neg_terms.length}) / ${article.evidence_count} = ${decimal.format(article.score)}` : "Sem termos encontrados: score 0 (sem evidência, não confiança na neutralidade)."}</p>
          <p className="news-model-note">{article.evidence_count} evidências · Tamanho do texto tokenizado: {article.raw_text_length} palavras. Menos de dois termos implica evidência limitada. Frases, negação e repetições são consideradas.</p>
        </>}
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
  const articles = snapshot ? filterArticles(snapshot.articles, filters) : [];
  const visible = articles.slice(0, limit);
  const positive = articles.filter(article => article.sentiment === "Otimista").length;
  const negative = articles.filter(article => article.sentiment === "Pessimista").length;
  const failed = snapshot?.feeds.filter(feed => !feed.ok).map(feed => feed.ticker) ?? [];
  const allFailed = !!snapshot && failed.length === snapshot.feeds.length;
  return <div className="news-page">
    <PageHeader title="Notícias do portfólio" description="Acompanhe eventos recentes e o tom das notícias sobre seus ativos." icon="news" />
    {bootstrapLoading ? <Loading>Restaurando sessão…</Loading> : <>
      {error && <Notice tone="negative">{error}</Notice>}
      {!session?.portfolio_ready && !busy && <div className="news-prereq"><EmptyState icon="news" title={session?.selection.length ? "Atualize o portfólio" : "Portfólio não configurado"} href="/portfolio">
        {session?.selection.length ? "A seleção precisa de uma análise atual. Volte para Portfólio e carregue o portfólio antes de consultar as notícias." : "Configure e carregue seu portfólio para acompanhar os ativos e seus pesos."}
      </EmptyState><aside className="news-prereq-guide" aria-labelledby="news-guide-title">
        <h2 id="news-guide-title">Sua carteira no noticiário</h2>
        <p>Reúna as notícias recentes dos seus ativos em um só lugar.</p>
        <dl>
          <div><dt>Leitura por ativo</dt><dd>Filtre a amostra por empresa, sentimento e intensidade.</dd></div>
          <div><dt>Tom da carteira</dt><dd>Veja a síntese ponderada pelos pesos da alocação carregada.</dd></div>
          <div><dt>Fontes e cobertura</dt><dd>Consulte os links, o motor de análise e a disponibilidade de cada feed.</dd></div>
        </dl>
      </aside></div>}
      {session?.portfolio_ready && <section className="news-task" aria-label="Carteira e atualização">
        {snapshot && <ul className="news-portfolio" aria-label="Carteira e pesos">{snapshot.tickers.map(ticker => <li key={ticker}><strong>{ticker}</strong>{percentage.format(snapshot.weights[ticker])}</li>)}</ul>}
        <div className="news-task-actions">
          <button className="primary" onClick={() => void execute(true)} disabled={busy}>{busy ? "Atualizando…" : "Atualizar notícias"}</button>
          {snapshot && <p className="news-asof">Amostra em {new Date(snapshot.as_of).toLocaleString("pt-BR")}</p>}
          {snapshot && <p className={`news-engine ${snapshot.model.available ? "" : snapshot.articles.length ? "is-fallback" : "is-none"}`}>{snapshot.model.available ? "Jev disponível" : snapshot.articles.length ? "PLN Léxico (fallback)" : "Sem notícias classificadas"}</p>}
        </div>
      </section>}
      {busy && <Loading>Buscando feeds e textos; classificando com Jev ou léxico quando indisponível…</Loading>}
      {snapshot && <>
        <div className="news-feed-status">{snapshot.feeds.map(feed => !feed.ok ? <Notice key={feed.ticker} tone="warning">Feed de notícias indisponível para {feed.ticker}; nenhum item foi usado na análise.</Notice> : !feed.recent_count ? <Notice key={feed.ticker}>Nenhuma notícia recente (últimos 7 dias) encontrada para {feed.ticker}.</Notice> : null)}</div>
        <Section title="Tom da amostra disponível">
          <div className="news-summary">
            <div className="news-summary-main">
              <Metrics items={[
                {label: "Nota do tom da carteira (/100)", value: snapshot.summary.score100, status: snapshot.summary.status, unit: "integer"},
                {label: "Cobertura do peso", value: snapshot.summary.coverage, unit: "fraction"},
                {label: "Artigos únicos otimistas", value: snapshot.summary.counts.positive, unit: "integer", status: "positive"},
                {label: "Artigos únicos neutros", value: snapshot.summary.counts.neutral, unit: "integer"},
                {label: "Artigos únicos pessimistas", value: snapshot.summary.counts.negative, unit: "integer", status: "negative"},
              ]} />
              <p className="news-verdict"><strong>{snapshot.summary.label}</strong><span>Motor: {snapshot.model.engines_used.join(" / ") || "sem classificação"}</span></p>
              <p className="news-score-note">Escala 0–100 do tom textual, não previsão de retorno ou risco. Cobertura é disponibilidade de peso, não confiança. Ativos sem notícias são excluídos; sem peso coberto a nota é —, não 50. Contagens usam artigos únicos antes dos filtros.</p>
            </div>
            <div className="news-summary-chart">
              <Chart figure={snapshot.summary.chart} title="Notícias por ativo" />
              <details><summary>Tabela dos dados do gráfico (contagens por ativo)</summary>
                <Table columns={[{key: "ticker", label: "Ativo"}, {key: "positive", label: "Otimistas", unit: "integer"}, {key: "neutral", label: "Neutras", unit: "integer"}, {key: "negative", label: "Pessimistas", unit: "integer"}]} rows={snapshot.summary.distribution} rowKey="ticker" />
              </details>
            </div>
          </div>
        </Section>
        <Section title="Feed de notícias da carteira">
          <div className="news-intro">
            <p className="news-help">Analisamos até {snapshot.max_per_ticker} notícias únicas mais recentes por ativo nos últimos {snapshot.window_days} dias, antes dos filtros: {snapshot.articles.length} artigos únicos · {snapshot.summary.observation_count} observações artigo-ativo. Feeds indisponíveis: {failed.join(", ") || "nenhum"}.</p>
            <p className="news-help">A cobertura depende do Google News e dos feeds acessíveis. Sem texto acessível usamos descrição RSS ou título. Importância é triagem por palavras-chave, não análise de materialidade. Filtros e “Ver mais” não alteram a nota, o gráfico ou a síntese.</p>
          </div>
          <div className="news-filters" role="group" aria-label="Filtros do feed">
            <label>Filtrar por ativo<select value={filters.ticker} onChange={event => changeFilter("ticker", event.target.value)}><option value="Todos">Todos os ativos</option>{snapshot.tickers.map(ticker => <option key={ticker} value={ticker}>{ticker} ({snapshot.articles.filter(article => article.tickers.includes(ticker)).length} notícias)</option>)}</select></label>
            <label>Ordenar por<select value={filters.sort} onChange={event => changeFilter("sort", event.target.value)}>{sortModes.map(sort => <option key={sort}>{sort}</option>)}</select></label>
            <label>Filtrar por sentimento<select value={filters.sentiment} onChange={event => changeFilter("sentiment", event.target.value)}><option>Todos</option><option value="Otimista">Otimistas</option><option value="Neutro">Neutras</option><option value="Pessimista">Pessimistas</option></select></label>
            <label>Filtrar por intensidade<select value={filters.intensity} onChange={event => changeFilter("intensity", event.target.value)}><option>Todos</option>{intensityLevels.filter(level => level !== "Evidência limitada" || snapshot.articles.some(article => !article.is_jev)).map(level => <option key={level}>{level}</option>)}</select></label>
            <p className="news-help news-filters-note">Faixas de intensidade se baseiam no score, não em materialidade. No léxico, menos de dois termos implica evidência limitada.</p>
          </div>
          <div className="news-results" role="status" aria-live="polite">
            <span>Mostrando <strong>{visible.length}</strong> de <strong>{articles.length}</strong> notícias após filtros</span>
            <span className="news-count"><SentimentIcon tone="positive" />{positive} otimistas</span>
            <span className="news-count"><SentimentIcon tone="neutral" />{articles.length - positive - negative} neutras</span>
            <span className="news-count"><SentimentIcon tone="negative" />{negative} pessimistas</span>
          </div>
          {!articles.length && <Notice>{snapshot.articles.length ? "Nenhuma notícia corresponde aos filtros atuais." : allFailed ? "Feeds indisponíveis; não foi possível verificar notícias recentes." : "Nenhuma notícia recente encontrada nos feeds disponíveis."}</Notice>}
          {visible.length > 0 && <ul className="news-list">{visible.map(article => <li key={article.id}><Article article={article} /></li>)}</ul>}
          {articles.length > limit ? <div className="news-more"><button onClick={() => setLimit(previous => previous + 15)}>Ver mais {Math.min(15, articles.length - limit)} notícias ({articles.length - limit} restantes)</button></div> : articles.length > 15 ? <div className="news-more"><p>Todas as {articles.length} notícias carregadas.</p></div> : null}
        </Section>
        <Section title="Síntese do tom das notícias por ativo">
          <p className="news-help">{snapshot.articles.length} artigos únicos analisados · Cobertura: {percentage.format(snapshot.summary.coverage)} do peso da carteira (não é medida de confiança). Tom textual não mede retorno nem risco financeiro.</p>
          {!snapshot.articles.length && <Notice>{allFailed ? "Feeds indisponíveis; não foi possível verificar notícias recentes." : failed.length ? "Nenhuma notícia recente nos feeds acessíveis; alguns feeds falharam." : "Nenhuma notícia recente retornada pelos feeds consultados."}</Notice>}
          <p className="news-tone-total">Nota do tom da carteira: <strong>{snapshot.summary.score100 === null ? "— (sem peso coberto)" : `${snapshot.summary.score100}/100`}</strong></p>
          <Table columns={[{key: "ticker", label: "Ativo"}, {key: "state", label: "Estado do feed"}, {key: "count", label: "Artigos", unit: "integer"}, {key: "positive", label: "+", unit: "integer"}, {key: "neutral", label: "0", unit: "integer"}, {key: "negative", label: "−", unit: "integer"}, {key: "signed_mean", label: "Média (−1 a +1)"}, {key: "covered_share", label: "% do peso coberto", unit: "fraction"}, {key: "signed_contribution", label: "Contribuição (pontos vs. 50)"}]}
            rows={snapshot.summary.tone.map(row => ({...row, signed_mean: row.mean === null ? null : signed.format(row.mean), signed_contribution: row.contribution === null ? null : signed.format(row.contribution)}))} rowKey="ticker" />
          <details><summary>Como é calculada a nota e a cobertura?</summary>
            <p>Cada notícia recebe score entre −1 e +1. Jev usa probabilidade POSITIVE − NEGATIVE, com as três probabilidades somando 100%. O léxico usa (termos positivos − negativos) / total de termos encontrados; sem termos, score 0. As duas rotas não são medidas diretamente comparáveis.</p>
            <p>Para cada ativo calculamos a média simples das notícias, incluindo artigos compartilhados. Score coberto = Σ(peso × média do ativo) / Σ(pesos dos ativos com notícias). Cada peso entra uma vez, independentemente do número de notícias. Contribuição = 50 × peso/peso coberto × média; nota = int(50 + soma das contribuições), de 0 a 100. Ativos sem notícias são excluídos, não tratados como neutros; sem peso coberto não há nota.</p>
            <p>Cobertura = peso dos ativos com notícias / peso total, não confiança. Contagens e notas usam os dez artigos únicos mais recentes por ativo nos últimos sete dias UTC, sem datas futuras, antes dos filtros e paginação. Google e acesso aos publicadores podem limitar os resultados. Importância é triagem por palavras-chave.</p>
            <p>Atualizar notícias limpa os caches compartilhados de RSS e extração para todos os visitantes e libera nova tentativa de carregar o modelo; preserva o modelo já carregado e o cache de sentimento de uma hora, inclusive fallback de falhas individuais. Sem modelo, novas tentativas ocorrem após cinco minutos ou atualização explícita.</p>
          </details>
        </Section>
      </>}
    </>}
  </div>;
}
