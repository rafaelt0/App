"use client";

import { useEffect, useRef, useState } from "react";
import type * as Plotly from "plotly.js";
import { Notice } from "./ui";

export type PlotFigure = { data: Plotly.Data[]; layout?: Partial<Plotly.Layout> };
let library: Promise<typeof Plotly> | null = null;
function loadPlotly() {
  if (!library) library = import("plotly.js-dist-min").then(module => module.default);
  return library;
}

export default function Chart({ figure, title }: { figure: PlotFigure; title?: string }) {
  const node = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let loaded: typeof Plotly | undefined;
    const element = node.current;
    if (!element) return;
    loadPlotly().then(async plotly => {
      loaded = plotly;
      if (disposed) return;
      const layout: Partial<Plotly.Layout> = {
        ...figure.layout,
        autosize: true,
        paper_bgcolor: "#161a22",
        plot_bgcolor: "#161a22",
        font: { ...figure.layout?.font, family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: "#f3f5f9" },
        margin: { l: 48, r: 20, b: 80, ...figure.layout?.margin, t: title ? 32 : figure.layout?.margin?.t ?? 40 },
        ...(title ? { title: { text: "" } } : {}),
      };
      await plotly.react(element, figure.data.map(trace => ({ ...trace })), layout, {
        responsive: true, displaylogo: false, scrollZoom: false,
        modeBarButtons: [
          ["toImage"],
          ["zoom2d", "pan2d", "select2d", "lasso2d"],
          ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        ],
      });
      if (disposed) { plotly.purge(element); return; }
      setError(null);
      observer = new ResizeObserver(() => { if (element.isConnected) void plotly.Plots.resize(element); });
      observer.observe(element);
    }).catch(reason => { if (!disposed) setError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { disposed = true; observer?.disconnect(); if (loaded) loaded.purge(element); };
  }, [figure, title]);
  return <figure className="plotly-panel">{title && <figcaption>{title}</figcaption>}{error ? <Notice tone="negative">Não foi possível renderizar o gráfico. {error}</Notice> : <div ref={node} className="plotly-chart" aria-label={title ?? "Gráfico da análise financeira"} />}</figure>;
}
