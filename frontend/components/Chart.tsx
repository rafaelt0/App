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

/* Presentation-only theme shared by every server figure; keep in sync with the tokens in globals.css. */
const ink = { text: "#f3f5f9", secondary: "#c5cad6", muted: "#9aa2b4", rule: "#2b313d", raised: "#252b37", strong: "#5e687e" };
const grid = "rgba(154, 162, 180, 0.14)";
const axisStyle: Partial<Plotly.LayoutAxis> = {
  gridcolor: grid, griddash: "dot", gridwidth: 1, zerolinecolor: "rgba(197, 202, 214, 0.32)", zerolinewidth: 1,
  linecolor: ink.rule, showline: true, ticks: "outside", ticklen: 4, tickcolor: ink.rule, automargin: true,
  tickfont: { size: 11, color: ink.muted },
};
const spikes: Partial<Plotly.LayoutAxis> = {
  showspikes: true, spikemode: "across", spikesnap: "cursor", spikethickness: 1, spikedash: "dot", spikecolor: "rgba(197, 202, 214, 0.5)",
};

function titleText(title: Partial<Plotly.Layout>["title"]): string | undefined {
  if (!title) return undefined;
  return typeof title === "string" ? title : title.text;
}

function themed(layout: Partial<Plotly.Layout> = {}, family: string, hasCaption: boolean): Partial<Plotly.Layout> {
  const source = layout as Record<string, unknown>;
  const axes: Record<string, Partial<Plotly.LayoutAxis>> = Object.fromEntries(Object.keys(source).filter(key => /^[xy]axis\d*$/.test(key)).map(key => {
    const axis = (source[key] ?? {}) as Partial<Plotly.LayoutAxis> & { title?: unknown };
    const axisTitle = typeof axis.title === "string" ? { text: axis.title } : (axis.title as object | undefined);
    return [key, {
      ...axis, ...axisStyle, ...(key.startsWith("x") ? spikes : {}),
      tickfont: { ...axis.tickfont, ...axisStyle.tickfont },
      title: axisTitle ? { ...axisTitle, font: { size: 12, color: ink.muted }, standoff: 12 } : undefined,
    }];
  }));
  if (!("xaxis" in axes)) axes.xaxis = { ...axisStyle, ...spikes };
  if (!("yaxis" in axes)) axes.yaxis = { ...axisStyle };
  const legend = layout.legend ?? {};
  return {
    ...layout,
    ...axes,
    template: undefined,
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    separators: ",.",
    font: { ...layout.font, family, size: 12, color: ink.secondary },
    hoverlabel: { bgcolor: ink.raised, bordercolor: ink.strong, align: "left", namelength: -1, font: { family, size: 13, color: ink.text } },
    legend: { ...legend, bgcolor: "rgba(0,0,0,0)", borderwidth: 0, font: { size: 12, color: ink.secondary }, itemsizing: "constant" },
    margin: { l: 56, r: 20, b: 64, ...layout.margin, t: hasCaption ? 16 : layout.margin?.t ?? 40 },
    ...(hasCaption ? { title: { text: "" } } : { title: { ...(typeof layout.title === "object" ? layout.title : {}), text: titleText(layout.title), font: { size: 14, color: ink.text }, x: 0, xanchor: "left" } }),
  };
}

export default function Chart({ figure, title }: { figure: PlotFigure; title?: string }) {
  const node = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const caption = title ?? titleText(figure.layout?.title);
  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let loaded: typeof Plotly | undefined;
    const element = node.current;
    if (!element) return;
    loadPlotly().then(async plotly => {
      loaded = plotly;
      await document.fonts.ready;
      if (disposed) return;
      const layout = themed(figure.layout, getComputedStyle(element).fontFamily, !!caption);
      await plotly.react(element, figure.data.map(trace => ({ ...trace })), layout, {
        responsive: true, displaylogo: false, scrollZoom: false,
        modeBarButtons: [
          ["toImage"],
          ["zoom2d", "pan2d"],
          ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        ],
      });
      if (disposed) { plotly.purge(element); return; }
      setError(null);
      observer = new ResizeObserver(() => { if (element.isConnected) void plotly.Plots.resize(element); });
      observer.observe(element);
    }).catch(reason => { if (!disposed) setError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { disposed = true; observer?.disconnect(); if (loaded) loaded.purge(element); };
  }, [figure, caption]);
  return <figure className="plotly-panel">{caption && <figcaption>{caption}</figcaption>}{error ? <Notice tone="negative">Não foi possível renderizar o gráfico. {error}</Notice> : <div ref={node} className="plotly-chart" aria-label={caption ?? "Gráfico da análise financeira"} />}</figure>;
}
