"use client";

import { useEffect, useRef, useState } from "react";
import type * as Plotly from "plotly.js";
import { formatPercent, formatValue, Notice } from "./ui";

export type PlotFigure = { data: Plotly.Data[]; layout?: Partial<Plotly.Layout> };
let library: Promise<typeof Plotly> | null = null;
function loadPlotly() {
  if (!library) library = import("plotly.js-dist-min").then(module => module.default);
  return library;
}

/* Presentation-only theme shared by every server figure; keep in sync with the tokens in globals.css. */
const ink = { text: "#f3f5f9", secondary: "#c5cad6", muted: "#9aa2b4", canvas: "#0e1015", raised: "#23262f" };
const axisStyle: Partial<Plotly.LayoutAxis> = {
  gridcolor: "rgba(255, 255, 255, 0.06)", griddash: "solid", gridwidth: 1, zerolinecolor: "rgba(197, 202, 214, 0.3)", zerolinewidth: 1,
  linecolor: "rgba(255, 255, 255, 0.12)", ticks: "", automargin: true, tickfont: { size: 11, color: ink.muted },
};
const spikes: Partial<Plotly.LayoutAxis> = {
  showspikes: true, spikemode: "across", spikesnap: "cursor", spikethickness: 1, spikedash: "solid", spikecolor: "rgba(197, 202, 214, 0.35)",
};
const dateLike = /^\d{4}-\d{2}-\d{2}/;
type Trace = Plotly.Data & { x?: unknown[]; y?: unknown[]; mode?: string; fill?: string; name?: string; showlegend?: boolean; visible?: unknown; line?: { color?: string }; marker?: { color?: unknown }; hovertemplate?: string; yaxis?: string };

function titleText(title: Partial<Plotly.Layout>["title"]): string | undefined {
  if (!title) return undefined;
  return typeof title === "string" ? title : title.text;
}

function isTimeSeries(data: Trace[]): boolean {
  return data.some(trace => Array.isArray(trace.x) && typeof trace.x[0] === "string" && dateLike.test(trace.x[0] as string));
}

function tagText(value: number, percent: boolean): string {
  if (percent) return formatPercent(value, 1);
  return formatValue(value, "number", Math.abs(value) >= 1000 ? 0 : 2);
}

/** Horizontal category bars shrink on narrow screens so each row stays readable. */
function responsiveHeight(figure: PlotFigure, width: number): number | undefined {
  const base = figure.layout?.height;
  const data = figure.data as Trace[];
  const horizontal = data.some(trace => trace.type === "bar" && trace.orientation === "h");
  if (!base || !horizontal || width >= 600) return base;
  const rows = Math.max(0, ...data.map(trace => (trace.y ?? []).length));
  return Math.min(base, Math.max(240, 120 + rows * 28));
}

/** Last-value tags on the right axis for a few line series, as in trading terminals. */
function lastValueTags(data: Trace[], layout: Partial<Plotly.Layout>): Partial<Plotly.Annotations>[] {
  const lines = data.filter(trace => (trace.mode ?? "lines").includes("lines") && trace.visible !== false && trace.visible !== "legendonly" && trace.showlegend !== false && (!trace.fill || trace.fill === "none" || trace.fill === "tozeroy") && (!trace.yaxis || trace.yaxis === "y"));
  if (!lines.length || lines.length > 3) return [];
  const yaxis = (layout.yaxis ?? {}) as Partial<Plotly.LayoutAxis>;
  return lines.flatMap(trace => {
    const values = (trace.y ?? []) as unknown[];
    const last = [...values].reverse().find(value => typeof value === "number" && Number.isFinite(value)) as number | undefined;
    if (last === undefined) return [];
    const percent = String(yaxis.tickformat ?? "").includes("%") || /:[^}]*%}/.test(trace.hovertemplate ?? "");
    const color = trace.line?.color ?? (typeof trace.marker?.color === "string" ? trace.marker.color : "#8b93ff");
    return [{ xref: "paper", x: 1, xanchor: "left", xshift: 6, yref: "y", y: last, text: tagText(last, percent), showarrow: false,
      bgcolor: color, borderpad: 3, font: { family: "var(--font-mono), monospace", size: 11, color: ink.canvas } } as Partial<Plotly.Annotations>];
  });
}

function themed(figure: PlotFigure, family: string, hasCaption: boolean, coarse: boolean, height: number | undefined): Partial<Plotly.Layout> {
  const layout = figure.layout ?? {};
  const data = figure.data as Trace[];
  const series = isTimeSeries(data);
  const source = layout as Record<string, unknown>;
  const axes: Record<string, Partial<Plotly.LayoutAxis>> = Object.fromEntries(Object.keys(source).filter(key => /^[xy]axis\d*$/.test(key)).map(key => {
    const axis = (source[key] ?? {}) as Partial<Plotly.LayoutAxis> & { title?: unknown };
    const axisTitle = typeof axis.title === "string" ? { text: axis.title } : (axis.title as object | undefined);
    const horizontal = key.startsWith("x");
    return [key, {
      ...axis, ...axisStyle, ...(horizontal ? { ...spikes, showgrid: false, showline: true } : { showline: false }),
      ...(!horizontal && series && key === "yaxis" ? { side: "right" as const } : {}),
      tickfont: { ...axis.tickfont, ...axisStyle.tickfont },
      title: axisTitle ? { ...axisTitle, font: { size: 11.5, color: ink.muted }, standoff: 10 } : undefined,
    }];
  }));
  if (!("xaxis" in axes)) axes.xaxis = { ...axisStyle, ...spikes, showgrid: false, showline: true };
  if (!("yaxis" in axes)) axes.yaxis = { ...axisStyle, showline: false, ...(series ? { side: "right" as const } : {}) };
  const legendShown = layout.showlegend !== false && data.filter(trace => trace.showlegend !== false).length > 1;
  const tags = series ? lastValueTags(data, layout) : [];
  const top = Math.max(legendShown ? 36 : 12, (layout.margin?.t ?? 0) - 50);
  return {
    ...layout,
    ...axes,
    template: undefined,
    autosize: true,
    height,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    separators: ",.",
    font: { ...layout.font, family, size: 12, color: ink.secondary },
    // Touch screens get no drag-zoom and nearest-point hover; values stay readable from the tap target.
    dragmode: coarse ? false : layout.dragmode,
    hovermode: coarse ? "closest" : layout.hovermode ?? (series ? "x unified" : "closest"),
    hoverlabel: { bgcolor: ink.raised, bordercolor: "rgba(255, 255, 255, 0.16)", align: "left", namelength: -1, font: { family, size: 12.5, color: ink.text } },
    legend: { ...layout.legend, orientation: "h", x: 0, xanchor: "left", y: 1, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)", borderwidth: 0, font: { size: 12, color: ink.secondary }, itemsizing: "constant" },
    bargap: layout.bargap ?? .3,
    annotations: [...(layout.annotations ?? []), ...tags],
    margin: { l: 12, r: tags.length ? 72 : 12, b: 44, pad: 4, t: top },
    ...(hasCaption ? { title: { text: "" } } : { title: { ...(typeof layout.title === "object" ? layout.title : {}), text: titleText(layout.title), font: { size: 14, color: ink.text }, x: 0, xanchor: "left" } }),
  };
}

export default function Chart({ figure, title }: { figure: PlotFigure; title?: string }) {
  const node = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);
  const caption = title ?? titleText(figure.layout?.title);
  const series = (figure.data as Trace[]).map(trace => trace.name).filter(Boolean).join(", ");
  const label = [caption ?? "Gráfico da análise financeira", series && `Séries: ${series}`].filter(Boolean).join(". ");
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
      const coarse = window.matchMedia("(pointer: coarse)").matches;
      let height = responsiveHeight(figure, element.clientWidth);
      const layout = themed(figure, getComputedStyle(element).fontFamily, !!caption, coarse, height);
      await plotly.react(element, figure.data.map(trace => ({ ...trace })), layout, {
        responsive: true, displaylogo: false, scrollZoom: false,
        modeBarButtons: [
          ["toImage"],
          ["zoom2d", "pan2d"],
          ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        ],
      });
      if (disposed) { plotly.purge(element); return; }
      setError(false);
      observer = new ResizeObserver(() => {
        if (!element.isConnected) return;
        const next = responsiveHeight(figure, element.clientWidth);
        if (next !== undefined && next !== height) { height = next; void plotly.relayout(element, { height: next }); }
        void plotly.Plots.resize(element);
      });
      observer.observe(element);
    }).catch(reason => {
      // The raw error stays in the console; the page shows a plain message only.
      console.error(reason);
      if (!disposed) setError(true);
    });
    return () => { disposed = true; observer?.disconnect(); if (loaded) loaded.purge(element); };
  }, [figure, caption]);
  return <figure className="plotly-panel">{caption && <figcaption>{caption}</figcaption>}{error ? <Notice tone="negative">Não foi possível renderizar o gráfico.</Notice> : <div ref={node} className="plotly-chart" role="img" aria-label={label} />}</figure>;
}
