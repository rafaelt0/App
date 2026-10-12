import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };
const domains: Record<string, true> = { session: true, universe: true, watchlist: true, home: true, screener: true, portfolio: true, simulation: true, news: true, fundamentus: true, market: true };

async function forward(request: NextRequest, context: Context) {
  const { path } = await context.params;
  if (!Object.hasOwn(domains, path[0]) || path.some(segment => !segment || segment === "." || segment === ".." || segment.includes("/"))) {
    return Response.json({ detail: "Rota indisponível." }, { status: 404 });
  }
  const upstream = new URL(`/api/${path.map(encodeURIComponent).join("/")}`, process.env.B3LAB_API_URL ?? "http://127.0.0.1:8000");
  upstream.search = request.nextUrl.search;
  const headers = new Headers();
  for (const name of ["content-type", "cookie", "origin", "x-b3lab-request", "accept"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const options: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    cache: "no-store",
    redirect: "manual",
    signal: request.signal,
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    options.body = request.body;
    options.duplex = "half";
  }
  try {
    const response = await fetch(upstream, options);
    const outgoing = new Headers({ "Cache-Control": "no-store" });
    for (const name of ["content-type", "content-disposition"]) {
      const value = response.headers.get(name);
      if (value) outgoing.set(name, value);
    }
    for (const value of response.headers.getSetCookie()) outgoing.append("set-cookie", value);
    return new Response(response.body, { status: response.status, headers: outgoing });
  } catch (error) {
    if (request.signal.aborted) return new Response(null, { status: 499 });
    console.error("B3Lab API request failed", error);
    return Response.json({ detail: "A API Python está indisponível. Verifique o servidor e tente novamente." }, { status: 502 });
  }
}

export { forward as GET, forward as POST, forward as PUT, forward as DELETE };
