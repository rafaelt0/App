export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Readiness for the platform and for uptime monitors. The container port is served by Next.js,
 * so this is the only place that can report both halves of the app in one request.
 *
 * A cold boot and a supervised API restart both take tens of seconds, and Render restarts an
 * instance after 60 seconds of failed checks. The grace below keeps those normal starts from
 * becoming a restart loop while still failing an API that never comes back.
 */
const STARTUP_GRACE_MS = 180_000;
const bootedAt = Date.now();
let apiHealthyAt: number | null = null;

export async function GET() {
  const base = process.env.B3LAB_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(new URL("/health", base), { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`API respondeu ${response.status}`);
    await response.json();
    apiHealthyAt = Date.now();
    return Response.json({ status: "ok", api: "ok" }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (Date.now() - (apiHealthyAt ?? bootedAt) < STARTUP_GRACE_MS) {
      return Response.json({ status: "starting", api: "unavailable" }, { headers: { "Cache-Control": "no-store" } });
    }
    console.error("B3Lab readiness failed", error);
    return Response.json({ status: "degraded", api: "unavailable" }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
