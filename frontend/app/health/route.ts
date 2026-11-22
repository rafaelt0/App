export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Readiness for the platform and for uptime monitors. The container port is served by Next.js,
 * so this is the only place that can report both halves of the app in one request.
 *
 * The API is unreachable only while it boots or restarts, and that window is bounded. An API that
 * answers at all is trusted immediately: a 503 from it means the database or the app is broken,
 * not starting. The 3-second probe stays inside Render's 5-second health-check deadline.
 */
const API_TIMEOUT_MS = 3_000;
const STARTUP_GRACE_MS = 60_000;
const bootedAt = Date.now();
let apiHealthyAt: number | null = null;

const noStore = { "Cache-Control": "no-store" };

export async function GET() {
  const base = process.env.B3LAB_API_URL ?? "http://127.0.0.1:8000";
  let response: Response;
  try {
    response = await fetch(new URL("/health", base), { cache: "no-store", signal: AbortSignal.timeout(API_TIMEOUT_MS) });
  } catch (error) {
    if (Date.now() - (apiHealthyAt ?? bootedAt) < STARTUP_GRACE_MS) {
      return Response.json({ status: "starting", api: "unavailable" }, { headers: noStore });
    }
    console.error("B3Lab readiness failed", error);
    return Response.json({ status: "degraded", api: "unavailable" }, { status: 503, headers: noStore });
  }
  try {
    const body: { status?: string } = await response.json();
    if (response.ok && body.status === "ok") {
      apiHealthyAt = Date.now();
      return Response.json({ status: "ok", api: "ok" }, { headers: noStore });
    }
  } catch {}
  return Response.json({ status: "degraded", api: "unhealthy" }, { status: 503, headers: noStore });
}
