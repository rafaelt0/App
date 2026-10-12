let pending = 0;
const listeners = new Set<(active: boolean) => void>();
function track<T>(work: Promise<T>): Promise<T> {
  pending += 1;
  if (pending === 1) listeners.forEach(listener => listener(true));
  return work.finally(() => {
    pending -= 1;
    if (pending === 0) listeners.forEach(listener => listener(false));
  });
}
/** Reports whether any API request is in flight; the shell uses it for its activity line. */
export function subscribeActivity(listener: (active: boolean) => void): () => void {
  listeners.add(listener);
  listener(pending > 0);
  return () => { listeners.delete(listener); };
}

export function api<T>(path: string, body?: unknown, method?: string): Promise<T> {
  return track(request<T>(path, body, method));
}

async function request<T>(path: string, body?: unknown, method?: string): Promise<T> {
  const response = await fetch(path, {
    method: method ?? (body === undefined ? "GET" : "POST"),
    credentials: "same-origin",
    cache: "no-store",
    headers: { "Content-Type": "application/json", "X-B3Lab-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `Não foi possível concluir a solicitação (${response.status}).`;
    try {
      const problem: { detail?: string | { msg: string }[]; message?: string } = await response.json();
      if (typeof problem.detail === "string") message = problem.detail;
      else if (Array.isArray(problem.detail)) message = problem.detail.map(item => item.msg).join(" ");
      else if (problem.message) message = problem.message;
    } catch {}
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function download(path: string): Promise<void> {
  return track(save(path));
}

async function save(path: string): Promise<void> {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  if (!response.ok) {
    const problem: { detail?: string } = await response.json();
    throw new Error(problem.detail ?? "Não foi possível exportar este resultado.");
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = response.headers.get("content-disposition")?.match(/filename="([^"]+)"/)?.[1] ?? "b3lab.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
