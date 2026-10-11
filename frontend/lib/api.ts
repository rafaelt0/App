export async function api<T>(path: string, body?: unknown, method?: string): Promise<T> {
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

export async function download(path: string): Promise<void> {
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
