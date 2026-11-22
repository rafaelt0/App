"use client";

/**
 * Last resort for a failure in the root layout itself, where the app shell is unavailable.
 * It must render its own document, so the styles here stay inline and dependency-free.
 */
export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  return (
    <html lang="pt-BR">
      <body style={{ margin: 0, minHeight: "100dvh", display: "grid", placeItems: "center", background: "#050607", color: "#e8eaed", font: "14px/1.6 system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif" }}>
        <main style={{ maxWidth: "34rem", padding: "2rem" }}>
          <h1 style={{ margin: "0 0 .5rem", fontSize: "1.25rem", fontWeight: 600 }}>O B3Lab não conseguiu carregar</h1>
          <p style={{ margin: "0 0 1.25rem", color: "#b4bac4" }}>
            A estrutura da página falhou ao iniciar. Nenhum dado salvo na sua sessão foi perdido.
          </p>
          <button type="button" onClick={() => window.location.reload()} style={{ padding: ".6rem 1rem", border: 0, borderRadius: 6, background: "#ffb224", color: "#000", font: "inherit", fontWeight: 600, cursor: "pointer" }}>
            Recarregar a página
          </button>
          {error.digest && <p style={{ margin: "1.25rem 0 0", color: "#8a929e", fontSize: 12 }}>Referência do erro: {error.digest}</p>}
        </main>
      </body>
    </html>
  );
}
