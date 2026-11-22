"use client";

import Link from "next/link";
import { useEffect } from "react";
import { Notice, Pane } from "../components/ui";

/**
 * Catches a render or data error in any page, so the shell and the navigation stay usable
 * and the visitor gets a retry instead of Next's blank error screen.
 *
 * A full reload is deliberate: reset() re-renders the segment against the same failed server
 * payload, so a recovered backend would never reach the page from here.
 */
export default function AppError({ error }: { error: Error & { digest?: string } }) {
  useEffect(() => {
    console.error("Falha ao renderizar a tela", error);
  }, [error]);
  return (
    <div className="desk">
      <Pane title="Algo falhou nesta tela" meta="Erro inesperado">
        <Notice tone="negative">
          A tela não pôde ser exibida com os dados recebidos. Os favoritos salvos continuam nesta sessão.
        </Notice>
        <div className="idle">
          <p>Tente carregar a tela de novo. Se o erro persistir, volte ao início e refaça a seleção de ativos.</p>
          <div className="param-run">
            <button type="button" className="primary" onClick={() => window.location.reload()}>Tentar novamente</button>
            <Link className="button-link" href="/">Voltar ao início</Link>
          </div>
        </div>
      </Pane>
    </div>
  );
}
