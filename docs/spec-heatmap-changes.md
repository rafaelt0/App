# Spec: variação do mapa de calor em dias sem pregão

## Objective

No sábado, 07/11/2026, o mapa de calor da Home mostrou "s/d" na variação. Em fim de semana ou feriado, o mapa precisa mostrar a variação do último pregão fechado (sexta, 06/11), e o rodapé precisa dizer de qual pregão ela é. "s/d" deve aparecer só para ações que de fato não negociaram nos pregões usados no cálculo.

### O que já foi verificado (07/11, ~12:50)

- O caminho é `HomeBoard.tsx` (`SectorMap`) → `GET /api/universe/changes` → `backend/home.py::universe_changes` → `utils/market_data.py::get_universe_changes` / `universe_changes`.
- O cálculo não depende do dia da semana. Ele usa as duas últimas linhas do download do Yahoo (`iloc[-1]` e `iloc[-2]`), e só remove linhas em que todos os tickers são NaN (`dropna(how="all")`).
- Com dados ao vivo, a rota passando pelo FastAPI respondeu `session = 2026-11-06`, com 332 tickers. Só 5 vieram `null`: AXIA6, EMAE4, OIBR3, OIBR4 e RPMG3, que são deslistadas ou estão sem cotação no Yahoo. **Neste horário, o defeito não se reproduz.**

- Captura do deploy (`b3lab-y4ha.onrender.com`, 07/11 12:57 BRT): quase todos os blocos mostravam "s/d". A cotação da faixa e do monitor, que vem de outras requisições, estava normal.
- Às 15:58 UTC, um minuto depois, `GET /api/universe/changes` no próprio deploy devolveu dados corretos (`session` 06/11, 5 nulos, `fetched_at` 15:58:21, ou seja, uma busca nova). Ao recarregar a página, o mapa ficou certo: 326 de 328 blocos com variação e rodapé "pregão de 06/11".
- **Conclusão:** a falha é intermitente. Uma resposta do Yahoo quase vazia foi tratada como sucesso, guardada pelo `@cache_data(ttl=600)` e mostrada até a página ser recarregada. O sábado não é a causa direta, porque o cálculo não olha o dia da semana.

### Causas possíveis da resposta quase vazia (o rodapé da captura não aparece, então não dá para separar as duas)

1. **Limite de requisições do Yahoo (mais provável no Render, que usa IP compartilhado).** `yf.download` não levanta exceção quando parte dos símbolos falha: esses símbolos vêm como colunas inteiras em NaN. Se só alguns símbolos voltam, `downloaded` não fica vazio, e todos os outros ficam `null`. Nesse caso o rodapé mostraria "pregão de 06/11".
2. **Linha parcial de "hoje".** O Yahoo às vezes devolve uma linha com a data de hoje preenchida para poucos símbolos. Ela não é removida, porque não está toda em NaN, e passa a ser `iloc[-1]`. Nesse caso o rodapé mostraria "pregão de 07/11".

## Mudança proposta

Em `utils/market_data.py::universe_changes`, duas proteções:

1. **Descartar linhas que não são pregão:** remover as linhas com cotação para menos de 50% do máximo de símbolos cotados em uma linha do download (corrige a causa 2).
2. **Não aceitar um resultado quebrado como sucesso:** se, depois disso, menos de 50% dos símbolos pedidos tiverem cotação no último pregão, levantar `ValueError`. A rota já transforma isso em 503 com a mensagem "Não foi possível carregar as variações do Yahoo Finance: …". Como a exceção não é cacheada, a próxima requisição tenta de novo, e o frontend já mostra "Variações indisponíveis: …" no rodapé em vez de pintar o mapa inteiro de "s/d" (corrige a causa 1).

A regra "nunca medir por cima de um buraco" continua valendo por ticker. Não mudam a fonte, a unidade (fração), o TTL nem o frontend.

## Tech Stack

Python 3.11, pandas, yfinance 0.2.65, FastAPI. Frontend Next.js, sem mudanças.

## Commands

```bash
PYTHONPATH=. pytest tests/test_home_api.py -k universe_changes
PYTHONPATH=. pytest
python scripts/smoke_api.py
```

## Project Structure

- `utils/market_data.py`: `universe_changes` (a correção entra aqui)
- `tests/test_home_api.py`: fixture `adjusted_closes` e testes de `universe_changes`

## Code Style

Seguir o estilo do arquivo: funções curtas, docstring explicando a regra, sem novas dependências.

```python
adjusted = adjusted.dropna(how="all")
quoted = adjusted.notna().sum(axis=1)
adjusted = adjusted[quoted >= _MIN_SESSION_COVERAGE * quoted.max()]
if adjusted.empty or adjusted.iloc[-1].notna().sum() < _MIN_SESSION_COVERAGE * len(tickers):
    raise ValueError("Yahoo Finance devolveu cotações para poucas ações.")
```

## Testing Strategy

pytest, determinístico, sem rede (mesmo padrão de `adjusted_closes()`):

1. Regressão: fixture com sessões de 02 a 06/11 completas e uma linha em 07/11 com valor só para um símbolo. Resultado esperado: `session == "2026-11-06"` e `day`/`week` calculados para todos os tickers.
2. Os testes atuais de buraco por ticker (BBB4, CCC3, DDD3) continuam passando sem alteração.
3. Regressão do limite de requisições: fixture em que só 1 de 4 símbolos tem cotação (os outros em NaN). Resultado esperado: `ValueError`, a rota responde 503 e nada fica no cache.
4. Download vazio: continua levantando `ValueError` e a rota responde 503.

## Boundaries

- Always: manter variações em fração, no Python; manter o rodapé com a data real do pregão; rodar o pytest completo.
- Ask first: mudar o `period` de 15 dias, o TTL do cache ou a fonte (Yahoo); mudar algo no frontend.
- Never: preencher variação com 0 ou com valor inventado; medir variação por cima de um pregão faltante; calcular a variação no React.

## Success Criteria

- Quando o Yahoo devolve cotação para menos da metade das ações, a rota responde 503 em vez de 200, e o mapa mostra "Variações indisponíveis: …" no rodapé em vez de "s/d" em todos os blocos. A tentativa seguinte busca de novo, sem esperar os 10 minutos do cache.
- Em sábado, domingo ou feriado, com o Yahoo devolvendo uma linha parcial de "hoje", a rota responde `session` = último pregão completo e não deixa `null` para tickers que negociaram nesse pregão.
- "s/d" fica restrito aos tickers sem cotação nos pregões usados (hoje, os 5 listados acima).
- Novo teste de regressão passando, e os testes atuais sem alteração.

## Open Questions

1. O limite de 50% (tanto para uma linha contar como pregão quanto para aceitar o resultado) está bom?
2. Quando a variação falhar, basta o rodapé "Variações indisponíveis: …" com os blocos neutros, ou você quer também um botão para tentar de novo no mapa? Hoje é preciso recarregar a página. O botão mudaria o frontend, que está em "Ask first".
