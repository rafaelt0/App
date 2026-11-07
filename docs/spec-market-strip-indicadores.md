# Spec: faixa de mercado com juro real, Focus e direção dos macro

## Objective

A faixa de mercado (fita no topo e painel "Macro e índices" da Home) deve resumir o que o investidor em ações brasileiras olha para decidir: nível e direção dos juros, inflação esperada, juro real e os preços externos que mais mexem com a B3. Sai o que é redundante; entra o que falta. Nenhum valor é inventado: fonte que falha vira item em `missing`.

### Removidos

| Item | Motivo |
|---|---|
| IBrX 50 (`^IBX50`) | Correlação diária de ~0,99 com o Ibovespa |
| CDI (SGS 4389) | Fica ~0,10 p.p. abaixo da Selic meta |
| Desemprego (SGS 24369) | Sai com meses de atraso e pesa pouco na decisão |

### Adicionados

| Chave | Rótulo | Fonte | Valor | Variação (`change`, p.p.) |
|---|---|---|---|---|
| `ntnb_2035`, `ntnb_2045` | IPCA+ 2035 / 2045 (nota "Tesouro, real a.a.") | Tesouro Transparente, CSV "Taxas dos Títulos Ofertados pelo Tesouro Direto" | Taxa de compra da manhã, % a.a. real | Ante a data-base anterior do CSV |
| `focus_ipca` | IPCA Focus {ano} | BCB Olinda `ExpectativasMercadoAnuais`, `baseCalculo eq 0` | Mediana da última `Data`, % no ano | Ante a mediana de 7 ou mais dias antes (o boletim da semana anterior) |
| `focus_selic` | Selic Focus {ano} | Idem, indicador `Selic` | Mediana da última `Data`, % a.a. no fim do ano | Idem |
| `real_rate` | Juro real ex-ante | Selic meta (SGS 432) e Focus IPCA 12 meses (`ExpectativasMercadoInflacao12Meses`, `Suavizada eq 'S'`, `baseCalculo eq 0`) | Fisher, % a.a. | Nenhuma |
| `spx` | S&P 500 | Yahoo `^GSPC` | Pontos | Fração diária, como as outras cotações |
| `brent` | Brent | Yahoo `BZ=F` | US$ por barril | Idem |

Minério de ferro fica de fora: o Yahoo não tem série confiável.

### Direção dos macro existentes

- **Selic meta:** variação em p.p. desde o último Copom que mudou a taxa, com a data em que a nova taxa passou a valer. A janela do SGS 432 vai a ~4 anos para alcançar a última mudança. Se não houve mudança na janela, `change` fica `None`.
- **IPCA 12 meses:** variação em p.p. ante o mês anterior.

### Decisões

1. **Juro real ex-ante pela fórmula de Fisher**, `(1 + Selic)/(1 + π) − 1`. É a mesma convenção da taxa real do Tesouro IPCA+, então os dois juros reais da faixa ficam comparáveis. A diferença simples superestimaria o juro real em ~0,7 p.p. com Selic de 13,75% e inflação de 5%.
2. **Inflação usada no juro real: Focus IPCA 12 meses à frente (suavizada)**, não o IPCA do fim do ano. Em novembro, o IPCA do ano corrente é quase todo inflação já realizada; o juro real ex-ante precisa da inflação esperada para o horizonte à frente. A série suavizada é a que o BCB usa para essa medida. O item mostra a data do Focus usado.
3. **Variação dos macro em p.p.** Cada item ganha `change_unit` (`"fraction"` nas cotações, `"pp"` nos macro) e `change_note` (texto curto em PT-BR que diz contra o quê a variação foi medida). O frontend formata conforme a unidade.
4. **Tesouro: só o início do CSV.** O arquivo tem ~14 MB e vem ordenado da data-base mais recente para a mais antiga (verificado em 07/11/2026: nenhuma inversão em 178 mil linhas). O servidor responde a um `Range` com o arquivo inteiro (206 com `Content-Range: bytes 0-14667959`), levando de 2 a 20 s. Por isso a busca lê o corpo em streaming e fecha a conexão depois de 128 KB (~1.800 linhas, várias semanas), descartando a última linha cortada. Leva ~0,2 s.
5. **Cache:** cotações 5 min (sem mudança). SGS, Focus e Tesouro 1 h: o Focus diário e o CSV do Tesouro são publicados uma vez por dia útil; 1 h deixa a faixa no máximo uma hora atrás da publicação.
6. **Fita (`Shell.tsx`):** busca de novo ao voltar a ficar visível (`visibilitychange`), além do intervalo de 5 min.

## Tech Stack

Python 3.11, requests, pandas, yfinance, FastAPI. Next.js/React no frontend.

## Commands

```bash
PYTHONPATH=. pytest tests/test_market_strip.py
PYTHONPATH=. pytest
npm --prefix frontend run typecheck
npm --prefix frontend run build
python -c "from utils.market_strip import market_strip; import json; print(json.dumps(market_strip(), ensure_ascii=False, indent=1))"
python run.py
```

## Project Structure

- `utils/market_strip.py`: fontes, parse e itens.
- `backend/market.py`: rota, sem mudança.
- `frontend/components/Shell.tsx`: tipo `StripItem`, `Change`, fita e recarga por visibilidade.
- `frontend/app/HomeBoard.tsx`: painel "Macro e índices" e rodapé de fontes.
- `tests/test_market_strip.py`: testes com rede simulada.

## Testing Strategy

pytest determinístico, com `requests.get` e os fetchers simulados. Para cada fonte nova: sucesso, falha (item em `missing`) e unidade. Mais o cálculo de Fisher, a variação da Selic desde o Copom e o juro real em `missing` quando falta uma das entradas.

## Boundaries

- Always: cálculo e normalização em Python; rótulos em PT-BR; unidade explícita.
- Never: preencher valor de fonte que falhou; adicionar dependência.

## Success Criteria

- A faixa não traz IBrX 50, CDI nem desemprego.
- Traz IPCA+ 2035 e 2045, IPCA Focus e Selic Focus do ano, juro real ex-ante, S&P 500 e Brent, cada um com data de referência.
- Selic meta e IPCA mostram variação em p.p.; o frontend escreve "p.p.".
- Cada fonte nova que falha aparece em `missing`.
- Testes, typecheck e build passam; chamada real a `market_strip()` traz os itens novos.
