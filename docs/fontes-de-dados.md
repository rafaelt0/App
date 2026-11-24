# Fontes de dados por classe

## Política atual

| Instrumento ou série | Análise e Screener | Carteira e simulação |
| --- | --- | --- |
| Ações B3 | Fundamentus para ficha, pares setoriais e Screener de ações. Se faltar uma ficha, Yahoo Finance para dados disponíveis, sem estimar indicadores ausentes. | Yahoo Finance: histórico ajustado para retornos e fechamento sem ajuste para estimativas de compra. |
| FIIs | Painel específico do Fundamentus para P/VP, proventos, liquidez e Screener. A análise usa a cotação datada e a faixa de 52 semanas do Yahoo em lote quando disponíveis; mantém o painel como fallback de preço e usa a ficha Yahoo quando o fundo não aparece no painel. | Yahoo Finance, com a mesma separação entre ajuste e compra. |
| Ações EUA e REITs | Ficha, Screener de ações EUA e consenso compartilham a resposta Yahoo (incluindo Dívida Líquida/PL, Liquidez Corrente, EV/EBIT e volume médio diário quando disponíveis). Quando o Yahoo falha, Nasdaq fornece cotação, resumo e demonstrações anuais para ações EUA e REITs elegíveis. | Yahoo Finance, histórico em BRL após conversão de USD. |
| ETFs e BDRs | Yahoo para cotação e proventos disponíveis, sem múltiplos corporativos. ETFs de renda fixa listados usam cotação em lote; os elegíveis dos EUA têm fallback de cotação no Nasdaq. | Yahoo para o instrumento listado, não para o ativo subjacente ou um título público usado como referência. |
| Cripto, commodities, índices e FX | Cotação Yahoo em lote, na moeda da fonte. Não há fundamentos corporativos. | Yahoo continua como fonte de pesquisa. Não se estima compra em cotas para a classe global de mercado. |
| Tesouro Direto (`TD-*`) | Tesouro Transparente, PU Compra Manhã. | Tesouro Transparente: PU Base Manhã para marcação a mercado e PU Compra Manhã para compra. |
| CDI, Selic e IPCA observados | Banco Central/SGS: série 12 (CDI diário), 432 (meta Selic), 13522 (IPCA acumulado em 12 meses). | O modelo busca CDI oficial apenas se uma exposição depende da série. Exposições sintéticas mantêm as premissas e os fallbacks já identificados pelo B3Lab; não são cotações, nem séries oficiais de IPCA. |

As cotações em lote são indicativas e podem ter atraso. O Screener ainda usa referências identificadas para classes fora de ações B3 e FIIs. Nenhum provedor de histórico da carteira foi trocado.

## Benchmark preliminar: brapi

Coleta de 23/11/2026, sem credenciais. Para o símbolo público `PETR4`, foram consultados `/api/quote/PETR4` com `range=1mo&interval=1d&modules=financialData,defaultKeyStatistics,summaryProfile` e com `range=6mo&interval=1d&dividends=true&includeRaw=true`.

- **Preços originais:** `rawOpen`, `rawHigh`, `rawLow`, `rawClose` e volume de PETR4 coincidiram com os arquivos [COTAHIST diários da B3](https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/historical-data/equities/historical-quote-data/) para 21 e 24/08/2026 e 18 e 19/11/2026 (mercado à vista, tipo 010). O campo de última negociação indicou o próprio dia 23/11. Isso não mede SLA, atraso intradiário nem cobertura de outros ativos.
- **Proventos e ajustes:** em 21/08 o fechamento original era R$ 44,30 e o ajustado R$ 43,1053; em 24/08, data ex informada pela brapi, ambos eram R$ 42,11. A diferença de R$ 1,1947 não equivale à soma bruta de R$ 1,34814331 dos eventos informados para essa data. Não se validou a regra de tratamento de JCP nem a série em um desdobramento. A [documentação brapi](https://brapi.dev/docs/acoes/historico) avisa que `close` pode já estar ajustado, reserva `includeRaw=true` ao plano Pro do endpoint histórico documentado e permite valores nulos. A resposta pública para PETR4 não comprova essa disponibilidade em produção.
- **Fundamentos:** na [CVM](https://dados.cvm.gov.br/), compararam-se ITR 2026 e DFP 2025 consolidados, Petrobras (`CD_CVM=009512`), escala MIL convertida em reais. A receita TTM da brapi (R$ 589,940 bilhões) coincidiu com a conta 3.01 (2025 + 9M26 − 9M25). A liquidez corrente foi 0,89808637 na brapi e 0,89808634 na CVM (contas 1.01 / 2.01 do ITR de 30/09/2026), compatível com arredondamento. Isso não valida os demais indicadores.
- **Cobertura:** `HGLG11` (FII), `BOVA11` (ETF) e `AAPL34` (BDR) retornaram HTTP 401, sem token. Não há validação dessas classes nem da cobertura ampla de ações. Usar informes mensais CVM para FIIs e dados do fundo e do administrador para ETFs; não comparar BDRs diretamente com múltiplos da companhia subjacente sem tratar a proporção do recibo.
- **Licença:** os [termos da brapi](https://brapi.dev/terms-of-use) restringem redistribuição e serviços de acesso público. É necessária autorização por escrito para o uso pretendido, inclusive exibição externa e exportações. Uma assinatura, por si só, não comprova esses direitos.

## Alternativa de produção para global e câmbio

[Twelve Data](https://twelvedata.com/exchanges) é candidato a teste para ações e ETFs internacionais, cripto, commodities e FX. O catálogo também anuncia B3 em fim de dia, mas não comprova cobertura de FIIs, BDRs, futuros específicos ou ações deslistadas. A chave pública `demo` retornou duas barras diárias de EUR/USD; USD/BRL não estava autorizado nessa chave. Ainda faltam uma amostra autenticada de USD/BRL e dos símbolos globais, semântica de ajustes e horários de fechamento, latência, limites e custo no volume do B3Lab.

Os [termos da Twelve Data](https://twelvedata.com/terms) exigem autorização expressa ou acordo para redistribuição e exibição externa, além das permissões de bolsas pertinentes. O [Yahoo via yfinance](https://github.com/ranaroussi/yfinance#legal-stuff) permanece opção de pesquisa; não é integração oficial nem pressupõe direitos de redistribuição. Não houve contratação.

## Critérios antes de promover uma fonte de carteira

1. **Cobertura e frescor:** testar todos os símbolos habilitados por classe, incluindo ativos com pouca liquidez, alterações de ticker e históricos longos. Comparar sessões, moeda, data efetiva, lacunas, atraso de publicação, latência p95 e falhas sob a carga prevista.
2. **Original e ajustado:** comparar OHLC original com COTAHIST para instrumentos B3; cruzar dividendos/JCP, desdobramentos, grupamentos e datas ex com avisos oficiais. Validar retornos e valores de compra separadamente. Nunca tratar fechamento ajustado como preço de compra nem preencher retornos ausentes com fechamento original.
3. **Fundamentos e FX:** alinhar moeda, escala, período, versão, instrumento e metodologia às fontes oficiais. Ensaio paralelo para USD/BRL com o mesmo calendário; manter câmbio da data da cotação nas compras e tolerância de até quatro dias para o histórico ajustado. Uma referência oficial como PTAX não é intercambiável com preço de mercado sem validação da convenção.
4. **Direitos e continuidade:** obter cobertura contratual para cache, exibição pública, exportação CSV, dados derivados e atribuição. Rodar em paralelo sem publicar a nova fonte; ativá-la apenas após aprovação e com reversão. Ausência de credencial, cobertura ou licença é uma pendência, não um resultado positivo.
