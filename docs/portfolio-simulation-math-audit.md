# Auditoria matemática de portfólio e simulação

## Escopo e resultado

Esta auditoria revisou `backend/portfolio.py`, `backend/simulation.py`, `utils/portfolio_data.py`, `utils/portfolio_charts.py` e `utils/simulation.py`, além dos testes determinísticos associados. A fonte de preços ajustados continua alimentando retornos e otimização. A fonte de preços brutos continua alimentando estimativas de cotas.

Foram corrigidos dois defeitos verificáveis:

- Pesos de `clean_weights()` podem somar ligeiramente acima ou abaixo de 1 porque a biblioteca arredonda e corta pesos próximos de zero. A sugestão de compras agora normaliza esses pesos antes de repartir o capital. O cálculo não pode mais recomendar gasto acima do capital por causa desse arredondamento.
- Quando o IBOVESPA está indisponível, o cálculo retornava antes de produzir métricas e diagnósticos que dependem apenas do portfólio. Retorno anualizado, VaR diário, VaR histórico, CVaR, concentração e drawdown agora continuam disponíveis. O CAGR usa o fechamento anterior ao primeiro retorno incluído como início do período, que corresponde ao capital inicial da série.

Os pesos exibidos e persistidos continuam sendo os pesos limpos do otimizador. Para manter o contrato já coberto pelos testes, o retorno histórico continua usando esses valores arredondados sem renormalização. A simulação renormaliza os pesos antes de calcular caminhos. A pequena diferença entre as duas leituras permanece uma convenção intencional; a sugestão de compras, por sua vez, usa pesos normalizados porque precisa caber no orçamento.

## Base matemática e convenções verificadas

- Mínima volatilidade, máximo Sharpe e retorno-alvo usam a fronteira eficiente long-only do PyPortfolioOpt. Retorno esperado e covariância vêm das mesmas linhas completas de retornos diários; a biblioteca recebe os retornos como dados de entrada e anualiza com 252 pregões. O modelo segue a formulação média-variância de Markowitz. O retorno histórico composto usado pelo PyPortfolioOpt para `mean_historical_return` é uma estimativa geométrica, enquanto a variância é a covariância amostral anualizada. Essas são as convenções atuais do projeto, não uma previsão de retorno.
- A série para o otimizador carrega o preço anterior por ativo em falhas internas. Isso registra retorno zero no dia sem cotação e inclui o movimento acumulado quando o preço volta a aparecer. Linhas anteriores ao início da série de qualquer ativo são removidas.
- Os pesos são long-only, não negativos e baseados no capital investido. A modalidade manual calcula o peso pelo número de cotas multiplicado pelo fechamento bruto mais recente.
- Os retornos históricos do portfólio usam rebalanceamento diário implícito: em cada dia, o retorno simples é o produto escalar entre os retornos dos ativos e os pesos fixos. Drawdown é medido contra o capital inicial antes do primeiro retorno.
- Volatilidade e razões Sharpe, Sortino e informação usam fator anual de `sqrt(252)`. O retorno livre de risco histórico usa CDI diário alinhado por data; a indisponibilidade do CDI aciona o fallback Selic atual constante e um aviso. O retorno anualizado geométrico usa dias corridos entre o fechamento inicial e o fechamento final.
- VaR histórico usa o quantil empírico de 5% dos retornos diários. CVaR usa a média dos retornos menores ou iguais a esse quantil. O VaR paramétrico continua separado e depende da aproximação normal já usada pela aplicação.
- A simulação gaussiana ajusta uma normal multivariada aos log-retornos diários dos ativos, transforma cada sorteio em retorno simples com `expm1` e calcula o retorno da carteira diária com os pesos normalizados. Os caminhos representam rebalanceamento diário.
- A comparação por reamostragem sorteia dias históricos completos. Isso preserva a dependência entre ativos no mesmo dia, mas trata dias diferentes como independentes. Não modela agrupamento de volatilidade ou autocorrelação. O gráfico e os percentis da simulação normal usam todas as trajetórias; apenas até 50 caminhos individuais são desenhados.
- Os geradores continuam locais e reprodutíveis com sementes 42 e 43. O tamanho de bloco 64 continua constante para não alterar os resultados determinísticos existentes.

## Limitações explícitas

As duas simulações supõem que as relações estimadas na janela escolhida continuam úteis no horizonte futuro. A normal multivariada também supõe uma covariância estável e caudas gaussianas nos log-retornos. A reamostragem diária ignora dependência temporal. As barras de erro IID e as probabilidades estimadas não corrigem incerteza de seleção de ativos, custos, impostos, liquidez, limites de lote ou mudança de regime. As trajetórias são cenários condicionais aos dados e ao modelo, não previsões.

O peso histórico arredondado e o peso normalizado da simulação podem diferir por alguns milionésimos ou pelo peso cortado próximo de zero. Para compra em dinheiro, os pesos são normalizados e cada posição é arredondada para baixo à cota inteira. O saldo resultante fica em caixa. A análise histórica não modela esse caixa residual nem taxas, impostos ou custos de transação.

## Referências

- Markowitz, H. (1952), *Portfolio Selection*, The Journal of Finance. [DOI do artigo](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- PyPortfolioOpt, [otimização média-variância](https://pyportfolioopt.readthedocs.io/en/stable/MeanVariance.html) e [estimativas de retorno](https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html). A documentação define `clean_weights()` como pesos arredondados e explica o retorno histórico composto e a covariância amostral.
- NumPy, [`Generator.multivariate_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html). A covariância precisa ser simétrica e semidefinida positiva para amostragem válida.
- Künsch, H. R. (1989), *The Jackknife and the Bootstrap for General Stationary Observations*, The Annals of Statistics. [DOI do artigo](https://doi.org/10.1214/aos/1176347265). A reamostragem em blocos é uma abordagem para manter dependência temporal em séries estacionárias; a implementação atual faz reamostragem IID de dias.

## Verificação

Os testes focados reproduziram o gasto acima do capital antes da correção, validaram a normalização da sugestão de compras, confirmaram as métricas independentes do benchmark e preservaram a convenção já existente de pesos limpos nos retornos históricos. A suíte Python completa passou com `PYTHONPATH=. /home/rafael/Projetos/apps/App/.venv/bin/python -m pytest -q`.
