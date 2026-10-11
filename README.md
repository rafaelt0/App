# B3Lab

B3Lab analisa ações brasileiras com uma interface React em Next.js e uma API Python em FastAPI. Os cálculos financeiros continuam em Python. A interface usa o sistema visual fintech-neutral definido em `DESIGN.md`.

## Funcionalidades

- **Fundamentos.** Seleção por empresa e setor, indicadores do Fundamentus, endividamento, comparação setorial, favoritos e consenso de preços-alvo do Yahoo Finance. A seleção exige a ação **Analisar** antes de buscar os detalhes.
- **Portfólio.** Mínima volatilidade, Markowitz com regularização L2, HRP e cotas manuais. Inclui alocação, estimativa de compra por cotas, fronteira eficiente, desempenho histórico, retornos mensais, Sharpe, Sortino, drawdown, beta, alpha, VaR diário, concentração, regime recente e crises históricas. A ação **Carregar portfólio** libera as análises seguintes.
- **Simulação.** De 10 a 3.000 trajetórias, com até 10 anos de horizonte. Compara o modelo normal multivariado com a reamostragem conjunta dos dias históricos. Mostra trajetórias, percentis, distribuição final, sensibilidade da calibração e média da cauda inferior. P5 não é VaR nem perda máxima. Alterar o número de trajetórias exibidas não executa novamente os modelos.
- **Notícias.** Feeds brasileiros dos últimos sete dias, até dez notícias por ativo, deduplicação, classificação com FinBERT-PT-BR e fallback léxico identificado. Filtros e paginação não alteram a nota ponderada da amostra completa. A nota mede tom textual, não retorno esperado.
- **Screener.** Sete pontos de partida, critérios opcionais, ordenação, escolha de colunas, favoritos e envio de ativos para a análise fundamentalista. Os dados vêm do Fundamentus.

As cinco exportações disponíveis são CSV de fundamentos, alocação, resumo da simulação, percentis por data e resultados completos do screener. Não há geração de relatório HTML.

## Executar localmente

Use Python 3.11 e Node.js 26. A instalação JavaScript usa o `package-lock.json` versionado.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
npm --prefix frontend ci
python run.py
```

Abra http://localhost:3000. O comando inicia o Next.js e a API privada em http://127.0.0.1:8000. Ao encerrar o comando, os dois processos terminam.

A lista de ações é local. Fundamentus, Yahoo Finance, Banco Central, Google News e a primeira obtenção do modelo FinBERT precisam de acesso à rede. A indisponibilidade de uma fonte aparece na interface. O fallback léxico não se apresenta como FinBERT.

Para executar os processos separadamente, use dois terminais na raiz do projeto.

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

```bash
npm --prefix frontend run dev
```

## Executar o build de produção

Com as dependências instaladas, execute os comandos na raiz.

```bash
npm --prefix frontend run build
python run.py --production
```

O launcher usa um único worker Python. Os resultados de análise e o recurso FinBERT pertencem a esse processo. Adicionar workers sem compartilhar o estado quebra essa arquitetura.

Para publicar atrás de um proxy HTTPS, mantenha a API privada e configure `B3LAB_ENV=production` e `B3LAB_ORIGIN` com a origem pública exata. A interface e os pedidos `/api` compartilham a mesma origem. O proxy precisa admitir a duração das análises, incluindo o carregamento inicial do modelo. Não há fila de execução nem promessa de implantação serverless.

## Configuração

| Variável | Padrão | Uso |
|---|---|---|
| `B3LAB_PORT` | `3000` | Porta do Next.js no launcher |
| `B3LAB_API_PORT` | `8000` | Porta da API no launcher |
| `B3LAB_API_URL` | `http://127.0.0.1:8000` | Endereço privado usado pelo proxy Next.js |
| `B3LAB_ORIGIN` | `http://localhost:3000` | Origem autorizada para mutações; o launcher adapta a porta |
| `B3LAB_DB` | `b3_data.db` na raiz | Arquivo SQLite gravável e persistente |
| `B3LAB_ENV` | Não definido | `production` exige cookie Secure e a origem configurada |

O launcher configura `B3LAB_API_URL` a partir de `B3LAB_API_PORT` quando não há endereço explícito. Portas personalizadas devem usar uma origem correspondente; `localhost` e `127.0.0.1` são origens distintas.

Mutações exigem a origem autorizada e o cabeçalho `X-B3Lab-Request: 1`. A API não aceita um identificador de usuário em URL ou corpo como autorização.

## Sessões e persistência

A API emite um cookie anônimo HttpOnly e SameSite=Lax, sem `Max-Age`. Em produção, o cookie também é Secure. A sessão pode ser compartilhada por abas e recarregamentos do mesmo navegador. Não é uma conta, e perder o cookie impede recuperar os registros anônimos. A restauração de sessões pelo navegador pode manter cookies de sessão.

O SQLite guarda favoritos e a última seleção com pesos. O registro do token expira após 30 dias de inatividade. Registros antigos da identidade Streamlit não são atribuídos a um novo visitante, e o banco existente não é apagado pela migração.

Resultados pesados são temporários. O processo mantém até 32 contextos ativos e descarta contextos ociosos após duas horas ou sob pressão de capacidade. Uma solicitação em andamento protege seu contexto contra descarte. Após perda de resultados, a carteira salva pode ser restaurada; mudar a seleção exige uma nova análise.

A API admite duas análises pesadas simultâneas. Uma terceira recebe uma resposta explícita de indisponibilidade. Simulação e notícias usam a carteira atual da sessão, não pesos enviados livremente pelo navegador.

## Verificar alterações

```bash
PYTHONPATH=. pytest
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

Os testes Python isolam o banco e as fontes de rede. Com a aplicação em execução, o smoke abaixo usa fontes reais e uma sessão independente.

```bash
python scripts/smoke_api.py --url http://localhost:3000
```

Ele executa portfólio, ambos os modelos de simulação, screener e fundamentos. Verifica os cinco CSVs, unidades numéricas, ordenação, acesso por outra sessão e rejeição de resultados desatualizados. Uma falha de fonte é uma falha do smoke, não um conjunto fictício de dados.

A verificação visual cobre as cinco rotas nas larguras 375, 768, 1024 e 1440, além de teclado, foco, ícones visíveis, controles de 44px, tabelas com rolagem interna e movimento reduzido.

## Estrutura

```text
backend/                  API, sessão e serviços por domínio
frontend/app/             Cinco páginas React e proxy de API
frontend/components/      Navegação, estado e componentes compartilhados
frontend/public/          Marca e ícones SVG
utils/                    Fontes, cálculos financeiros, cache e SQLite
scripts/smoke_api.py       Verificação HTTP com fontes reais
run.py                    Launcher dos dois processos
requirements*.txt         Dependências Python
```

O frontend não refaz cálculos financeiros. As unidades distinguem frações, pontos percentuais, múltiplos e valores monetários. Gráficos usam resultados calculados em Python e renderização Plotly no navegador. A barra de ferramentas permite operações locais, sem envio de gráficos para a nuvem.

## Contato

Desenvolvido por Rafael Eiki Teruya. [rafael_teruya@usp.br](mailto:rafael_teruya@usp.br).
