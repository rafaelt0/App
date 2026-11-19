# B3Lab

B3Lab analisa ações brasileiras com uma interface React em Next.js e uma API Python em FastAPI. Os cálculos financeiros continuam em Python. A interface usa o sistema visual fintech-neutral definido em `DESIGN.md`.

## Funcionalidades

- **Fundamentos.** Seleção por empresa e setor, indicadores do Fundamentus, endividamento, comparação setorial, favoritos e consenso de preços-alvo do Yahoo Finance. A seleção exige a ação **Analisar** antes de buscar os detalhes.
- **Portfólio.** Mínima volatilidade, Markowitz com regularização L2, HRP e cotas manuais. Inclui alocação, estimativa de compra por cotas, fronteira eficiente, desempenho histórico, retornos mensais, Sharpe, Sortino, drawdown, beta, alpha, VaR diário, concentração, regime recente e crises históricas. A ação **Carregar portfólio** libera as análises seguintes.
- **Simulação.** De 10 a 3.000 trajetórias, com até 10 anos de horizonte. Compara o modelo normal multivariado com a reamostragem conjunta dos dias históricos. Mostra trajetórias, percentis, distribuição final, sensibilidade da calibração e média da cauda inferior. P5 não é VaR nem perda máxima. Alterar o número de trajetórias exibidas não executa novamente os modelos.
- **Notícias.** Feeds brasileiros dos últimos sete dias, até dez notícias por ativo, deduplicação, classificação com Jev via API e fallback léxico identificado. Filtros e paginação não alteram a nota ponderada da amostra completa. A nota mede tom textual, não retorno esperado.
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

A lista de ações é local. Fundamentus, Yahoo Finance, Banco Central, Google News e a API do Jev precisam de acesso à rede. A indisponibilidade de uma fonte aparece na interface. O fallback léxico não se apresenta como Jev.

Exporte `TYPESAFE_API_KEY` no ambiente antes de iniciar a aplicação. O backend envia o título e até 12.000 caracteres do texto combinado à [API TypeSafe](https://docs.typesafe.ai/api), usando `jev-latest`. Probabilidades positivas menos negativas produzem o score. Resultados válidos ficam em cache por uma hora; falhas usam o léxico e suspendem novas chamadas por cinco minutos na sessão. “Atualizar notícias” libera nova tentativa imediatamente.

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

O launcher usa um único worker Python. Os resultados de análise pertencem a esse processo. Adicionar workers sem compartilhar o estado quebra essa arquitetura.

Para publicar atrás de um proxy HTTPS, mantenha a API privada e configure `B3LAB_ENV=production` e `B3LAB_ORIGIN` com a origem pública exata. A interface e os pedidos `/api` compartilham a mesma origem. O proxy precisa admitir a duração das análises, incluindo as chamadas à API do Jev. Não há fila de execução nem promessa de implantação serverless.

### Render

O `render.yaml` configura o serviço web Docker com a instância paga `0.5c-512mb` (Starter), que continua ativa sem tráfego. O disco persistente de 1 GB é montado em `/app/storage` e `B3LAB_DB` aponta para `/app/storage/b3_data.db`. O contêiner usa `PORT` e `RENDER_EXTERNAL_URL` como porta e origem pública. O Next.js atende externamente e a API continua em `127.0.0.1`, com um único worker.

Configure `TYPESAFE_API_KEY` como segredo no Render para classificar notícias com Jev. O contêiner não carrega modelos locais. Sem a chave ou quando a API falha, as notícias usam o fallback léxico identificado.

1. Confirme no painel que a mudança será aplicada ao serviço existente (`srv-db5domo473hc73app7mg`) e mantenha a URL pública atual. Antes de alterar o plano, anexar o disco ou sincronizar o Blueprint, obtenha autorização para a cobrança e resolva a migração dos dados. O sistema de arquivos do plano gratuito é apagado ao entrar em repouso ou ao fazer deploy, inclusive na troca de plano. O plano gratuito não oferece shell e o B3Lab não oferece exportação desse banco. Se não houver cópia, confirme com o suporte do Render se é possível recuperar o arquivo da instância atual.
2. Se houver dados a preservar, programe uma janela de manutenção, bloqueie novas gravações antes da cópia final e mantenha o bloqueio até a troca. Antes de iniciar a troca, verifique fora da instância um backup consistente, como o gerado pela API de backup do SQLite. Se alguma etapa for inviável, obtenha autorização explícita para a janela de perda de dados identificada ou para iniciar com um banco vazio, conforme o caso.
3. Para restaurar uma cópia, mantenha o PR sem merge e, se houver um Blueprint vinculado, desative temporariamente o `Auto Sync`. Provisione o plano pago e o disco pelo painel, mantendo o caminho de banco em uso. Transfira a cópia para `/app/storage/b3_data.db` pelo Shell do painel com `magic-wormhole`, conforme a documentação do Render. Compare o hash da cópia com o backup e execute `PRAGMA integrity_check` em modo somente leitura antes de mudar `B3LAB_DB`. Se a preparação do disco ou a verificação da cópia falhar, suspenda a publicação. Após a verificação, faça o merge aprovado e sincronize o Blueprint, se vinculado, restaurando a configuração anterior de `Auto Sync` após confirmar o deploy. Em um serviço gerenciado pelo painel, aplique `B3LAB_DB` no serviço existente antes do merge que remove os pings. Para uma instalação vazia autorizada, o plano, o disco e a variável podem ser aplicados juntos. Deploys com disco exigem uma breve interrupção. Consulte as instruções do [plano gratuito](https://render.com/docs/free), do [Auto Sync](https://render.com/docs/infrastructure-as-code#disabling-automatic-sync) e da [transferência de arquivos para o disco](https://render.com/docs/disks#transferring-files).
4. Após publicar, confirme no painel que o deploy concluído usa a instância paga e que o disco está montado. Num navegador novo com largura de celular, cronometre da navegação até a tabela de liquidez e o mapa de calor estarem preenchidos. Confira as respostas HTTP de `/api/session`, `/api/universe`, `/api/universe/liquidity`, `/api/universe/changes` e `/api/market/strip`. Repita a visita em outra sessão nova após mais de 15 minutos sem tráfego público, confirmado na métrica de volume de requisições ou nos logs HTTP quando disponíveis. Compare os tempos e confirme que a tela de inicialização do Render não aparece. Para validar o disco, confirme que um favorito de teste permanece após um reinício usando o mesmo cookie. Consulte as [métricas de requisições do Render](https://render.com/docs/service-metrics#http-requests).

## Configuração

| Variável | Padrão | Uso |
|---|---|---|
| `TYPESAFE_API_KEY` | Não definido | Chave da API TypeSafe, usada somente no backend para Jev |
| `B3LAB_PORT` | `3000` | Porta do Next.js no launcher |
| `B3LAB_API_PORT` | `8000` | Porta da API no launcher |
| `B3LAB_API_URL` | `http://127.0.0.1:8000` | Endereço privado usado pelo proxy Next.js |
| `B3LAB_ORIGIN` | `http://localhost:3000` | Origem autorizada para mutações; o launcher adapta a porta |
| `B3LAB_DB` | `b3_data.db` na raiz | Arquivo SQLite gravável. O Blueprint do Render define `/app/storage/b3_data.db` |
| `B3LAB_ENV` | Não definido | `production` exige cookie Secure e a origem configurada |
| `B3LAB_HOST` | `127.0.0.1` | Endereço de escuta do Next.js no launcher |

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
