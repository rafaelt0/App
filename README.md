# B3 Explorer — Plataforma Quantitativa de Análise de Ações

## 📈 Descrição

O **B3 Explorer** é uma aplicação web interativa desenvolvida com **Streamlit** para análise quantitativa de ações listadas na B3 (Bolsa de Valores do Brasil). A plataforma reúne, em um único fluxo de trabalho, análise fundamentalista, otimização de portfólio, simulação Monte Carlo, monitoramento de notícias com análise de sentimento, valuation por Enterprise DCF e um screener de mercado.

O projeto integra dados históricos do Yahoo Finance (`yfinance`), indicadores fundamentalistas do Fundamentus e a taxa Selic do Banco Central (`python-bcb`), oferecendo uma ferramenta completa de apoio à decisão de investimento.

🔗 **Acesse a aplicação:** https://b3explorer.streamlit.app/

---

## ⚙️ Funcionalidades

A aplicação é organizada como um **fluxo de análise em 6 etapas**:

### 1. Análise Fundamentalista *(página inicial)*
* Panorama de todo o mercado com dados atualizados do Fundamentus (a cada hora).
* Cartões de preço, indicadores por setor e ranking setorial de ativos.
* Painel de endividamento e watchlist personalizada (favoritos por navegador).

### 2. Portfolio — Análise & Otimização
* Importação de preços históricos ajustados via `yfinance`.
* Otimização de alocação por **Markowitz** (fronteira eficiente) e **Hierarchical Risk Parity (HRP)**, além de alocação manual de pesos.
* Métricas institucionais de risco/retorno: Sharpe, Sortino, VaR, CVaR e Drawdown.
* Visualização do valor do portfólio vs. benchmark (IBOVESPA), beta móvel, Sharpe móvel e drawdown.
* Taxa livre de risco baseada na Selic real.

### 3. Simulação Monte Carlo
* Projeção de trajetórias de retorno com 1.000+ simulações estocásticas.
* Gráfico interativo estilo *fan chart* com faixas de confiança.
* Estatísticas resumo: valor esperado, VaR e cenários extremos.

### 4. Notícias & Sentimento
* Monitoramento de notícias em tempo real das ações da carteira (RSS).
* Análise de sentimento com o modelo de deep learning **FinBERT-PT-BR** (classificação otimista / neutro / pessimista).
* Agregação do impacto qualitativo sobre o portfólio.

### 5. Valuation — McKinsey / Koller
* **Enterprise DCF** completo em 8 etapas: NOPLAT → Invested Capital → ROIC histórico → Projeção → Continuing Value → WACC/CAPM → Enterprise Value → Validação por múltiplos.
* Cálculo de WACC via CAPM, com custo de capital derivado da Selic.
* Validação cruzada por múltiplos de mercado e ranking setorial.

### 6. Screener B3
* Filtragem de todas as ações listadas na B3 por indicadores fundamentalistas.
* Ordenação e busca de candidatos de investimento.
* Dados atualizados a cada hora via Fundamentus.

### Exportação
* Geração e download de relatório completo em HTML usando `quantstats`.

---

## 🛠 Tecnologias e Bibliotecas

* [Streamlit](https://streamlit.io/) — Framework para apps web interativos em Python.
* [yfinance](https://github.com/ranaroussi/yfinance) — Dados financeiros históricos.
* [Fundamentus](https://pypi.org/project/fundamentus/) — Indicadores fundamentalistas da B3.
* [python-bcb](https://github.com/wilsonfreitas/python-bcb) — Séries do Banco Central (Selic).
* [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) — Otimização de portfólio (Markowitz, HRP).
* [QuantStats](https://github.com/ranaroussi/quantstats) — Análise estatística e relatórios financeiros.
* [Transformers](https://huggingface.co/docs/transformers) + [PyTorch](https://pytorch.org/) — Modelo FinBERT-PT-BR para análise de sentimento.
* [Plotly](https://plotly.com/python/) — Visualizações interativas.
* [Matplotlib](https://matplotlib.org/) e [Seaborn](https://seaborn.pydata.org/) — Gráficos estáticos.
* [Pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/) — Manipulação e análise de dados.
* [scikit-learn](https://scikit-learn.org/) e [SciPy](https://scipy.org/) — Suporte numérico e estatístico.
* **SQLite** — Cache persistente, watchlist e portfólio por navegador.

---

## 🚀 Executando localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/rafaelt0/app.git
cd app

# 2. (Opcional) criar um ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Iniciar a aplicação
streamlit run Main_Page.py
```

A aplicação abre em `http://localhost:8501`.

### Testes

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📁 Estrutura do Projeto

```
.
├── Main_Page.py              # Página inicial — Análise Fundamentalista
├── pages/
│   ├── 1_Portfolio.py        # Análise & otimização de portfólio
│   ├── 2_Simulação.py        # Simulação Monte Carlo
│   ├── 3_Notícias.py         # Notícias & análise de sentimento (FinBERT)
│   ├── 4_Valuation.py        # Enterprise DCF (McKinsey / Koller)
│   └── 5_Screener.py         # Screener de ações da B3
├── utils/                    # Módulos de dados, gráficos, valuation e UI
├── tests/                    # Testes com pytest
├── .streamlit/config.toml    # Tema e configurações do Streamlit
├── requirements.txt
└── requirements-dev.txt
```

---

## 📞 Contato

Desenvolvido por **Rafael Eiki Teruya** — [rafael_teruya@usp.br](mailto:rafael_teruya@usp.br)
