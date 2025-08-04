# B3 Explorer
Claro! Aqui está um exemplo de README profissional e completo para seu projeto de análise e simulação de portfólio com Streamlit e Python:

---

# B3 Explorer - Análise e Simulação de Portfólio

## 📈 Descrição

O **B3 Explorer** é uma aplicação web interativa desenvolvida com **Streamlit** que permite ao usuário realizar análises detalhadas, otimização e simulação Monte Carlo de portfólios de ações listadas na B3 (Bolsa de Valores do Brasil).

O projeto integra dados históricos do Yahoo Finance, indicadores financeiros e métodos avançados de alocação de ativos para oferecer uma ferramenta completa de apoio à decisão de investimento.

---

## ⚙️ Funcionalidades

* **Análise de Portfólio:**

  * Importação de dados históricos ajustados de preços das ações da B3 via `yfinance`.
  * Cálculo de retornos, volatilidade, correlação entre ativos.
  * Otimização da alocação de ativos pelo método Hierarchical Risk Parity (HRP).
  * Alocação manual de pesos pelo usuário.
  * Visualização gráfica dos valores do portfólio e benchmark (IBOVESPA).
  * Estatísticas descritivas, indicadores de risco (Sharpe, Sortino, VaR, CVaR, Drawdown).
  * Visualização de drawdown, beta móvel e sharpe móvel.

* **Simulação Monte Carlo:**

  * Simulações estocásticas para projeção do valor futuro do portfólio.
  * Gráfico interativo estilo “fan chart” com faixas de confiança.
  * Estatísticas resumo (valor esperado, VaR, cenários extremos).

* **Exportação:**

  * Geração e download de relatório completo em HTML usando `quantstats`.

---

## 🛠 Tecnologias e Bibliotecas

* [Streamlit](https://streamlit.io/) - Framework para criação de apps web interativos em Python.
* [yfinance](https://github.com/ranaroussi/yfinance) - Para obtenção de dados financeiros históricos.
* [pypfopt](https://pyportfolioopt.readthedocs.io/en/latest/) - Biblioteca para otimização de portfólio.
* [quantstats](https://github.com/ranaroussi/quantstats) - Análise estatística e relatórios financeiros.
* [Plotly](https://plotly.com/python/) - Visualizações interativas.
* [Seaborn](https://seaborn.pydata.org/) e [Matplotlib](https://matplotlib.org/) - Gráficos estáticos.
* [Pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/) - Manipulação e análise de dados.

---

## 📥 Como usar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu_usuario/b3-explorer.git
   cd b3-explorer
   ```

2. Instale as dependências (recomenda-se ambiente virtual):

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o app:

   ```bash
   streamlit run app.py
   ```

4. No menu lateral, escolha a data inicial, o capital inicial, selecione os ativos do portfólio, defina o modo de alocação (manual ou otimizado) e explore os resultados nas abas.

---

## 📊 Dados

* Lista de ações e setores está em `acoes-listadas-b3.csv` (arquivo necessário).
* Dados históricos obtidos via API do Yahoo Finance.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra issues para bugs ou solicitações de funcionalidades.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 📞 Contato

Desenvolvido por \[Seu Nome] — [seu.email@exemplo.com](mailto:seu.email@exemplo.com)

---

Se desejar, posso ajudar a criar também o arquivo `requirements.txt` com as bibliotecas usadas. Quer?
