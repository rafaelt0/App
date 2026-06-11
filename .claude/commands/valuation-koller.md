---
description: >
  Assistente especializado em valuation de empresas com base na metodologia McKinsey/Koller
  ("Valuation: Measuring and Managing the Value of Companies", 4ª ed.). Use esta skill quando
  o usuário pedir para valorar uma empresa, calcular WACC, estimar fluxo de caixa livre,
  projetar crescimento, calcular valor terminal/continuing value, analisar ROIC, ou usar
  qualquer modelo de DCF. Também use quando o usuário mencionar: "Enterprise DCF",
  "Economic Profit", "NOPLAT", "Invested Capital", "valor residual", "custo de capital",
  "beta desalavancado", "multiples valuation", "free cash flow to firm", ou pedir para
  "valorar" ou "precificar" um negócio — mesmo que não cite explicitamente a metodologia.
---

# Valuation McKinsey/Koller — Guia Técnico Completo

## Princípio Fundamental

> **Valor = f(ROIC, Crescimento, Custo de Capital)**
> Empresas criam valor quando ROIC > WACC. O crescimento só cria valor se ROIC > WACC.

---

## Etapa 1 — Definir o Modelo Adequado

| Situação | Modelo Recomendado |
|---|---|
| Estrutura de capital estável (D/V constante) | **Enterprise DCF com WACC** |
| Estrutura de capital muda significativamente | **APV (Adjusted Present Value)** |
| Foco em destruição/criação de valor por período | **Economic Profit** |
| Banco, seguradora ou financeira | **Equity Cash Flow** |
| Checagem rápida ou ausência de projeções detalhadas | **Múltiplos** |

> Use Enterprise DCF como modelo primário + Economic Profit como check interpretativo.

---

## Etapa 2 — Análise Histórica

**Objetivo:** Calcular ROIC e crescimento histórico para calibrar projeções.

### 2a. Reorganizar as Demonstrações Financeiras

Separar itens **operacionais** de **não-operacionais** e **financeiros**:

```
NOPLAT = EBITA × (1 − t)
       = (Receita − COGS − SG&A − D&A) × (1 − t)

Notas:
- EBITA = EBIT + Amortização de Goodwill
- Excluir itens não-recorrentes (reestruturação, ganhos/perdas de ativos)
- t = alíquota efetiva de impostos sobre lucro operacional
```

### 2b. Calcular Invested Capital

```
Invested Capital = Capital de Giro Operacional
                 + Ativo Imobilizado Líquido
                 + Goodwill e Intangíveis
                 + Outros Ativos Operacionais de LP
                 − Outros Passivos Operacionais de LP

Capital de Giro Operacional = Recebíveis + Estoques + Outros AC Operacionais
                             − Fornecedores − Outros Passivos Operacionais CP
(EXCLUIR caixa excedente e dívida de curto prazo)
```

### 2c. Calcular ROIC

```
ROIC = NOPLAT / Invested Capital (início do período)
     = NOPLAT / IC_t-1

ROIC sem Goodwill = NOPLAT / (IC − Goodwill − Intangíveis)
(usar para benchmark operacional puro)
```

### 2d. Calcular Free Cash Flow Histórico

```
FCF = NOPLAT − Variação no Invested Capital
    = NOPLAT − ΔWCO − ΔImobilizado − ΔOutros Ativos Operacionais

Ou equivalentemente:
FCF = NOPLAT + D&A − CAPEX − ΔWCO

Verificação: FCF = NOPLAT × (1 − Taxa de Reinvestimento)
onde Taxa de Reinvestimento = g / ROIC
```

---

## Etapa 3 — Drivers de Valor (ROIC e Crescimento)

Calibrar expectativas com evidências empíricas (Cap. 6):

| Métrica | Evidência Empírica |
|---|---|
| ROIC mediano (mercado amplo) | ~10–12% |
| ROIC de alto desempenho | Declina para 15% em 15 anos (mean reversion) |
| Crescimento real mediano (40 anos) | ~6,3% real; ~10,2% nominal |
| Crescimento > 20% real | Cai para ~8% em 5 anos |
| Grandes empresas (Fortune 50) | ~1% real após ingresso |

> **Regra de ouro:** Seja cético com projeções de crescimento > 10% por mais de 5 anos, a menos que haja justificativa estratégica sólida.

---

## Etapa 4 — Projeção de Performance

**Horizonte:** Geralmente 10–15 anos de projeção explícita até o valor terminal.

### 4a. Projetar a DRE

```
Receita  = Receita_t × (1 + g)
COGS     = Receita × (COGS/Receita histórico)
SG&A     = Receita × (SG&A/Receita histórico)
EBITA    = Receita − COGS − SG&A
NOPLAT   = EBITA × (1 − t)
```

### 4b. Projetar o Balanço (Invested Capital)

Método preferido — projetar em **stocks** (saldos), não fluxos:

```
Recebíveis   = Receita × (Recebíveis/Receita histórico)
Estoques     = CMV × (Estoques/CMV histórico)
Fornecedores = CMV × (Fornecedores/CMV histórico)
PP&E Líquido = Receita × (PP&E/Receita histórico)
CAPEX        = ΔPP&E_líq + Depreciação
```

> Sempre verificar se o CAPEX implícito é positivo e razoável.

### 4c. Calcular FCF Projetado

```
FCF_t = NOPLAT_t − ΔIC_t
```

---

## Etapa 5 — Valor Terminal (Continuing Value)

O valor terminal representa o valor além do horizonte explícito.

**Fórmula preferida (Gordon Growth Model adaptado):**

```
CV_T = NOPLAT_{T+1} × (1 − g/ROIC_cv) / (WACC − g)

Onde:
- NOPLAT_{T+1} = NOPLAT do primeiro ano após o horizonte
- g            = taxa de crescimento na perpetuidade (≈ PIB real + inflação)
- ROIC_cv      = ROIC esperado na perpetuidade
- WACC         = custo médio ponderado de capital
```

**Premissas razoáveis para CV:**
- `g` típico: 2–4% nominal (≈ PIB real + inflação de longo prazo)
- `ROIC_cv`: conservador — usar WACC se não há vantagem competitiva sustentável; usar ROIC histórico ajustado se há moat

**Fórmula alternativa (quando ROIC_cv = WACC):**

```
CV_T = NOPLAT_{T+1} / WACC
(retorno incremental = custo de capital → crescimento não cria valor adicional)
```

> ⚠️ O CV geralmente representa **60–80% do Enterprise Value**. Sensibilidades em `g` e `ROIC_cv` são críticas.

---

## Etapa 6 — Custo de Capital (WACC)

```
WACC = (E/V) × ke + (D/V) × kd × (1 − t)

Onde:
- E/V, D/V = pesos pelo valor de mercado (não valor contábil)
- ke        = custo do equity (CAPM)
- kd        = custo da dívida (yield to maturity de dívida de LP)
- t         = alíquota marginal de impostos
```

### 6a. Custo do Equity — CAPM

```
ke = rf + β × (Rm − rf)

Onde:
- rf        = taxa livre de risco (yield de T-Bond de 10 anos)
- (Rm − rf) = prêmio de risco de mercado ≈ 4–6% (histórico EUA)
- β         = beta relevado da empresa (ou beta da indústria relevado)
```

### 6b. Estimativa de Beta

Processo de 4 passos (beta da indústria):

1. **Regressão histórica:** beta bruto de cada empresa comparável vs. índice de mercado
2. **Desalavancar:** `βu = βe / (1 + D/E)` — remove o efeito da alavancagem financeira
3. **Mediana da indústria:** `βu_indústria = mediana(βu_comparáveis)`
4. **Realavancar para a empresa-alvo:** `βe = βu × (1 + D/E_alvo)`

**Suavização de Bloomberg (quando poucos comparáveis):**
```
β_ajustado = 0,33 + 0,67 × β_bruto
```

---

## Etapa 7 — Calcular e Interpretar Resultados

### 7a. Enterprise Value

```
Enterprise Value = Σ [FCF_t / (1 + WACC)^t]  (t = 1 a T)
                 + CV_T / (1 + WACC)^T
```

### 7b. Equity Value

```
Equity Value = Enterprise Value
             + Ativos Não-Operacionais (caixa excedente, subsidiárias, investimentos)
             − Valor de Mercado da Dívida
             − Outros Passivos Não-Equity (opções de funcionários, pensões não-fundadas,
               preferred stock, participações minoritárias)

Preço por Ação = Equity Value / Ações em Circulação
```

**Caixa excedente** = Caixa total − Caixa operacional necessário (≈ 0,5–2% da Receita)

> ⚠️ Nunca incluir ativos não-operacionais no FCF *e* no ajuste de EV simultaneamente (double-counting).

---

## Etapa 8 — Validação por Múltiplos

| Múltiplo | Fórmula | Quando usar |
|---|---|---|
| EV/EBITDA | EV / EBITDA | Mais comum; neutro para D&A |
| EV/EBITA | EV / EBITA | Melhor para comparação entre indústrias |
| EV/NOPLAT | EV / NOPLAT | Consistente com DCF; ajusta impostos |
| P/E | Preço / LPA | Simples; inclui efeito de alavancagem |
| EV/Receita | EV / Receita | Para empresas sem lucro ou early-stage |

**Regras para uso de múltiplos:**
- Sempre usar múltiplos **forward** (baseado em projeção de 12 meses) quando possível
- Ajustar para diferenças de crescimento e ROIC entre comparáveis (PEG ratio)
- Múltiplos são *contexto*, não *substitutos* do DCF

---

## Checklist de Consistência

Antes de apresentar o resultado, verificar:

- [ ] FCF histórico reconcilia com demonstrações financeiras?
- [ ] ROIC implícito nas projeções é consistente com benchmarks da indústria?
- [ ] Taxa de crescimento do CV ≤ crescimento do PIB de longo prazo?
- [ ] WACC usa pesos de mercado (não contábeis)?
- [ ] Ativos não-operacionais não estão no FCF *e* no ajuste de EV?
- [ ] Sensibilidades em `g` e `WACC` foram realizadas?
- [ ] Resultado do DCF foi triangulado com múltiplos?

---

## Referências do Livro (por capítulo)

| Tópico | Capítulo |
|---|---|
| Frameworks DCF (Enterprise, APV, Equity) | Cap. 5 |
| ROIC e crescimento empírico | Cap. 6 |
| Análise histórica (NOPLAT, IC) | Cap. 7 |
| Projeção de performance | Cap. 8 |
| Continuing Value | Cap. 9 |
| Custo de capital | Cap. 10 |
| Cálculo e interpretação de resultados | Cap. 11 |
| Múltiplos | Cap. 12 |
| Medição de performance (ROIC vs. EPS) | Cap. 13 |
| Empresas de alto crescimento | Cap. 19 |
| Mercados emergentes | Cap. 22 |
| Empresas cíclicas | Cap. 24 |
| Instituições financeiras | Cap. 25 |
