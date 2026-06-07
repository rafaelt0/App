import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import random
import re
import unicodedata
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import email.utils
import traceback

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ─── SVG Icon Library ─────────────────────────────────────────────────────────
def _svg(body, size=14):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
            f'{body}</svg>')

ICO_OK      = _svg('<circle cx="12" cy="12" r="9" stroke="#00ff87" stroke-width="1.8"/>'
                   '<path d="M8 12l3 3 5-5" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
ICO_WARN    = _svg('<path d="M12 3L22 21H2L12 3Z" stroke="#ffd600" stroke-width="1.8" stroke-linejoin="round"/>'
                   '<line x1="12" y1="10" x2="12" y2="14" stroke="#ffd600" stroke-width="2" stroke-linecap="round"/>'
                   '<circle cx="12" cy="17.5" r="1" fill="#ffd600"/>')
ICO_CRIT    = _svg('<circle cx="12" cy="12" r="9" stroke="#ff3d5a" stroke-width="1.8"/>'
                   '<line x1="9" y1="9" x2="15" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
                   '<line x1="15" y1="9" x2="9" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>')
ICO_NEWS    = _svg('<rect x="3" y="4" width="18" height="16" rx="2" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<line x1="7" y1="8" x2="17" y2="8" stroke="#00d2ff" stroke-width="1.5" stroke-linecap="round"/>'
                   '<line x1="7" y1="12" x2="13" y2="12" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>'
                   '<line x1="7" y1="16" x2="15" y2="16" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>', 16)
ICO_CHART   = _svg('<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
                   '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
                   '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>', 16)
ICO_TARGET  = _svg('<circle cx="12" cy="12" r="9" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<circle cx="12" cy="12" r="5" stroke="#ffd600" stroke-width="1.5"/>'
                   '<circle cx="12" cy="12" r="2" fill="#00ff87"/>', 16)
ICO_IDEA    = _svg('<circle cx="12" cy="10" r="6" stroke="#ffd600" stroke-width="1.8"/>'
                   '<path d="M9 16.5h6M10 19h4" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="12" y1="4" x2="12" y2="2" stroke="#ffd600" stroke-width="1.5" stroke-linecap="round"/>', 16)
ICO_CPU     = _svg('<rect x="4" y="4" width="16" height="16" rx="2" stroke="#eab308" stroke-width="1.8"/>'
                   '<path d="M9 9h6v6H9z" fill="#eab308" opacity="0.3"/>'
                   '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" stroke="#eab308" stroke-width="1.5"/>', 16)

def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f'{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True)

def diag_row(icon_svg, text, color):
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:4px 0;'
        f'color:{color};font-size:0.88rem;line-height:1.4;">'
        f'<div style="flex-shrink:0;margin-top:2px;display:flex;align-items:center;">{icon_svg}</div>'
        f'<div style="flex-grow:1;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True)

def get_diag_row_html(icon_svg, text, color):
    return (
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;'
        f'color:{color};font-size:0.88rem;line-height:1.4;">'
        f'<div style="flex-shrink:0;margin-top:2px;display:flex;align-items:center;">{icon_svg}</div>'
        f'<div style="flex-grow:1;">{text}</div>'
        f'</div>'
    )

# Customização do Plotly para o tema Obsidian Neo-Financial
def apply_plotly_theme(fig):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Grotesk, sans-serif", color="#f8fafc"),
        xaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        )
    )

# Hero Header
st.markdown("""
<div class="page-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="1.5">
          <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z"/>
          <path d="M14 4v4h4"/>
          <line x1="7" y1="9" x2="11" y2="9" stroke-linecap="round"/>
          <line x1="7" y1="13" x2="17" y2="13" stroke-linecap="round"/>
          <line x1="7" y1="17" x2="17" y2="17" stroke-linecap="round"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">Notícias & Sentimento do Portfólio</h1>
        <p class="page-hero-subtitle">Monitore notícias em tempo real e analise o sentimento de mercado agregando o impacto qualitativo nas ações de sua carteira.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── ALGORITMO PLN DE SENTIMENTO (Lexicon-Based Fallback) ────────────────────────
def analise_sentimento_pln(title, summary):
    # Converte para minúsculas
    text = (title + " " + summary).lower()
    
    # Normalização de acentos para robustez do matching
    def remover_acentos(txt):
        return "".join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    
    text_normalized = remover_acentos(text)
    
    # Lexicon Financeiro em Português
    pos_words = [
        'alta', 'lucro', 'lucros', 'recorde', 'recordes', 'crescimento', 'crescimentos', 'descoberta', 
        'descobertas', 'acordo', 'acordos', 'aprova', 'aprovou', 'aprovado', 'aprovados', 'aprovacao', 
        'expande', 'expandiu', 'expansao', 'liderança', 'lider', 'lidera', 'recuperacao', 'recuperou', 
        'forte', 'fortes', 'positivo', 'positiva', 'positivos', 'ganho', 'ganhos', 'ganhou', 'compra', 
        'compras', 'dividendos', 'dividendo', 'jcp', 'eficiencia', 'eficiente', 'modernizacao', 
        'parceria', 'parcerias', 'valorizacao', 'valorizou', 'descarbonizacao', 'melhora', 'melhorou', 
        'subiu', 'subiram', 'superou', 'superaram', 'estabilizou', 'estabilidade', 'confortavel'
    ]
    
    neg_words = [
        'queda', 'quedas', 'prejuizo', 'prejuizos', 'greve', 'greves', 'suspende', 'suspendeu', 
        'inadimplencia', 'atraso', 'atrasos', 'pressao', 'pressoes', 'perda', 'perdas', 'reducao', 
        'rebaixado', 'rebaixada', 'rebaixamento', 'concorrencia', 'fraca', 'fraco', 'sofre', 'sofreu', 
        'paralisacao', 'divida', 'dividas', 'pessimista', 'crise', 'risco', 'riscos', 'cair', 'caiu', 
        'recua', 'recuou', 'defasagem', 'alavancagem'
    ]
    
    # Compostos Semânticos (Modificadores)
    pos_compounds = [
        'reduz divida', 'reduz dividas', 'reduz custo', 'reduz custos', 
        'reduz inadimplencia', 'reducao de divida', 'reducao de dividas', 
        'reducao de custos', 'reducao de custo', 'queda da inadimplencia',
        'queda de inadimplencia', 'reduzindo divida', 'reduz alavancagem',
        'reduzir divida', 'reducao de alavancagem', 'renegociacao de dividas'
    ]
    
    neg_compounds = [
        'aumento de custos', 'aumento de despesas', 'alta da inadimplencia', 
        'aumento de divida', 'aumento de dividas', 'aumento da inadimplencia'
    ]
    
    matched_pos = []
    matched_neg = []
    
    # 1. Verifica compostos primeiro e limpa do texto para evitar contagem dupla
    for comp in pos_compounds:
        if comp in text_normalized:
            matched_pos.append(comp)
            text_normalized = text_normalized.replace(comp, "")
            
    for comp in neg_compounds:
        if comp in text_normalized:
            matched_neg.append(comp)
            text_normalized = text_normalized.replace(comp, "")
            
    # 2. Tokenização simples com regex
    words = re.findall(r'\b\w+\b', text_normalized)
    for w in words:
        if w in pos_words:
            matched_pos.append(w)
        elif w in neg_words:
            matched_neg.append(w)
            
    # 3. Cálculo matemático do score
    num_pos = len(matched_pos)
    num_neg = len(matched_neg)
    
    if num_pos + num_neg > 0:
        score = (num_pos - num_neg) / (num_pos + num_neg)
    else:
        score = 0.0
        
    # Classificação por limiar
    if score >= 0.20:
        sentiment = "Otimista"
    elif score <= -0.20:
        sentiment = "Pessimista"
    else:
        sentiment = "Neutro"
        
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "pos_terms": list(set(matched_pos)),
        "neg_terms": list(set(matched_neg)),
        "raw_text_length": len(words)
    }

# Dados de notícias pré-definidas por papel principal (Fallback offline)
news_database = {
    "PETR": [
        {
            "title": "Petrobras anuncia descoberta de óleo leve em bloco da Bacia de Santos",
            "summary": "A estatal comunicou a descoberta de indícios de hidrocarbonetos no poço pioneiro do bloco de exploração. Análise preliminar indica óleo de excelente qualidade comercial.",
            "provider": "Bloomberg Línea",
            "impact": "Alto"
        },
        {
            "title": "Conselho da Petrobras aprova nova política de dividendos e plano de investimentos",
            "summary": "O colegiado da petroleira definiu as diretrizes estratégicas de alocação de capital para o próximo ciclo de 5 anos. Decisão reduz volatilidade e agrada analistas de mercado.",
            "provider": "Valor Econômico",
            "impact": "Médio-Alto"
        },
        {
            "title": "Petrobras enfrenta greve parcial de petroleiros nas refinarias do Sudeste",
            "summary": "Sindicatos iniciaram paralisação preventiva alegando descumprimento de cláusulas do acordo coletivo. Companhia ativou plano de contingência para evitar desabastecimento.",
            "provider": "Estadão Broadcast",
            "impact": "Médio"
        },
        {
            "title": "Flutuação do preço internacional do barril de Brent pressiona margens de refino da Petrobras",
            "summary": "A oscilação do barril de petróleo no mercado de Londres aumenta as pressões sobre a defasagem interna de preços de combustíveis da estatal brasileira.",
            "provider": "InfoMoney",
            "impact": "Baixo"
        }
    ],
    "VALE": [
        {
            "title": "Vale registra forte alta nas exportações de minério de ferro de alta pureza para a Ásia",
            "summary": "O volume embarcado de pelotas de alto teor de ferro subiu 12% em comparação ao trimestre anterior. Demanda de siderúrgicas chinesas apoia o resultado operacional.",
            "provider": "Valor Econômico",
            "impact": "Alto"
        },
        {
            "title": "Vale assina acordo estratégico de descarbonização com consórcio europeu",
            "summary": "A mineradora fechou parceria para o desenvolvimento de soluções industriais focadas na redução de emissões do escopo 3. Iniciativa melhora o rating ESG global da empresa.",
            "provider": "Bloomberg Línea",
            "impact": "Médio"
        },
        {
            "title": "Tribunal de Justiça suspende provisoriamente licença de operação de mina da Vale no Pará",
            "summary": "A decisão cautelar atende a pedido de associação comunitária local. Vale informou que recorrerá e que a mina representa menos de 3% da produção anual consolidada.",
            "provider": "Estadão Broadcast",
            "impact": "Médio-Alto"
        },
        {
            "title": "Preço da tonelada do minério de ferro em Dalian opera em estabilidade após dados de estoques",
            "summary": "Os estoques portuários na China apresentaram leve variação, levando analistas a preverem manutenção dos preços do minério de ferro no curto prazo.",
            "provider": "InfoMoney",
            "impact": "Baixo"
        }
    ],
    "ITUB": [
        {
            "title": "Itaú Unibanco reporta lucro recorde no trimestre impulsionado por carteira de crédito corporativo",
            "summary": "O maior banco privado do país superou as projeções de consenso do mercado. O Retorno sobre o Patrimônio Líquido (ROE) atingiu patamar de liderança no setor.",
            "provider": "Valor Econômico",
            "impact": "Alto"
        },
        {
            "title": "Itaú expande plataforma de investimentos digitais e atrai R$ 15 bilhões em captação líquida",
            "summary": "O aplicativo de investimentos do banco registrou forte fluxo de novos clientes, reduzindo o custo de aquisição (CAC) e consolidando a liderança de varejo digital.",
            "provider": "InfoMoney",
            "impact": "Médio"
        },
        {
            "title": "Inadimplência de curto prazo do Itaú apresenta leve alta em carteiras de crédito pessoal",
            "summary": "O indicador de atrasos entre 15 e 90 dias subiu 0.15 pontos percentuais. A diretoria do banco declarou estar confortável e com provisões adequadas.",
            "provider": "Estadão Broadcast",
            "impact": "Baixo-Médio"
        }
    ],
    "BBDC": [
        {
            "title": "Bradesco acelera plano de reestruturação de agências e eficiência operacional",
            "summary": "O banco anunciou o fechamento de postos físicos redundantes e foco em atendimento digitalizado, prevendo economia de R$ 1,2 bilhão em despesas administrativas anuais.",
            "provider": "Valor Econômico",
            "impact": "Médio-Alto"
        },
        {
            "title": "Bradesco reduz provisões para devedores duvidosos (PDD) sinalizando melhora do ciclo de crédito",
            "summary": "A diretoria informou estabilização da inadimplência nos cartões de crédito, permitindo menor alocação preventiva de capital no balanço.",
            "provider": "Bloomberg Línea",
            "impact": "Alto"
        },
        {
            "title": "Bradesco enfrenta concorrência acirrada de fintechs no segmento de microcrédito e pequenos comércios",
            "summary": "A margem financeira líquida no varejo de baixa renda continua sob pressão devido à oferta agressiva de players puramente digitais.",
            "provider": "InfoMoney",
            "impact": "Médio"
        }
    ],
    "BBAS": [
        {
            "title": "Banco do Brasil registra forte expansão na carteira de crédito do agronegócio nacional",
            "summary": "A instituição pública desembolsou volume recorde de recursos na safra atual, mantendo taxas de inadimplência muito abaixo da média de mercado.",
            "provider": "Estadão Broadcast",
            "impact": "Alto"
        },
        {
            "title": "Banco do Brasil anuncia pagamento de R$ 2,5 bilhões em Juros sobre Capital Próprio (JCP)",
            "summary": "O conselho de administração aprovou o provento aos acionistas com base nos lucros acumulados. O dividend yield implícito agrada o mercado financeiro.",
            "provider": "Valor Econômico",
            "impact": "Médio-Alto"
        },
        {
            "title": "Discussões sobre governança em empresas de controle estatal elevam prêmio de risco das ações BBAS3",
            "summary": "Analistas de bancos estrangeiros rebaixaram levemente o preço-alvo das ações citando volatilidade política e governança corporativa no radar.",
            "provider": "Bloomberg Línea",
            "impact": "Médio-Alto"
        }
    ],
    "WEGE": [
        {
            "title": "WEG assina contrato bilionário de fornecimento de geradores eólicos para complexo no Nordeste",
            "summary": "A fabricante catarinense fechou parceria para equipar um dos maiores parques eólicos em construção do país, reforçando sua liderança na transição energética.",
            "provider": "Valor Econômico",
            "impact": "Alto"
        },
        {
            "title": "WEG expande capacidade produtiva de motores elétricos de alta eficiência na Europa",
            "summary": "Com a ampliação de instalações fabris na Alemanha, a WEG reduces prazos de entrega regionais e atende à crescente demanda de substituição industrial de motores antigos.",
            "provider": "Bloomberg Línea",
            "impact": "Médio-Alto"
        },
        {
            "title": "Oscilação cambial e fortalecimento do Real afetam margens de receitas de exportação da WEG",
            "summary": "Como grande parte da receita é dolarizada, a valorização do Real no período atua como redutor contábil no faturamento reportado em moeda local.",
            "provider": "Estadão Broadcast",
            "impact": "Médio"
        }
    ]
}

generic_news_templates = [
    {
        "title": "Companhia {ticker} anuncia investimentos focados em eficiência energética e ESG",
        "summary": "A diretoria da {ticker} aprovou o plano plurianual de modernização de processos, estimando reduzir custos de energia em 15% nos próximos 24 meses.",
        "provider": "Valor Econômico",
        "impact": "Médio"
    },
    {
        "title": "{ticker} reporta resultados operacionais estáveis e em linha com estimativas de mercado",
        "summary": "A empresa apresentou receita líquida estável. O conselho de administração sinalizou manutenção das taxas históricas de payout aos acionistas.",
        "provider": "Bloomberg Línea",
        "impact": "Baixo"
    },
    {
        "title": "Analistas elevam recomendação de {ticker} citando resiliência e solidez financeira no atual cenário",
        "summary": "O time de análise de banco de investimentos elevou a recomendação das ações para compra, apontando forte geração de caixa da empresa.",
        "provider": "InfoMoney",
        "impact": "Médio-Alto"
    },
    {
        "title": "{ticker} enfrenta pressões inflacionárias de custos logísticos e alta de tarifas de transporte",
        "summary": "O aumento no frete rodoviário e nas tarifas portuárias pressionou as margens de lucro bruto da companhia, que avalia repasse parcial de preços.",
        "provider": "Estadão Broadcast",
        "impact": "Médio"
    }
]


# ─── REAL-TIME NEWS RSS FETCHING ──────────────────────────────────────────────
ticker_to_name = {
    'PETR3': 'Petrobras', 'PETR4': 'Petrobras',
    'VALE3': 'Vale',
    'ITUB3': 'Itaú', 'ITUB4': 'Itaú',
    'BBDC3': 'Bradesco', 'BBDC4': 'Bradesco',
    'BBAS3': 'Banco do Brasil',
    'WEGE3': 'Weg',
    'MGLU3': 'Magazine Luiza',
    'ABEV3': 'Ambev',
    'ELET3': 'Eletrobras', 'ELET6': 'Eletrobras',
    'RENT3': 'Localiza',
    'LREN3': 'Lojas Renner',
    'PRIO3': 'PetroRio',
    'HAPV3': 'Hapvida',
    'SANB11': 'Santander',
    'VVAR3': 'Via Varejo', 'BHIA3': 'Casas Bahia',
    'GGBR4': 'Gerdau',
    'ITSA4': 'Itaúsa',
    'SUZB3': 'Suzano',
    'JBSS3': 'JBS',
    'UGPA3': 'Ultrapar',
    'RADL3': 'RaiaDrogasil',
    'EQTL3': 'Equatorial',
    'CSAN3': 'Cosan',
    'CPFE3': 'CPFL Energia',
    'SBSP3': 'Sabesp',
    'TAEE11': 'Taesa',
    'KLBN11': 'Klabin'
}

@st.cache_data(ttl=600)
def get_brazilian_news(ticker_name):
    # Clean up and combine company name to improve search query
    name = ticker_to_name.get(ticker_name, "")
    if name:
        query = f"{ticker_name} OR \"{name}\""
    else:
        query = ticker_name
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        news_items = []
        for item in root.findall('.//item')[:3]:  # Max 3 real news per asset
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            source = item.find('source').text if item.find('source') is not None else ''
            
            # Format date nicely
            formatted_date = ""
            if pub_date:
                try:
                    dt = email.utils.parsedate_to_datetime(pub_date)
                    formatted_date = dt.strftime('%d/%m/%Y %H:%M')
                except:
                    formatted_date = pub_date
            
            # Clean source name from title
            if source and title.endswith(f" - {source}"):
                title = title[:-len(f" - {source}")]
                
            news_items.append({
                'title': title,
                'link': link,
                'date': formatted_date,
                'provider': source,
                'summary': ''
            })
        return news_items
    except Exception as e:
        return []


# ─── DEEP LEARNING MODEL LOAD (FinBERT-PT-BR) ───────────────────────────────
@st.cache_resource
def load_finbert_pipeline():
    try:
        from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
        model_name = "lucas-leme/FinBERT-PT-BR"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
        return nlp
    except Exception as e:
        return None

# Load FinBERT
finbert_nlp = load_finbert_pipeline()


def analise_sentimento_finbert(title, summary, nlp):
    text = title
    if summary:
        text += " " + summary
        
    if nlp is None:
        # Fallback to the lexicon-based model
        res_pln = analise_sentimento_pln(title, summary)
        scores = [
            {'label': 'POSITIVE', 'score': max(0.0, res_pln['score']) if res_pln['sentiment'] == 'Otimista' else 0.0},
            {'label': 'NEGATIVE', 'score': abs(res_pln['score']) if res_pln['sentiment'] == 'Pessimista' else 0.0},
            {'label': 'NEUTRAL', 'score': 1.0 if res_pln['sentiment'] == 'Neutro' else 0.0}
        ]
        return {
            "sentiment": res_pln["sentiment"],
            "score": res_pln["score"],
            "scores": scores,
            "is_finbert": False,
            "pos_terms": res_pln["pos_terms"],
            "neg_terms": res_pln["neg_terms"],
            "raw_text_length": res_pln["raw_text_length"]
        }
        
    try:
        # FinBERT prediction
        res = nlp([text])[0]
        # Map label scores
        score_dict = {item['label']: item['score'] for item in res}
        pos_score = score_dict.get('POSITIVE', 0.0)
        neg_score = score_dict.get('NEGATIVE', 0.0)
        
        # Decide sentiment label based on the highest probability
        max_label = max(score_dict, key=score_dict.get)
        
        if max_label == 'POSITIVE':
            sentiment = "Otimista"
            score = pos_score
        elif max_label == 'NEGATIVE':
            sentiment = "Pessimista"
            score = -neg_score
        else:
            sentiment = "Neutro"
            score = 0.0
            
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "scores": res,
            "is_finbert": True,
            "raw_text_length": len(text.split())
        }
    except Exception as e:
        # Fallback on error
        res_pln = analise_sentimento_pln(title, summary)
        scores = [
            {'label': 'POSITIVE', 'score': max(0.0, res_pln['score']) if res_pln['sentiment'] == 'Otimista' else 0.0},
            {'label': 'NEGATIVE', 'score': abs(res_pln['score']) if res_pln['sentiment'] == 'Pessimista' else 0.0},
            {'label': 'NEUTRAL', 'score': 1.0 if res_pln['sentiment'] == 'Neutro' else 0.0}
        ]
        return {
            "sentiment": res_pln["sentiment"],
            "score": res_pln["score"],
            "scores": scores,
            "is_finbert": False,
            "pos_terms": res_pln["pos_terms"],
            "neg_terms": res_pln["neg_terms"],
            "raw_text_length": res_pln["raw_text_length"]
        }


# ─── CORE STREAMLIT PAGE LOGIC ───────────────────────────────────────────────

# Verifica se a carteira está carregada
if "peso_manual_df" not in st.session_state or st.session_state["peso_manual_df"] is None:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:16px;padding:2.5rem;text-align:center;margin-top:2rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
            <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem">Portfólio não configurado</div>
        <div style="font-size:0.875rem;color:#94a3b8;max-width:400px;margin:0 auto 1.2rem;">
            Configure e carregue seu portfólio na página <strong style="color:#00ff87">Portfolio</strong> para que as notícias sejam filtradas para os ativos da sua carteira.
        </div>
        <div style="display:inline-block;background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.25);border-radius:8px;padding:0.5rem 1.2rem;font-size:0.85rem;color:#00ff87;font-weight:600;">
            → Acesse Portfolio na barra lateral
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Recupera ativos e pesos
peso_df = st.session_state["peso_manual_df"]
tickers = [t.replace(".SA", "") for t in peso_df.index]
pesos = {t.replace(".SA", ""): row.iloc[0] for t, row in peso_df.iterrows()}

# Informar o estado do algoritmo NLP na barra lateral
if finbert_nlp is not None:
    st.sidebar.success(f"🤖 FinBERT-PT-BR Ativo")
else:
    st.sidebar.info(f"📝 PLN Léxico Ativo (Fallback)")

# Gerador dinâmico de notícias baseado nos ativos selecionados com avaliação PLN em tempo real
news_items = []
random.seed(42)  # Semente estática para consistência entre renderizações da mesma sessão

# Barra de progresso para download/fetch de notícias em tempo real
with st.spinner("Buscando notícias e processando sentimento NLP..."):
    for t in tickers:
        real_news = get_brazilian_news(t)
        
        if real_news:
            for item in real_news:
                sentiment_res = analise_sentimento_finbert(item["title"], item.get("summary", ""), finbert_nlp)
                
                news_items.append({
                    "ticker": t,
                    "title": item["title"],
                    "summary": item["summary"] if item["summary"] else "Clique no link do título para ler os detalhes da notícia na fonte oficial.",
                    "sentiment": sentiment_res["sentiment"],
                    "score": sentiment_res["score"],
                    "scores": sentiment_res.get("scores", []),
                    "is_finbert": sentiment_res.get("is_finbert", False),
                    "pos_terms": sentiment_res.get("pos_terms", []),
                    "neg_terms": sentiment_res.get("neg_terms", []),
                    "raw_text_length": sentiment_res.get("raw_text_length", 0),
                    "provider": item["provider"],
                    "impact": "Alto" if abs(sentiment_res["score"]) > 0.6 else "Médio" if abs(sentiment_res["score"]) > 0.3 else "Baixo",
                    "pub_time": item["date"],
                    "link": item["link"],
                    "peso": pesos[t]
                })
        else:
            # Fallback to simulated news database templates if no news found or offline
            prefix = t[:4]
            if prefix in news_database:
                templates = news_database[prefix]
            else:
                templates = []
                for g_temp in generic_news_templates:
                    templates.append({
                        "title": g_temp["title"].format(ticker=t),
                        "summary": g_temp["summary"].format(ticker=t),
                        "provider": g_temp["provider"],
                        "impact": g_temp["impact"]
                    })
            
            hours_ago = random.randint(1, 23)
            for idx, temp in enumerate(templates[:3]):
                pub_time = f"Há {hours_ago + idx*4} horas" if hours_ago + idx*4 < 24 else f"Ontem às {random.randint(9, 20):02d}:{random.randint(0, 59):02d}"
                sentiment_res = analise_sentimento_finbert(temp["title"], temp["summary"], finbert_nlp)
                
                news_items.append({
                    "ticker": t,
                    "title": temp["title"],
                    "summary": temp["summary"],
                    "sentiment": sentiment_res["sentiment"],
                    "score": sentiment_res["score"],
                    "scores": sentiment_res.get("scores", []),
                    "is_finbert": sentiment_res.get("is_finbert", False),
                    "pos_terms": sentiment_res.get("pos_terms", []),
                    "neg_terms": sentiment_res.get("neg_terms", []),
                    "raw_text_length": sentiment_res.get("raw_text_length", 0),
                    "provider": temp["provider"],
                    "impact": temp["impact"],
                    "pub_time": pub_time,
                    "link": "#",
                    "peso": pesos[t]
                })

# Cálculos de sentimentos consolidados baseados nas métricas dinâmicas do NLP
total_score = 0.0
total_weight = 0.0
pos_count = 0
neg_count = 0
neu_count = 0

for item in news_items:
    total_score += item["score"] * item["peso"]
    total_weight += item["peso"]
    if item["sentiment"] == "Otimista":
        pos_count += 1
    elif item["sentiment"] == "Pessimista":
        neg_count += 1
    else:
        neu_count += 1

# Normaliza score global de -1 a +1 para 0 a 100
avg_score = (total_score / total_weight) if total_weight > 0 else 0.0
normalized_score = int((avg_score + 1.0) / 2.0 * 100)

sentiment_label = "FORTEMENTE OTIMISTA" if normalized_score >= 80 else \
                  "OTIMISTA" if normalized_score >= 60 else \
                  "NEUTRO / EQUILIBRADO" if normalized_score >= 40 else \
                  "PREOCUPANTE" if normalized_score >= 20 else "CRÍTICO"

score_color = "#00ff87" if normalized_score >= 60 else \
              "#ffd600" if normalized_score >= 40 else "#ff3d5a"

# Exibição do painel principal
col_g1, col_g2 = st.columns([1, 2])

with col_g1:
    # Card do Score de Sentimento
    nlp_engine_label = "FinBERT-PT-BR" if finbert_nlp is not None else "PLN Léxico"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0e1b2f, #080c14); 
                border: 2px solid {score_color}; 
                border-radius: 16px; 
                padding: 1.8rem 1.5rem; 
                text-align: center; 
                box-shadow: 0 0 20px {score_color}1a;
                margin-bottom: 1.5rem;">
        <div style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">Sentimento Consolidado ({nlp_engine_label})</div>
        <div style="font-size: 3.5rem; font-weight: 900; color: {score_color}; font-family: 'JetBrains Mono', monospace; margin: 0.5rem 0;">
            {normalized_score}<span style="font-size: 1.5rem; font-weight: 500; color: #94a3b8;">/100</span>
        </div>
        <div style="font-size: 0.85rem; font-weight: 700; color: {score_color}; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.8rem;">
            {sentiment_label}
        </div>
        <div style="display: flex; justify-content: space-around; border-top: 1px solid #1e293b; padding-top: 0.8rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
            <div>
                <span style="color: #4ade80; font-weight: 700;">{pos_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Positivas</div>
            </div>
            <div>
                <span style="color: #60a5fa; font-weight: 700;">{neu_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Neutras</div>
            </div>
            <div>
                <span style="color: #f87171; font-weight: 700;">{neg_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Negativas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_g2:
    # Gráfico de Distribuição do Sentimento por Ativo
    asset_sentiments = []
    for t in tickers:
        t_items = [x for x in news_items if x["ticker"] == t]
        pos = sum(1 for x in t_items if x["sentiment"] == "Otimista")
        neu = sum(1 for x in t_items if x["sentiment"] == "Neutro")
        neg = sum(1 for x in t_items if x["sentiment"] == "Pessimista")
        asset_sentiments.append({"Ativo": t, "Otimista": pos, "Neutro": neu, "Pessimista": neg})
        
    df_sent = pd.DataFrame(asset_sentiments)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Pessimista', y=df_sent['Ativo'], x=df_sent['Pessimista'], 
        orientation='h', marker=dict(color='#f87171')
    ))
    fig.add_trace(go.Bar(
        name='Neutro', y=df_sent['Ativo'], x=df_sent['Neutro'], 
        orientation='h', marker=dict(color='#60a5fa')
    ))
    fig.add_trace(go.Bar(
        name='Otimista', y=df_sent['Ativo'], x=df_sent['Otimista'], 
        orientation='h', marker=dict(color='#4ade80')
    ))
    
    fig.update_layout(
        barmode='stack',
        title="Volume de Notícias e Distribuição por Ativo",
        xaxis=dict(title="Quantidade de Notícias", dtick=1),
        yaxis=dict(title="Ativos"),
        height=200 + len(tickers) * 35,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ─── NEWS FEED LISTING ───────────────────────────────────────────────────────

# Filtro lateral/superior de notícias
section_header(ICO_NEWS, "Feed Qualitativo de Notícias da Carteira", "h3")

col_filter, col_sort = st.columns([1, 1])
with col_filter:
    selected_ticker = st.selectbox(
        "Filtrar por ativo",
        ["Todos os Ativos"] + [f"{t} ({sum(1 for x in news_items if x['ticker']==t)} notícias)" for t in tickers],
    )
    # Normaliza a seleção (remove o sufixo de contagem)
    selected_ticker_clean = selected_ticker.split(" (")[0] if selected_ticker != "Todos os Ativos" else "Todos os Ativos"
with col_sort:
    sort_mode = st.selectbox(
        "Ordenar por",
        ["Mais recentes", "Mais impactantes", "Mais otimistas", "Mais pessimistas"]
    )

filtered_news = news_items if selected_ticker_clean == "Todos os Ativos" else [x for x in news_items if x["ticker"] == selected_ticker_clean]

# Ordenar notícias
def parse_pub_time(pub_time):
    # Try to parse string formats to sort news nicely
    if "Há" in pub_time:
        try:
            return int(pub_time.split()[1])
        except:
            return 24
    elif "/" in pub_time:
        try:
            dt = datetime.datetime.strptime(pub_time, '%d/%m/%Y %H:%M')
            # return negative timestamp for descending sort
            return -int(dt.timestamp())
        except:
            return 0
    return 24

if sort_mode == "Mais recentes":
    filtered_news = sorted(filtered_news, key=lambda x: parse_pub_time(x["pub_time"]))
elif sort_mode == "Mais impactantes":
    impact_rank = {"Alto": 0, "Médio-Alto": 1, "Médio": 2, "Baixo-Médio": 3, "Baixo": 4}
    filtered_news = sorted(filtered_news, key=lambda x: impact_rank.get(x["impact"], 5))
elif sort_mode == "Mais otimistas":
    filtered_news = sorted(filtered_news, key=lambda x: -x["score"])
elif sort_mode == "Mais pessimistas":
    filtered_news = sorted(filtered_news, key=lambda x: x["score"])

total_news = len(filtered_news)
pos_f = sum(1 for x in filtered_news if x["sentiment"] == "Otimista")
neg_f = sum(1 for x in filtered_news if x["sentiment"] == "Pessimista")
neu_f = total_news - pos_f - neg_f
st.caption(f"Exibindo {total_news} notícias — {pos_f} otimistas · {neu_f} neutras · {neg_f} pessimistas")

for news in filtered_news:
    badge_bg = "rgba(74, 222, 128, 0.1)" if news["sentiment"] == "Otimista" else \
               "rgba(248, 113, 113, 0.1)" if news["sentiment"] == "Pessimista" else "rgba(96, 165, 250, 0.1)"
    badge_color = "#4ade80" if news["sentiment"] == "Otimista" else \
                  "#f87171" if news["sentiment"] == "Pessimista" else "#60a5fa"
                  
    impact_color = "#4ade80" if news["impact"] == "Baixo" else \
                   "#ffd600" if "Médio" in news["impact"] else "#ff3d5a"

    # Clickable title if link is present
    title_html = f'<a class="news-title-link" href="{news["link"]}" target="_blank">{news["title"]}</a>' if news["link"] != "#" else news["title"]

    # News Card Container
    st.markdown(f"""
    <div style="background-color: var(--panel-bg); 
                border: 1px solid var(--border-color); 
                border-radius: 12px; 
                padding: 1.2rem; 
                margin-bottom: 0.5rem; 
                box-shadow: var(--shadow-dark);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="background: rgba(0, 210, 255, 0.1); color: var(--secondary-color); border: 1px solid rgba(0, 210, 255, 0.25); border-radius: 4px; padding: 0.1rem 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;">
                    {news["ticker"]}
                </span>
                <span style="color: var(--text-muted); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">
                    {news["provider"]} • {news["pub_time"]}
                </span>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}40; border-radius: 10rem; padding: 0.15rem 0.5rem; font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; font-weight: 700;">
                    {news["sentiment"].upper()}
                </span>
                <span style="font-size: 0.7rem; color: #94a3b8; font-weight: 600;">
                    IMPACTO: <span style="color: {impact_color}; font-weight: 800;">{news["impact"].upper()}</span>
                </span>
            </div>
        </div>
        <h4 style="margin: 0.3rem 0 0.5rem 0 !important; font-size: 1rem !important; font-weight: 600; line-height: 1.4; color: var(--text-main);">
            {title_html}
        </h4>
        <p style="margin: 0; color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.6rem;">
            {news["summary"]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # NLP Explainer Expander
    if news.get("is_finbert", False):
        with st.expander(f"🧠 Detalhes do Algoritmo PLN (FinBERT-PT-BR - Score: {news['score']})"):
            scores_list = news.get("scores", [])
            scores_dict = {item['label']: item['score'] for item in scores_list}
            pos_prob = scores_dict.get('POSITIVE', 0.0) * 100
            neg_prob = scores_dict.get('NEGATIVE', 0.0) * 100
            neu_prob = scores_dict.get('NEUTRAL', 0.0) * 100
            
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; margin-top: 0.2rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    Distribuição de Probabilidade — FinBERT-PT-BR (LLM)
                </div>
                
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Otimista (POSITIVE)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #00ff87;">{pos_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #00ff87; width: {pos_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #00ff8780;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Pessimista (NEGATIVE)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #ff3d5a;">{neg_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #ff3d5a; width: {neg_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #ff3d5a80;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Neutro (NEUTRAL)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #ffd600;">{neu_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #ffd600; width: {neu_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #ffd60080;"></div>
                    </div>
                </div>
                <p style="font-size:0.7rem; color:#64748b; margin-top:8px; margin-bottom:0;">
                    Classificação baseada em modelo de linguagem BERT pré-treinado em finanças. Tamanho do texto: {news['raw_text_length']} palavras.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander(f"📝 Detalhes do Algoritmo PLN (Léxico - Score: {news['score']})"):
            col_exp1, col_exp2 = st.columns([1, 1])
            with col_exp1:
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-bottom:2px; font-weight:600;'> termos positivos encontrados </p>", unsafe_allow_html=True)
                if news["pos_terms"]:
                    pos_html = " ".join([f"<span style='background:rgba(74, 222, 128, 0.15); color:#4ade80; border:1px solid #4ade8040; border-radius:4px; padding:2px 6px; font-size:0.72rem; font-family:\"JetBrains Mono\", monospace;'>{w}</span>" for w in news["pos_terms"]])
                    st.markdown(pos_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='font-size:0.72rem; color:#64748b; font-style:italic;'>Nenhum</span>", unsafe_allow_html=True)
                    
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-top:8px; margin-bottom:2px; font-weight:600;'> termos negativos encontrados </p>", unsafe_allow_html=True)
                if news["neg_terms"]:
                    neg_html = " ".join([f"<span style='background:rgba(248, 113, 113, 0.15); color:#f87171; border:1px solid #f8717140; border-radius:4px; padding:2px 6px; font-size:0.72rem; font-family:\"JetBrains Mono\", monospace;'>{w}</span>" for w in news["neg_terms"]])
                    st.markdown(neg_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='font-size:0.72rem; color:#64748b; font-style:italic;'>Nenhum</span>", unsafe_allow_html=True)
            with col_exp2:
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-bottom:2px; font-weight:600;'> equação do score pln </p>", unsafe_allow_html=True)
                pos_len = len(news["pos_terms"])
                neg_len = len(news["neg_terms"])
                denom = pos_len + neg_len
                denom_str = str(denom) if denom > 0 else "1 (suavizado)"
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 8px; padding: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                    Score = (Pos - Neg) / (Pos + Neg)<br>
                    Score = ({pos_len} - {neg_len}) / {denom_str}<br>
                    <b>Score Final = {news['score']}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.7rem; color:#64748b; margin-top:4px;'>Tamanho do texto tokenizado: {news['raw_text_length']} palavras.</p>", unsafe_allow_html=True)
                
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

st.markdown("---")

# Painel de Decisão de Notícias / Insights de Alocação
section_header(ICO_TARGET, "Insights Estratégicos & Análise de Risco Qualitativo", "h3")

insights_html = []

# Gerar diagnósticos baseados no score médio
if normalized_score >= 60:
    insights_html.append(get_diag_row_html(ICO_OK, f"<b>Fator de Sentimento Positivo:</b> A carteira possui sentimentos favoráveis dominantes. Isto apoia a tese de manutenção ou leve ampliação em correções técnicas.", "#00ff87"))
elif normalized_score >= 40:
    insights_html.append(get_diag_row_html(ICO_WARN, f"<b>Sentimento de Consolidação:</b> Fluxo de notícias equilibrado entre fatores macro e dinâmicas internas. Mantenha os rebalanceamentos normais programados.", "#ffd600"))
else:
    insights_html.append(get_diag_row_html(ICO_CRIT, f"<b>Sinal de Alerta Qualitativo:</b> Sentimento desfavorável predominante nos ativos selecionados. Monitore potenciais rompimentos de suporte técnico.", "#ff3d5a"))
    
# Análise de concentração qualitativa (pesos elevados em ações com sentimento negativo)
risco_alto = False
for item in news_items:
    if item["sentiment"] == "Pessimista" and item["peso"] >= 0.25:
        risco_alto = True
        insights_html.append(get_diag_row_html(ICO_CRIT, f"<b>Risco de Concentração Negativa:</b> O ativo <b>{item['ticker']}</b> tem peso expressivo ({item['peso']*100:.1f}%) e está sob fluxo de notícias pessimistas (<i>{item['title']}</i>).", "#ff3d5a"))
        
if not risco_alto:
    insights_html.append(get_diag_row_html(ICO_OK, "<b>Risco Qualitativo Controlado:</b> Não foram detectadas posições altamente concentradas em ativos com fluxo de notícias pessimistas graves.", "#00ff87"))
    
insights_html.append(get_diag_row_html(ICO_IDEA, "<b>Sugestão de Alocação:</b> Use as notícias e o indicador qualitativo para programar rebalanceamentos operacionais. Em momentos de sentimento extremo, a volatilidade de curto prazo tende a se elevar.", "#ffd600"))

st.markdown(f"""
<div class="financial-panel">
    {"".join(insights_html)}
</div>
""", unsafe_allow_html=True)
