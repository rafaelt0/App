"""Universal multi-asset catalog, ticker resolution, synthetic fixed income, and Yahoo Finance lookup.

Supports:
- Ações Brasileiras (B3): local CSV universe + any B3 share class
- Fundos Imobiliários (FIIs, Fiagros, FI-Infra na B3)
- Ações Americanas (NYSE / NASDAQ / ADRs)
- REITs (US Real Estate Investment Trusts)
- Títulos e Renda Fixa:
  * Títulos Públicos (Tesouro Selic, Tesouro IPCA+, Tesouro Prefixado, Renda+, Educa+)
  * Renda Fixa Bancária e Crédito Privado (CDI, CDB % CDI, LCI/LCA % CDI, CDI+, IPCA+, Prefixado, Poupança)
  * ETFs de Renda Fixa na B3 (IMAB11, B5P211, IRFM11, LFTS11, DEBB11, etc.)
  * Títulos Americanos / US Treasuries & Bond ETFs (SGOV, SHY, IEF, TLT, TIP, BND, AGG, LQD, HYG, etc.)
- ETFs de Renda Variável, BDRs, Criptoativos, Commodities e Moedas
- Qualquer ativo global via busca dinâmica no Yahoo Finance ou expressão de renda fixa personalizada.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import logging
import math
import re
import threading
from typing import Any

import numpy as np
import pandas as pd
import requests

from utils.cache import cache_data
from utils import fund_data
from utils.equity_history import get_equity_history
from utils.market_data import get_fii_market_data, get_listed_stocks, get_yahoo_info, get_yahoo_quotes
from utils.treasury_data import TREASURY_SHARE_STEP, TREASURY_TITLES, latest_treasury_detail, latest_treasury_quote

logger = logging.getLogger(__name__)

ASSET_CLASSES: tuple[dict[str, str], ...] = (
    {"key": "acao_br", "label": "Ações B3", "badge": "Ação B3", "default_currency": "BRL"},
    {"key": "fii", "label": "FIIs (Fundos Imobiliários)", "badge": "FII", "default_currency": "BRL"},
    {"key": "acao_us", "label": "Ações Americanas (EUA)", "badge": "Ação EUA", "default_currency": "USD"},
    {"key": "reit", "label": "REITs (Imobiliário EUA)", "badge": "REIT", "default_currency": "USD"},
    {"key": "titulo", "label": "Títulos e Renda Fixa", "badge": "Título / RF", "default_currency": "BRL"},
    {"key": "etf", "label": "ETFs, BDRs e Fundos", "badge": "ETF / BDR", "default_currency": "BRL"},
    {"key": "outros", "label": "Cripto, Commodities e Global", "badge": "Global", "default_currency": "USD"},
)
ASSET_CLASS_LABELS = {item["key"]: item["label"] for item in ASSET_CLASSES}
ASSET_CLASS_BADGES = {item["key"]: item["badge"] for item in ASSET_CLASSES}
EQUITY_CLASSES = frozenset({"acao_br", "acao_us"})
_EQUITY_INDICATORS = frozenset({"pl", "pvp", "ev_ebitda", "roe", "net_margin", "ebit_margin", "dy"})
ASSET_INDICATORS = {
    "acao_br": _EQUITY_INDICATORS | {"roic", "growth"},
    "acao_us": _EQUITY_INDICATORS,
    "fii": frozenset({"pvp", "dy"}),
    "reit": frozenset({"pvp", "dy"}),
    "etf": frozenset({"dy"}),
    "titulo": frozenset(),
    "outros": frozenset(),
}

# Tuple format:
# (ticker, company, legal_name, sector, asset_class, currency, yahoo_symbol, ref_metrics)
# ref_metrics provides baseline reference figures (cotacao, dy, pl, pvp, roe, liq2m) used by
# the multi-asset screener when live individual scraping of 700+ international tickers isn't run at once.
_CATALOG_ENTRIES: tuple[tuple[str, str, str, str, str, str, str | None, dict[str, float | None]], ...] = (
    # ── Títulos Públicos e Renda Fixa Brasileira (BCB / Tesouro Direto) ──────────────
    ("CDI", "CDI 100% (Renda Fixa Pós-fixada)", "Certificado de Depósito Interbancário (100% CDI)", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1375, "pvp": 1.0, "liq2m": 50_000_000}),
    ("SELIC", "Taxa Selic (Renda Fixa Pós-fixada)", "Sistema Especial de Liquidação e de Custódia (Selic)", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1375, "pvp": 1.0, "liq2m": 50_000_000}),
    ("TESOURO-SELIC", "Tesouro Selic (LFT · Pós-fixado)", "Tesouro Direto Letra Financeira do Tesouro (Tesouro Selic)", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 15850.0, "dy": 0.1375, "pvp": 1.0, "liq2m": 45_000_000}),
    ("TESOURO-SELIC-2027", "Tesouro Selic 2027 (LFT)", "Tesouro Direto Tesouro Selic 2027", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 15850.0, "dy": 0.1380, "pvp": 1.0, "liq2m": 30_000_000}),
    ("TESOURO-SELIC-2029", "Tesouro Selic 2029 (LFT)", "Tesouro Direto Tesouro Selic 2029", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 15820.0, "dy": 0.1385, "pvp": 1.0, "liq2m": 30_000_000}),
    ("TESOURO-IPCA", "Tesouro IPCA+ (NTN-B Principal)", "Tesouro Direto Nota do Tesouro Nacional Série B (IPCA+)", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 4250.0, "dy": 0.1150, "pvp": 1.0, "liq2m": 35_000_000}),
    ("TESOURO-IPCA-2029", "Tesouro IPCA+ 2029 (NTN-B Principal)", "Tesouro Direto Tesouro IPCA+ 2029", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 3320.0, "dy": 0.1160, "pvp": 1.0, "liq2m": 25_000_000}),
    ("TESOURO-IPCA-2035", "Tesouro IPCA+ 2035 (NTN-B Principal)", "Tesouro Direto Tesouro IPCA+ 2035", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 2515.0, "dy": 0.1180, "pvp": 1.0, "liq2m": 30_000_000}),
    ("TESOURO-IPCA-2045", "Tesouro IPCA+ 2045 (NTN-B Principal)", "Tesouro Direto Tesouro IPCA+ 2045", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 1275.0, "dy": 0.1165, "pvp": 1.0, "liq2m": 22_000_000}),
    ("TESOURO-IPCA-2050", "Tesouro IPCA+ 2050 (NTN-B)", "Tesouro Direto Tesouro IPCA+ 2050", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 1050.0, "dy": 0.1160, "pvp": 1.0, "liq2m": 18_000_000}),
    ("TESOURO-PRE", "Tesouro Prefixado (LTN)", "Tesouro Direto Letra do Tesouro Nacional (Prefixado)", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 780.0, "dy": 0.1320, "pvp": 1.0, "liq2m": 25_000_000}),
    ("TESOURO-PRE-2027", "Tesouro Prefixado 2027 (LTN)", "Tesouro Direto Tesouro Prefixado 2027", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 795.0, "dy": 0.1310, "pvp": 1.0, "liq2m": 20_000_000}),
    ("TESOURO-PRE-2031", "Tesouro Prefixado 2031 (LTN)", "Tesouro Direto Tesouro Prefixado 2031", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 540.0, "dy": 0.1340, "pvp": 1.0, "liq2m": 18_000_000}),
    ("TESOURO-RENDA+", "Tesouro Renda+ Aposentadoria Extra (NTN-B1)", "Tesouro Direto Renda+ Aposentadoria Extra IPCA+", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 980.0, "dy": 0.1170, "pvp": 1.0, "liq2m": 15_000_000}),
    ("TESOURO-EDUCA+", "Tesouro Educa+ (NTN-B1)", "Tesouro Direto Educa+ IPCA+", "Títulos Públicos · Tesouro Direto", "titulo", "BRL", None, {"cotacao": 2150.0, "dy": 0.1165, "pvp": 1.0, "liq2m": 12_000_000}),
    ("CDB-100CDI", "CDB Liquidez Diária (100% do CDI)", "Certificado de Depósito Bancário 100% CDI", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1375, "pvp": 1.0, "liq2m": 30_000_000}),
    ("CDB-105CDI", "CDB Bancário (105% do CDI)", "Certificado de Depósito Bancário 105% CDI", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1444, "pvp": 1.0, "liq2m": 20_000_000}),
    ("CDB-110CDI", "CDB Bancário (110% do CDI)", "Certificado de Depósito Bancário 110% CDI", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1512, "pvp": 1.0, "liq2m": 20_000_000}),
    ("CDB-115CDI", "CDB Bancário (115% do CDI)", "Certificado de Depósito Bancário 115% CDI", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1581, "pvp": 1.0, "liq2m": 15_000_000}),
    ("CDB-120CDI", "CDB Bancário (120% do CDI)", "Certificado de Depósito Bancário 120% CDI", "Renda Fixa · Pós-fixado (CDI)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1650, "pvp": 1.0, "liq2m": 12_000_000}),
    ("LCI-90CDI", "LCI Isenta de IR (90% do CDI)", "Letra de Crédito Imobiliário 90% CDI", "Renda Fixa · Isentos (LCI/LCA)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1238, "pvp": 1.0, "liq2m": 15_000_000}),
    ("LCI-95CDI", "LCI Isenta de IR (95% do CDI)", "Letra de Crédito Imobiliário 95% CDI", "Renda Fixa · Isentos (LCI/LCA)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1306, "pvp": 1.0, "liq2m": 15_000_000}),
    ("LCA-95CDI", "LCA Isenta de IR (95% do CDI)", "Letra de Crédito do Agronegócio 95% CDI", "Renda Fixa · Isentos (LCI/LCA)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1306, "pvp": 1.0, "liq2m": 15_000_000}),
    ("CRI-CDI+2", "CRI Crédito Privado (CDI + 2,0% a.a.)", "Certificado de Recebíveis Imobiliários CDI + 2%", "Renda Fixa · Crédito Privado", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1575, "pvp": 1.0, "liq2m": 10_000_000}),
    ("CRA-CDI+2", "CRA Crédito Privado (CDI + 2,0% a.a.)", "Certificado de Recebíveis do Agronegócio CDI + 2%", "Renda Fixa · Crédito Privado", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1575, "pvp": 1.0, "liq2m": 10_000_000}),
    ("IPCA+6", "Renda Fixa / Debênture Incentivada (IPCA + 6,0% a.a.)", "Título Indexado ao IPCA + 6,0% a.a.", "Renda Fixa · Inflação (IPCA+)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1080, "pvp": 1.0, "liq2m": 15_000_000}),
    ("IPCA+7", "Renda Fixa / Debênture Incentivada (IPCA + 7,0% a.a.)", "Título Indexado ao IPCA + 7,0% a.a.", "Renda Fixa · Inflação (IPCA+)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1180, "pvp": 1.0, "liq2m": 15_000_000}),
    ("IPCA+8", "Renda Fixa / Crédito Privado (IPCA + 8,0% a.a.)", "Título Indexado ao IPCA + 8,0% a.a.", "Renda Fixa · Inflação (IPCA+)", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1280, "pvp": 1.0, "liq2m": 10_000_000}),
    ("PRE-12", "Renda Fixa Prefixada (12,0% a.a.)", "Título de Renda Fixa Prefixado 12,0% a.a.", "Renda Fixa · Prefixado", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1200, "pvp": 1.0, "liq2m": 12_000_000}),
    ("PRE-13", "Renda Fixa Prefixada (13,0% a.a.)", "Título de Renda Fixa Prefixado 13,0% a.a.", "Renda Fixa · Prefixado", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1300, "pvp": 1.0, "liq2m": 12_000_000}),
    ("PRE-14", "Renda Fixa Prefixada (14,0% a.a.)", "Título de Renda Fixa Prefixado 14,0% a.a.", "Renda Fixa · Prefixado", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.1400, "pvp": 1.0, "liq2m": 12_000_000}),
    ("POUPANCA", "Caderneta de Poupança (TR + 6,17% a.a.)", "Caderneta de Poupança Brasileira Isenta", "Renda Fixa · Poupança", "titulo", "BRL", None, {"cotacao": 100.0, "dy": 0.0810, "pvp": 1.0, "liq2m": 40_000_000}),

    # ── ETFs de Renda Fixa na B3 ────────────────────────────────────────────────────
    ("IMAB11", "It Now IMA-B (Tesouro IPCA+)", "It Now ID ETF IMA-B Fundo de Índice", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "IMAB11.SA", {"cotacao": 98.5, "dy": 0.112, "pvp": 1.0, "liq2m": 14_000_000}),
    ("B5P211", "It Now IMA-B5 P2 (Tesouro IPCA+ Curto)", "It Now ID ETF IMA-B5 P2 Fundo de Índice", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "B5P211.SA", {"cotacao": 91.2, "dy": 0.108, "pvp": 1.0, "liq2m": 25_000_000}),
    ("IB5M11", "It Now IMA-B5+ (Tesouro IPCA+ Longo)", "It Now ID ETF IMA-B5+ Fundo de Índice", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "IB5M11.SA", {"cotacao": 108.0, "dy": 0.115, "pvp": 1.0, "liq2m": 6_000_000}),
    ("IRFM11", "It Now IRF-M P2 (Tesouro Prefixado)", "It Now ID ETF IRF-M P2 Fundo de Índice", "ETF · Renda Fixa B3 (Prefixado)", "titulo", "BRL", "IRFM11.SA", {"cotacao": 84.5, "dy": 0.125, "pvp": 1.0, "liq2m": 8_000_000}),
    ("LFTS11", "Investo Teva Tesouro Selic ETF", "Investo Teva Tesouro Selic ETF Fundo de Índice", "ETF · Renda Fixa B3 (Selic)", "titulo", "BRL", "LFTS11.SA", {"cotacao": 128.0, "dy": 0.137, "pvp": 1.0, "liq2m": 18_000_000}),
    ("LFTB11", "Investo Tesouro Selic + IPCA ETF", "Investo Teva Tesouro Selic + IPCALonga ETF", "ETF · Renda Fixa B3 (Misto)", "titulo", "BRL", "LFTB11.SA", {"cotacao": 104.0, "dy": 0.128, "pvp": 1.0, "liq2m": 4_500_000}),
    ("NTNS11", "Investo Tesouro IPCA+ Curto ETF", "Investo Teva Tesouro IPCA+ 0 a 4 Anos ETF", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "NTNS11.SA", {"cotacao": 102.0, "dy": 0.110, "pvp": 1.0, "liq2m": 5_000_000}),
    ("DEBB11", "BTG Pactual Teva Debêntures DI ETF", "BTG Pactual Teva Debêntures DI Fundo de Índice", "ETF · Renda Fixa B3 (Crédito)", "titulo", "BRL", "DEBB11.SA", {"cotacao": 115.0, "dy": 0.145, "pvp": 1.0, "liq2m": 3_500_000}),
    ("FIXA11", "Mirae Asset Renda Fixa Pré ETF", "Mirae Asset Renda Fixa Pré Fundo de Índice", "ETF · Renda Fixa B3 (Prefixado)", "titulo", "BRL", "FIXA11.SA", {"cotacao": 14.5, "dy": 0.128, "pvp": 1.0, "liq2m": 2_500_000}),
    ("B5MB11", "Bradesco IMA-B5+ ETF", "Bradesco IMA-B5+ Fundo de Índice", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "B5MB11.SA", {"cotacao": 101.0, "dy": 0.114, "pvp": 1.0, "liq2m": 2_000_000}),
    ("IMBB11", "Bradesco IMA-B ETF", "Bradesco IMA-B Fundo de Índice", "ETF · Renda Fixa B3 (IPCA+)", "titulo", "BRL", "IMBB11.SA", {"cotacao": 96.0, "dy": 0.112, "pvp": 1.0, "liq2m": 2_000_000}),

    # ── Títulos Americanos / US Treasuries & Bond ETFs (USD) ────────────────────────
    ("SGOV", "iShares 0-3 Month Treasury Bond ETF (T-Bills)", "iShares 0-3 Month Treasury Bond ETF", "Títulos EUA · T-Bills (0-3 Meses)", "titulo", "USD", "SGOV", {"cotacao": 100.5, "dy": 0.051, "pvp": 1.0, "liq2m": 80_000_000}),
    ("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "SPDR Bloomberg 1-3 Month T-Bill ETF", "Títulos EUA · T-Bills (1-3 Meses)", "titulo", "USD", "BIL", {"cotacao": 91.6, "dy": 0.050, "pvp": 1.0, "liq2m": 75_000_000}),
    ("SHY", "iShares 1-3 Year Treasury Bond ETF", "iShares 1-3 Year Treasury Bond ETF", "Títulos EUA · Treasuries Curto Prazo", "titulo", "USD", "SHY", {"cotacao": 82.4, "dy": 0.044, "pvp": 1.0, "liq2m": 60_000_000}),
    ("IEI", "iShares 3-7 Year Treasury Bond ETF", "iShares 3-7 Year Treasury Bond ETF", "Títulos EUA · Treasuries Médio Prazo", "titulo", "USD", "IEI", {"cotacao": 117.0, "dy": 0.041, "pvp": 1.0, "liq2m": 45_000_000}),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "iShares 7-10 Year Treasury Bond ETF", "Títulos EUA · Treasuries 7-10 Anos", "titulo", "USD", "IEF", {"cotacao": 95.8, "dy": 0.042, "pvp": 1.0, "liq2m": 90_000_000}),
    ("TLH", "iShares 10-20 Year Treasury Bond ETF", "iShares 10-20 Year Treasury Bond ETF", "Títulos EUA · Treasuries 10-20 Anos", "titulo", "USD", "TLH", {"cotacao": 104.5, "dy": 0.044, "pvp": 1.0, "liq2m": 40_000_000}),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "iShares 20+ Year Treasury Bond ETF", "Títulos EUA · Treasuries Longo Prazo (20+ Anos)", "titulo", "USD", "TLT", {"cotacao": 92.5, "dy": 0.045, "pvp": 1.0, "liq2m": 100_000_000}),
    ("GOVT", "iShares U.S. Treasury Bond ETF", "iShares U.S. Treasury Bond ETF", "Títulos EUA · Treasuries Geral", "titulo", "USD", "GOVT", {"cotacao": 22.8, "dy": 0.041, "pvp": 1.0, "liq2m": 55_000_000}),
    ("TIP", "iShares TIPS Bond ETF (Inflação EUA)", "iShares TIPS Bond ETF", "Títulos EUA · TIPS (Indexados à Inflação)", "titulo", "USD", "TIP", {"cotacao": 108.2, "dy": 0.040, "pvp": 1.0, "liq2m": 50_000_000}),
    ("STIP", "iShares 0-5 Year TIPS Bond ETF", "iShares 0-5 Year TIPS Bond ETF", "Títulos EUA · TIPS Curto Prazo", "titulo", "USD", "STIP", {"cotacao": 100.8, "dy": 0.043, "pvp": 1.0, "liq2m": 35_000_000}),
    ("USFR", "WisdomTree Floating Rate Treasury Fund", "WisdomTree Floating Rate Treasury Fund", "Títulos EUA · Treasury Pós-fixado", "titulo", "USD", "USFR", {"cotacao": 50.4, "dy": 0.052, "pvp": 1.0, "liq2m": 45_000_000}),
    ("TFLO", "iShares Treasury Floating Rate Bond ETF", "iShares Treasury Floating Rate Bond ETF", "Títulos EUA · Treasury Pós-fixado", "titulo", "USD", "TFLO", {"cotacao": 50.6, "dy": 0.051, "pvp": 1.0, "liq2m": 40_000_000}),
    ("BND", "Vanguard Total Bond Market ETF", "Vanguard Total Bond Market Index Fund ETF", "Títulos EUA · Renda Fixa Agregada", "titulo", "USD", "BND", {"cotacao": 73.5, "dy": 0.043, "pvp": 1.0, "liq2m": 85_000_000}),
    ("AGG", "iShares Core U.S. Aggregate Bond ETF", "iShares Core U.S. Aggregate Bond ETF", "Títulos EUA · Renda Fixa Agregada", "titulo", "USD", "AGG", {"cotacao": 98.4, "dy": 0.042, "pvp": 1.0, "liq2m": 85_000_000}),
    ("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "Títulos EUA · Crédito Corporativo IG", "titulo", "USD", "LQD", {"cotacao": 108.9, "dy": 0.049, "pvp": 1.0, "liq2m": 75_000_000}),
    ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "iShares iBoxx $ High Yield Corporate Bond ETF", "Títulos EUA · Crédito High Yield", "titulo", "USD", "HYG", {"cotacao": 79.2, "dy": 0.062, "pvp": 1.0, "liq2m": 80_000_000}),
    ("VCSH", "Vanguard Short-Term Corporate Bond ETF", "Vanguard Short-Term Corporate Bond ETF", "Títulos EUA · Crédito Corporativo Curto", "titulo", "USD", "VCSH", {"cotacao": 78.5, "dy": 0.046, "pvp": 1.0, "liq2m": 45_000_000}),
    ("VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "Vanguard Intermediate-Term Corporate Bond ETF", "Títulos EUA · Crédito Corporativo Médio", "titulo", "USD", "VCIT", {"cotacao": 81.2, "dy": 0.048, "pvp": 1.0, "liq2m": 50_000_000}),
    ("VGSH", "Vanguard Short-Term Treasury ETF", "Vanguard Short-Term Treasury ETF", "Títulos EUA · Treasuries Curto Prazo", "titulo", "USD", "VGSH", {"cotacao": 58.6, "dy": 0.044, "pvp": 1.0, "liq2m": 38_000_000}),
    ("VGIT", "Vanguard Intermediate-Term Treasury ETF", "Vanguard Intermediate-Term Treasury ETF", "Títulos EUA · Treasuries Médio Prazo", "titulo", "USD", "VGIT", {"cotacao": 59.4, "dy": 0.041, "pvp": 1.0, "liq2m": 38_000_000}),
    ("VGLT", "Vanguard Long-Term Treasury ETF", "Vanguard Long-Term Treasury ETF", "Títulos EUA · Treasuries Longo Prazo", "titulo", "USD", "VGLT", {"cotacao": 58.1, "dy": 0.045, "pvp": 1.0, "liq2m": 38_000_000}),
    ("BNDX", "Vanguard Total International Bond ETF", "Vanguard Total International Bond ETF", "Títulos Globais · Renda Fixa Internacional", "titulo", "USD", "BNDX", {"cotacao": 49.5, "dy": 0.041, "pvp": 1.0, "liq2m": 40_000_000}),
    ("EMB", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "Títulos Globais · Soberanos Emergentes", "titulo", "USD", "EMB", {"cotacao": 91.0, "dy": 0.056, "pvp": 1.0, "liq2m": 45_000_000}),

    # ── Fundos Imobiliários (FIIs, Fiagros e FI-Infra na B3) ────────────────────────
    ("HGLG11", "CSHG Logística FII", "Pátria Logística Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "HGLG11.SA", {"cotacao": 158.0, "dy": 0.084, "pvp": 1.02, "liq2m": 11_500_000}),
    ("KNRI11", "Kinea Renda Imobiliária FII", "Kinea Renda Imobiliária Fundo de Investimento Imobiliário", "FII · Híbrido (Lajes e Logística)", "fii", "BRL", "KNRI11.SA", {"cotacao": 146.0, "dy": 0.082, "pvp": 0.92, "liq2m": 9_200_000}),
    ("MXRF11", "Maxi Renda FII", "Maxi Renda Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "MXRF11.SA", {"cotacao": 9.85, "dy": 0.118, "pvp": 1.01, "liq2m": 13_800_000}),
    ("XPML11", "XP Malls FII", "XP Malls Fundo de Investimento Imobiliário", "FII · Shoppings", "fii", "BRL", "XPML11.SA", {"cotacao": 106.5, "dy": 0.098, "pvp": 0.96, "liq2m": 14_200_000}),
    ("VISC11", "Vinci Shopping Centers FII", "Vinci Shopping Centers Fundo de Investimento Imobiliário", "FII · Shoppings", "fii", "BRL", "VISC11.SA", {"cotacao": 108.2, "dy": 0.096, "pvp": 0.91, "liq2m": 8_700_000}),
    ("BTLG11", "BTG Pactual Logística FII", "BTG Pactual Logística Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "BTLG11.SA", {"cotacao": 99.4, "dy": 0.092, "pvp": 0.97, "liq2m": 10_400_000}),
    ("KNCR11", "Kinea Rendimentos Imobiliários FII", "Kinea Rendimentos Imobiliários Fundo de Investimento Imobiliário", "FII · Papel e CRI (CDI)", "fii", "BRL", "KNCR11.SA", {"cotacao": 104.5, "dy": 0.115, "pvp": 1.03, "liq2m": 15_100_000}),
    ("KNIP11", "Kinea Índices de Preços FII", "Kinea Índices de Preços Fundo de Investimento Imobiliário", "FII · Papel e CRI (IPCA+)", "fii", "BRL", "KNIP11.SA", {"cotacao": 91.8, "dy": 0.112, "pvp": 0.97, "liq2m": 12_900_000}),
    ("HGRU11", "CSHG Renda Urbana FII", "Pátria Renda Urbana Fundo de Investimento Imobiliário", "FII · Renda Urbana", "fii", "BRL", "HGRU11.SA", {"cotacao": 124.0, "dy": 0.089, "pvp": 0.98, "liq2m": 7_400_000}),
    ("BCFF11", "BTG Pactual Fundo de Fundos FII", "BTG Pactual Fundo de Fundos Imobiliários", "FII · Fundo de Fundos (FOF)", "fii", "BRL", "BCFF11.SA", {"cotacao": 8.40, "dy": 0.106, "pvp": 0.91, "liq2m": 5_800_000}),
    ("IRDM11", "Iridium Recebíveis Imobiliários FII", "Iridium Recebíveis Imobiliários Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "IRDM11.SA", {"cotacao": 69.5, "dy": 0.121, "pvp": 0.86, "liq2m": 7_100_000}),
    ("XPLG11", "XP Log FII", "XP Log Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "XPLG11.SA", {"cotacao": 98.8, "dy": 0.095, "pvp": 0.92, "liq2m": 8_300_000}),
    ("BRCO11", "Bresco Logística FII", "Bresco Logística Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "BRCO11.SA", {"cotacao": 112.0, "dy": 0.091, "pvp": 0.94, "liq2m": 5_900_000}),
    ("HGBS11", "Hedge Brasil Shopping FII", "Hedge Brasil Shopping Fundo de Investimento Imobiliário", "FII · Shoppings", "fii", "BRL", "HGBS11.SA", {"cotacao": 204.0, "dy": 0.094, "pvp": 0.92, "liq2m": 6_400_000}),
    ("VILG11", "Vinci Logística FII", "Vinci Logística Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "VILG11.SA", {"cotacao": 84.5, "dy": 0.093, "pvp": 0.78, "liq2m": 4_800_000}),
    ("CPTS11", "Capitânia Securities II FII", "Capitânia Securities II Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "CPTS11.SA", {"cotacao": 7.65, "dy": 0.119, "pvp": 0.89, "liq2m": 8_500_000}),
    ("RECR11", "REC Recebíveis Imobiliários FII", "REC Recebíveis Imobiliários Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "RECR11.SA", {"cotacao": 81.2, "dy": 0.122, "pvp": 0.88, "liq2m": 6_200_000}),
    ("RBRR11", "RBR Rendimento High Grade FII", "RBR Rendimento High Grade Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "RBRR11.SA", {"cotacao": 88.4, "dy": 0.114, "pvp": 0.95, "liq2m": 5_600_000}),
    ("VGIP11", "Valora CRI Índice de Preço FII", "Valora CRI Índice de Preço Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "VGIP11.SA", {"cotacao": 85.0, "dy": 0.118, "pvp": 0.93, "liq2m": 4_700_000}),
    ("VRTA11", "Fator Verità FII", "Fator Verità Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "VRTA11.SA", {"cotacao": 84.0, "dy": 0.117, "pvp": 0.92, "liq2m": 4_200_000}),
    ("HGRE11", "CSHG Real Estate FII", "Pátria Real Estate Fundo de Investimento Imobiliário", "FII · Lajes Corporativas", "fii", "BRL", "HGRE11.SA", {"cotacao": 114.0, "dy": 0.088, "pvp": 0.76, "liq2m": 5_100_000}),
    ("JSRE11", "JS Real Estate Multigestão FII", "JS Real Estate Multigestão Fundo de Investimento Imobiliário", "FII · Lajes Corporativas", "fii", "BRL", "JSRE11.SA", {"cotacao": 63.5, "dy": 0.091, "pvp": 0.61, "liq2m": 4_400_000}),
    ("PVBI11", "VBI Prime Properties FII", "VBI Prime Properties Fundo de Investimento Imobiliário", "FII · Lajes Corporativas", "fii", "BRL", "PVBI11.SA", {"cotacao": 86.0, "dy": 0.086, "pvp": 0.84, "liq2m": 6_300_000}),
    ("BRCR11", "BTG Pactual Corporate Office Fund FII", "FII BTG Pactual Corporate Office Fund", "FII · Lajes Corporativas", "fii", "BRL", "BRCR11.SA", {"cotacao": 49.0, "dy": 0.098, "pvp": 0.56, "liq2m": 3_800_000}),
    ("HSML11", "HSI Malls FII", "HSI Malls Fundo de Investimento Imobiliário", "FII · Shoppings", "fii", "BRL", "HSML11.SA", {"cotacao": 88.5, "dy": 0.102, "pvp": 0.89, "liq2m": 6_100_000}),
    ("MALL11", "Genial Malls FII", "Genial Malls Fundo de Investimento Imobiliário", "FII · Shoppings", "fii", "BRL", "MALL11.SA", {"cotacao": 104.0, "dy": 0.095, "pvp": 0.88, "liq2m": 4_900_000}),
    ("TRXF11", "TRX Real Estate FII", "TRX Real Estate Fundo de Investimento Imobiliário", "FII · Renda Urbana", "fii", "BRL", "TRXF11.SA", {"cotacao": 103.5, "dy": 0.105, "pvp": 1.01, "liq2m": 9_800_000}),
    ("GGRC11", "GGR Covepi Renda FII", "GGR Covepi Renda Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "GGRC11.SA", {"cotacao": 10.2, "dy": 0.108, "pvp": 0.91, "liq2m": 4_600_000}),
    ("ALZR11", "Alianza Trust Renda Imobiliária FII", "Alianza Trust Renda Imobiliária Fundo de Investimento Imobiliário", "FII · Renda Urbana e Híbrido", "fii", "BRL", "ALZR11.SA", {"cotacao": 104.8, "dy": 0.092, "pvp": 0.99, "liq2m": 5_400_000}),
    ("LVBI11", "VBI Logístico FII", "VBI Logístico Fundo de Investimento Imobiliário", "FII · Logística e Galpões", "fii", "BRL", "LVBI11.SA", {"cotacao": 105.0, "dy": 0.093, "pvp": 0.91, "liq2m": 5_200_000}),
    ("KNSC11", "Kinea Securities FII", "Kinea Securities Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "KNSC11.SA", {"cotacao": 8.95, "dy": 0.116, "pvp": 0.99, "liq2m": 7_800_000}),
    ("HGCR11", "CSHG Recebíveis Imobiliários FII", "Pátria Recebíveis Imobiliários Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "HGCR11.SA", {"cotacao": 98.5, "dy": 0.115, "pvp": 0.98, "liq2m": 6_900_000}),
    ("RBRF11", "RBR Alpha Multiestratégia Real Estate FII", "RBR Alpha Multiestratégia Real Estate FII", "FII · Fundo de Fundos (FOF)", "fii", "BRL", "RBRF11.SA", {"cotacao": 7.10, "dy": 0.109, "pvp": 0.85, "liq2m": 4_100_000}),
    ("HFOF11", "Hedge Top FOFII 3 FII", "Hedge Top FOFII 3 Fundo de Investimento Imobiliário", "FII · Fundo de Fundos (FOF)", "fii", "BRL", "HFOF11.SA", {"cotacao": 69.0, "dy": 0.107, "pvp": 0.86, "liq2m": 4_500_000}),
    ("KFOF11", "Kinea Fundo de Fundos FII", "Kinea Fundo de Fundos Fundo de Investimento Imobiliário", "FII · Fundo de Fundos (FOF)", "fii", "BRL", "KFOF11.SA", {"cotacao": 88.0, "dy": 0.104, "pvp": 0.92, "liq2m": 3_600_000}),
    ("TGAR11", "TG Ativo Real FII", "TG Ativo Real Fundo de Investimento Imobiliário", "FII · Desenvolvimento e Híbrido", "fii", "BRL", "TGAR11.SA", {"cotacao": 106.0, "dy": 0.132, "pvp": 0.94, "liq2m": 7_900_000}),
    ("MCCI11", "Mauá Capital Recebíveis Imobiliários FII", "Mauá Capital Recebíveis Imobiliários FII", "FII · Papel e CRI", "fii", "BRL", "MCCI11.SA", {"cotacao": 86.5, "dy": 0.116, "pvp": 0.92, "liq2m": 5_700_000}),
    ("CVBI11", "VBI CRI FII", "VBI CRI Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "CVBI11.SA", {"cotacao": 85.2, "dy": 0.119, "pvp": 0.93, "liq2m": 4_800_000}),
    ("RBRY11", "RBR Crédito Imobiliário Estruturado FII", "RBR Crédito Imobiliário Estruturado FII", "FII · Papel e CRI", "fii", "BRL", "RBRY11.SA", {"cotacao": 95.0, "dy": 0.124, "pvp": 0.97, "liq2m": 4_900_000}),
    ("XPCI11", "XP Crédito Imobiliário FII", "XP Crédito Imobiliário Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "XPCI11.SA", {"cotacao": 81.0, "dy": 0.118, "pvp": 0.89, "liq2m": 3_900_000}),
    ("BTCI11", "BTG Pactual Crédito Imobiliário FII", "BTG Pactual Crédito Imobiliário FII", "FII · Papel e CRI", "fii", "BRL", "BTCI11.SA", {"cotacao": 9.30, "dy": 0.117, "pvp": 0.91, "liq2m": 4_600_000}),
    ("VGHF11", "Valora Hedge Fund FII", "Valora Hedge Fund Fundo de Investimento Imobiliário", "FII · Multiestratégia", "fii", "BRL", "VGHF11.SA", {"cotacao": 8.20, "dy": 0.123, "pvp": 0.93, "liq2m": 6_800_000}),
    ("KISU11", "Kilima FII SUNO 30", "Kilima Fundo de Investimento em Cotas de FII SUNO 30", "FII · Fundo de Fundos (FOF)", "fii", "BRL", "KISU11.SA", {"cotacao": 7.60, "dy": 0.108, "pvp": 0.88, "liq2m": 3_400_000}),
    ("KNHY11", "Kinea High Yield CRI FII", "Kinea High Yield CRI Fundo de Investimento Imobiliário", "FII · Papel e CRI", "fii", "BRL", "KNHY11.SA", {"cotacao": 102.5, "dy": 0.125, "pvp": 1.01, "liq2m": 7_200_000}),
    ("KNUQ11", "Kinea Unique HY CDI FII", "Kinea Unique High Yield CDI Fundo de Investimento Imobiliário", "FII · Papel e CRI (CDI)", "fii", "BRL", "KNUQ11.SA", {"cotacao": 103.0, "dy": 0.126, "pvp": 1.02, "liq2m": 6_800_000}),
    ("GARE11", "Guardian Real Estate FII", "Guardian Real Estate Fundo de Investimento Imobiliário", "FII · Renda Urbana e Logística", "fii", "BRL", "GARE11.SA", {"cotacao": 8.95, "dy": 0.114, "pvp": 0.97, "liq2m": 8_100_000}),
    ("RBVA11", "Rio Bravo Renda Varejo FII", "Rio Bravo Renda Varejo Fundo de Investimento Imobiliário", "FII · Renda Urbana", "fii", "BRL", "RBVA11.SA", {"cotacao": 102.0, "dy": 0.109, "pvp": 0.95, "liq2m": 4_300_000}),
    ("RZTR11", "Riza Terrax FII", "Riza Terrax Fundo de Investimento Imobiliário", "FII · Terras Agrícolas (Agro)", "fii", "BRL", "RZTR11.SA", {"cotacao": 93.0, "dy": 0.121, "pvp": 0.95, "liq2m": 5_900_000}),
    ("KNCA11", "Kinea Crédito Agro Fiagro", "Kinea Crédito Agro Fiagro Imobiliário", "FII · Fiagro e Infra", "fii", "BRL", "KNCA11.SA", {"cotacao": 96.5, "dy": 0.128, "pvp": 0.94, "liq2m": 6_100_000}),
    ("FGAA11", "FG/Agro Fiagro", "FG/Agro Fundo de Investimento nas Cadeias Produtivas Agroindustriais", "FII · Fiagro e Infra", "fii", "BRL", "FGAA11.SA", {"cotacao": 8.80, "dy": 0.134, "pvp": 0.92, "liq2m": 4_200_000}),
    ("VGIA11", "Valora CRA Fiagro", "Valora CRA Fundo de Investimento nas Cadeias Produtivas Agroindustriais", "FII · Fiagro e Infra", "fii", "BRL", "VGIA11.SA", {"cotacao": 8.50, "dy": 0.135, "pvp": 0.90, "liq2m": 5_300_000}),
    ("RURA11", "Itaú Asset Rural Fiagro", "Itaú Asset Rural Fiagro Imobiliário", "FII · Fiagro e Infra", "fii", "BRL", "RURA11.SA", {"cotacao": 9.10, "dy": 0.131, "pvp": 0.93, "liq2m": 4_800_000}),
    ("JURO11", "Sparta Infra FI-Infra", "Sparta Infra CDI FIC FI-Infra Renda Fixa", "FII · Fiagro e Infra", "fii", "BRL", "JURO11.SA", {"cotacao": 103.5, "dy": 0.122, "pvp": 1.01, "liq2m": 7_500_000}),
    ("CDII11", "Sparta Infra CDI FI-Infra", "Sparta Infra CDI FIC FI-Infra", "FII · Fiagro e Infra", "fii", "BRL", "CDII11.SA", {"cotacao": 105.0, "dy": 0.132, "pvp": 1.01, "liq2m": 6_400_000}),
    ("KDIF11", "Kinea Infra FI-Infra", "Kinea Infra Fundo de Investimento em Cotas de FI-Infra", "FII · Fiagro e Infra", "fii", "BRL", "KDIF11.SA", {"cotacao": 134.0, "dy": 0.118, "pvp": 0.99, "liq2m": 5_800_000}),

    # ── Ações Americanas (US Equities — NYSE & NASDAQ) ──────────────────────────────
    ("AAPL", "Apple Inc.", "Apple Inc.", "Tecnologia (EUA)", "acao_us", "USD", "AAPL", {"cotacao": 225.0, "pl": 31.5, "pvp": 45.0, "dy": 0.005, "roe": 1.45, "roic": 0.52, "evebitda": 24.5, "liq2m": 95_000_000}),
    ("MSFT", "Microsoft Corporation", "Microsoft Corporation", "Tecnologia (EUA)", "acao_us", "USD", "MSFT", {"cotacao": 425.0, "pl": 34.0, "pvp": 11.8, "dy": 0.007, "roe": 0.36, "roic": 0.28, "evebitda": 23.0, "liq2m": 95_000_000}),
    ("NVDA", "NVIDIA Corporation", "NVIDIA Corporation", "Tecnologia (EUA)", "acao_us", "USD", "NVDA", {"cotacao": 135.0, "pl": 48.0, "pvp": 42.0, "dy": 0.0003, "roe": 0.91, "roic": 0.65, "evebitda": 38.0, "liq2m": 100_000_000}),
    ("AMZN", "Amazon.com Inc.", "Amazon.com, Inc.", "Consumo e Cloud (EUA)", "acao_us", "USD", "AMZN", {"cotacao": 205.0, "pl": 41.0, "pvp": 8.2, "dy": 0.0, "roe": 0.22, "roic": 0.15, "evebitda": 19.5, "liq2m": 95_000_000}),
    ("GOOGL", "Alphabet Inc. (Class A)", "Alphabet Inc.", "Comunicação e Internet (EUA)", "acao_us", "USD", "GOOGL", {"cotacao": 180.0, "pl": 24.0, "pvp": 7.1, "dy": 0.004, "roe": 0.31, "roic": 0.27, "evebitda": 16.5, "liq2m": 90_000_000}),
    ("GOOG", "Alphabet Inc. (Class C)", "Alphabet Inc.", "Comunicação e Internet (EUA)", "acao_us", "USD", "GOOG", {"cotacao": 182.0, "pl": 24.2, "pvp": 7.1, "dy": 0.004, "roe": 0.31, "roic": 0.27, "evebitda": 16.5, "liq2m": 85_000_000}),
    ("META", "Meta Platforms Inc.", "Meta Platforms, Inc.", "Comunicação e Internet (EUA)", "acao_us", "USD", "META", {"cotacao": 580.0, "pl": 27.5, "pvp": 9.1, "dy": 0.003, "roe": 0.35, "roic": 0.29, "evebitda": 17.8, "liq2m": 90_000_000}),
    ("TSLA", "Tesla Inc.", "Tesla, Inc.", "Automóveis e Energia (EUA)", "acao_us", "USD", "TSLA", {"cotacao": 255.0, "pl": 68.0, "pvp": 11.5, "dy": 0.0, "roe": 0.20, "roic": 0.14, "evebitda": 42.0, "liq2m": 95_000_000}),
    ("BRK-B", "Berkshire Hathaway Inc.", "Berkshire Hathaway Inc. Class B", "Financeiro e Holdings (EUA)", "acao_us", "USD", "BRK-B", {"cotacao": 455.0, "pl": 15.2, "pvp": 1.6, "dy": 0.0, "roe": 0.12, "roic": 0.10, "evebitda": 12.0, "liq2m": 80_000_000}),
    ("AVGO", "Broadcom Inc.", "Broadcom Inc.", "Tecnologia (EUA)", "acao_us", "USD", "AVGO", {"cotacao": 175.0, "pl": 38.0, "pvp": 11.2, "dy": 0.012, "roe": 0.32, "roic": 0.21, "evebitda": 25.0, "liq2m": 85_000_000}),
    ("LLY", "Eli Lilly and Company", "Eli Lilly and Company", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "LLY", {"cotacao": 820.0, "pl": 62.0, "pvp": 48.0, "dy": 0.006, "roe": 0.65, "roic": 0.34, "evebitda": 41.0, "liq2m": 80_000_000}),
    ("JPM", "JPMorgan Chase & Co.", "JPMorgan Chase & Co.", "Financeiro e Bancos (EUA)", "acao_us", "USD", "JPM", {"cotacao": 235.0, "pl": 13.0, "pvp": 2.0, "dy": 0.021, "roe": 0.16, "liq2m": 85_000_000}),
    ("V", "Visa Inc.", "Visa Inc.", "Financeiro e Pagamentos (EUA)", "acao_us", "USD", "V", {"cotacao": 295.0, "pl": 30.0, "pvp": 15.0, "dy": 0.008, "roe": 0.50, "roic": 0.35, "evebitda": 22.5, "liq2m": 80_000_000}),
    ("MA", "Mastercard Incorporated", "Mastercard Incorporated", "Financeiro e Pagamentos (EUA)", "acao_us", "USD", "MA", {"cotacao": 510.0, "pl": 35.0, "pvp": 65.0, "dy": 0.005, "roe": 1.75, "roic": 0.55, "evebitda": 26.0, "liq2m": 75_000_000}),
    ("WMT", "Walmart Inc.", "Walmart Inc.", "Consumo e Varejo (EUA)", "acao_us", "USD", "WMT", {"cotacao": 84.0, "pl": 34.0, "pvp": 7.6, "dy": 0.010, "roe": 0.21, "roic": 0.14, "evebitda": 18.0, "liq2m": 80_000_000}),
    ("XOM", "Exxon Mobil Corporation", "Exxon Mobil Corporation", "Petróleo e Energia (EUA)", "acao_us", "USD", "XOM", {"cotacao": 118.0, "pl": 14.2, "pvp": 1.8, "dy": 0.033, "roe": 0.14, "roic": 0.12, "evebitda": 7.5, "liq2m": 85_000_000}),
    ("UNH", "UnitedHealth Group Inc.", "UnitedHealth Group Incorporated", "Saúde e Seguros (EUA)", "acao_us", "USD", "UNH", {"cotacao": 575.0, "pl": 22.0, "pvp": 5.6, "dy": 0.014, "roe": 0.25, "roic": 0.16, "evebitda": 14.5, "liq2m": 80_000_000}),
    ("COST", "Costco Wholesale Corporation", "Costco Wholesale Corporation", "Consumo e Varejo (EUA)", "acao_us", "USD", "COST", {"cotacao": 910.0, "pl": 52.0, "pvp": 16.5, "dy": 0.005, "roe": 0.31, "roic": 0.24, "evebitda": 32.0, "liq2m": 75_000_000}),
    ("HD", "The Home Depot Inc.", "The Home Depot, Inc.", "Consumo e Varejo (EUA)", "acao_us", "USD", "HD", {"cotacao": 405.0, "pl": 27.0, "pvp": 85.0, "dy": 0.022, "roe": 2.50, "roic": 0.32, "evebitda": 18.5, "liq2m": 75_000_000}),
    ("PG", "Procter & Gamble Co.", "The Procter & Gamble Company", "Consumo Básico (EUA)", "acao_us", "USD", "PG", {"cotacao": 170.0, "pl": 26.5, "pvp": 7.8, "dy": 0.024, "roe": 0.31, "roic": 0.21, "evebitda": 18.0, "liq2m": 70_000_000}),
    ("JNJ", "Johnson & Johnson", "Johnson & Johnson", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "JNJ", {"cotacao": 156.0, "pl": 15.5, "pvp": 5.2, "dy": 0.032, "roe": 0.22, "roic": 0.15, "evebitda": 12.5, "liq2m": 70_000_000}),
    ("ORCL", "Oracle Corporation", "Oracle Corporation", "Tecnologia (EUA)", "acao_us", "USD", "ORCL", {"cotacao": 185.0, "pl": 35.0, "pvp": 42.0, "dy": 0.009, "roe": 1.10, "roic": 0.18, "evebitda": 22.0, "liq2m": 75_000_000}),
    ("NFLX", "Netflix Inc.", "Netflix, Inc.", "Comunicação e Streaming (EUA)", "acao_us", "USD", "NFLX", {"cotacao": 795.0, "pl": 42.0, "pvp": 14.5, "dy": 0.0, "roe": 0.34, "roic": 0.25, "evebitda": 30.0, "liq2m": 80_000_000}),
    ("AMD", "Advanced Micro Devices Inc.", "Advanced Micro Devices, Inc.", "Tecnologia (EUA)", "acao_us", "USD", "AMD", {"cotacao": 142.0, "pl": 40.0, "pvp": 4.1, "dy": 0.0, "roe": 0.10, "roic": 0.08, "evebitda": 28.0, "liq2m": 85_000_000}),
    ("CRM", "Salesforce Inc.", "Salesforce, Inc.", "Tecnologia (EUA)", "acao_us", "USD", "CRM", {"cotacao": 320.0, "pl": 44.0, "pvp": 5.2, "dy": 0.005, "roe": 0.11, "roic": 0.09, "evebitda": 24.0, "liq2m": 70_000_000}),
    ("BAC", "Bank of America Corp.", "Bank of America Corporation", "Financeiro e Bancos (EUA)", "acao_us", "USD", "BAC", {"cotacao": 45.0, "pl": 13.8, "pvp": 1.3, "dy": 0.023, "roe": 0.10, "liq2m": 80_000_000}),
    ("KO", "The Coca-Cola Company", "The Coca-Cola Company", "Consumo e Bebidas (EUA)", "acao_us", "USD", "KO", {"cotacao": 64.0, "pl": 25.0, "pvp": 10.5, "dy": 0.030, "roe": 0.40, "roic": 0.18, "evebitda": 20.0, "liq2m": 70_000_000}),
    ("PEP", "PepsiCo Inc.", "PepsiCo, Inc.", "Consumo e Bebidas (EUA)", "acao_us", "USD", "PEP", {"cotacao": 162.0, "pl": 23.0, "pvp": 11.5, "dy": 0.033, "roe": 0.48, "roic": 0.19, "evebitda": 16.0, "liq2m": 65_000_000}),
    ("CVX", "Chevron Corporation", "Chevron Corporation", "Petróleo e Energia (EUA)", "acao_us", "USD", "CVX", {"cotacao": 158.0, "pl": 15.0, "pvp": 1.8, "dy": 0.041, "roe": 0.12, "roic": 0.11, "evebitda": 7.8, "liq2m": 75_000_000}),
    ("MRK", "Merck & Co. Inc.", "Merck & Co., Inc.", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "MRK", {"cotacao": 102.0, "pl": 16.5, "pvp": 6.1, "dy": 0.031, "roe": 0.34, "roic": 0.22, "evebitda": 12.0, "liq2m": 65_000_000}),
    ("ABBV", "AbbVie Inc.", "AbbVie Inc.", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "ABBV", {"cotacao": 175.0, "pl": 18.0, "pvp": 35.0, "dy": 0.036, "roe": 0.70, "roic": 0.18, "evebitda": 13.5, "liq2m": 70_000_000}),
    ("ADBE", "Adobe Inc.", "Adobe Inc.", "Tecnologia (EUA)", "acao_us", "USD", "ADBE", {"cotacao": 510.0, "pl": 36.0, "pvp": 14.8, "dy": 0.0, "roe": 0.38, "roic": 0.30, "evebitda": 25.0, "liq2m": 65_000_000}),
    ("DIS", "The Walt Disney Company", "The Walt Disney Company", "Comunicação e Entretenimento (EUA)", "acao_us", "USD", "DIS", {"cotacao": 112.0, "pl": 28.0, "pvp": 2.0, "dy": 0.008, "roe": 0.07, "roic": 0.06, "evebitda": 14.0, "liq2m": 70_000_000}),
    ("MCD", "McDonald's Corporation", "McDonald's Corporation", "Consumo e Restaurantes (EUA)", "acao_us", "USD", "MCD", {"cotacao": 292.0, "pl": 25.0, "pvp": 18.0, "dy": 0.024, "roe": 0.55, "roic": 0.21, "evebitda": 18.0, "liq2m": 65_000_000}),
    ("CSCO", "Cisco Systems Inc.", "Cisco Systems, Inc.", "Tecnologia (EUA)", "acao_us", "USD", "CSCO", {"cotacao": 58.0, "pl": 18.5, "pvp": 5.1, "dy": 0.028, "roe": 0.26, "roic": 0.18, "evebitda": 13.0, "liq2m": 60_000_000}),
    ("IBM", "International Business Machines", "International Business Machines Corporation", "Tecnologia (EUA)", "acao_us", "USD", "IBM", {"cotacao": 215.0, "pl": 24.0, "pvp": 8.0, "dy": 0.031, "roe": 0.33, "roic": 0.15, "evebitda": 15.0, "liq2m": 55_000_000}),
    ("GE", "GE Aerospace", "General Electric Company", "Indústria e Aeroespacial (EUA)", "acao_us", "USD", "GE", {"cotacao": 182.0, "pl": 34.0, "pvp": 9.8, "dy": 0.006, "roe": 0.26, "roic": 0.18, "evebitda": 21.0, "liq2m": 60_000_000}),
    ("CAT", "Caterpillar Inc.", "Caterpillar Inc.", "Indústria e Máquinas (EUA)", "acao_us", "USD", "CAT", {"cotacao": 395.0, "pl": 18.0, "pvp": 9.2, "dy": 0.014, "roe": 0.52, "roic": 0.24, "evebitda": 13.5, "liq2m": 60_000_000}),
    ("NKE", "NIKE Inc.", "NIKE, Inc.", "Consumo e Vestuário (EUA)", "acao_us", "USD", "NKE", {"cotacao": 78.0, "pl": 24.0, "pvp": 8.1, "dy": 0.019, "roe": 0.34, "roic": 0.22, "evebitda": 16.5, "liq2m": 60_000_000}),
    ("INTC", "Intel Corporation", "Intel Corporation", "Tecnologia (EUA)", "acao_us", "USD", "INTC", {"cotacao": 24.5, "pl": 28.0, "pvp": 1.0, "dy": 0.0, "roe": 0.03, "roic": 0.02, "evebitda": 14.0, "liq2m": 70_000_000}),
    ("QCOM", "QUALCOMM Incorporated", "QUALCOMM Incorporated", "Tecnologia (EUA)", "acao_us", "USD", "QCOM", {"cotacao": 164.0, "pl": 18.5, "pvp": 7.2, "dy": 0.021, "roe": 0.38, "roic": 0.26, "evebitda": 14.0, "liq2m": 65_000_000}),
    ("TXN", "Texas Instruments Inc.", "Texas Instruments Incorporated", "Tecnologia (EUA)", "acao_us", "USD", "TXN", {"cotacao": 202.0, "pl": 34.0, "pvp": 10.5, "dy": 0.027, "roe": 0.32, "roic": 0.24, "evebitda": 22.0, "liq2m": 55_000_000}),
    ("UBER", "Uber Technologies Inc.", "Uber Technologies, Inc.", "Tecnologia e Mobilidade (EUA)", "acao_us", "USD", "UBER", {"cotacao": 72.0, "pl": 32.0, "pvp": 9.5, "dy": 0.0, "roe": 0.28, "roic": 0.16, "evebitda": 24.0, "liq2m": 70_000_000}),
    ("GS", "The Goldman Sachs Group Inc.", "The Goldman Sachs Group, Inc.", "Financeiro e Bancos (EUA)", "acao_us", "USD", "GS", {"cotacao": 585.0, "pl": 15.5, "pvp": 1.6, "dy": 0.021, "roe": 0.11, "liq2m": 65_000_000}),
    ("MS", "Morgan Stanley", "Morgan Stanley", "Financeiro e Bancos (EUA)", "acao_us", "USD", "MS", {"cotacao": 132.0, "pl": 17.5, "pvp": 2.1, "dy": 0.028, "roe": 0.13, "liq2m": 60_000_000}),
    ("BLK", "BlackRock Inc.", "BlackRock, Inc.", "Financeiro e Gestão (EUA)", "acao_us", "USD", "BLK", {"cotacao": 1020.0, "pl": 24.5, "pvp": 3.5, "dy": 0.020, "roe": 0.15, "roic": 0.12, "evebitda": 17.5, "liq2m": 55_000_000}),
    ("PFE", "Pfizer Inc.", "Pfizer Inc.", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "PFE", {"cotacao": 26.5, "pl": 13.0, "pvp": 1.6, "dy": 0.062, "roe": 0.11, "roic": 0.08, "evebitda": 10.5, "liq2m": 65_000_000}),
    ("PLTR", "Palantir Technologies Inc.", "Palantir Technologies Inc.", "Tecnologia e IA (EUA)", "acao_us", "USD", "PLTR", {"cotacao": 62.0, "pl": 95.0, "pvp": 28.0, "dy": 0.0, "roe": 0.22, "roic": 0.18, "evebitda": 75.0, "liq2m": 85_000_000}),
    ("PANW", "Palo Alto Networks Inc.", "Palo Alto Networks, Inc.", "Tecnologia e Cibersegurança (EUA)", "acao_us", "USD", "PANW", {"cotacao": 385.0, "pl": 48.0, "pvp": 22.0, "dy": 0.0, "roe": 0.42, "roic": 0.24, "evebitda": 36.0, "liq2m": 60_000_000}),
    ("CRWD", "CrowdStrike Holdings Inc.", "CrowdStrike Holdings, Inc.", "Tecnologia e Cibersegurança (EUA)", "acao_us", "USD", "CRWD", {"cotacao": 345.0, "pl": 65.0, "pvp": 26.0, "dy": 0.0, "roe": 0.18, "roic": 0.15, "evebitda": 48.0, "liq2m": 60_000_000}),
    ("SNOW", "Snowflake Inc.", "Snowflake Inc.", "Tecnologia e Cloud (EUA)", "acao_us", "USD", "SNOW", {"cotacao": 168.0, "pl": 85.0, "pvp": 11.0, "dy": 0.0, "roe": 0.05, "roic": 0.04, "evebitda": 55.0, "liq2m": 55_000_000}),
    ("SHOP", "Shopify Inc.", "Shopify Inc.", "Tecnologia e E-commerce (EUA)", "acao_us", "USD", "SHOP", {"cotacao": 108.0, "pl": 58.0, "pvp": 13.0, "dy": 0.0, "roe": 0.21, "roic": 0.16, "evebitda": 42.0, "liq2m": 60_000_000}),
    ("MELI", "MercadoLibre Inc.", "MercadoLibre, Inc.", "Tecnologia e E-commerce (EUA/LatAm)", "acao_us", "USD", "MELI", {"cotacao": 1980.0, "pl": 52.0, "pvp": 24.0, "dy": 0.0, "roe": 0.44, "roic": 0.26, "evebitda": 29.0, "liq2m": 65_000_000}),
    ("NU", "Nu Holdings Ltd. (Nubank)", "Nu Holdings Ltd.", "Financeiro e Fintech (EUA/Brasil)", "acao_us", "USD", "NU", {"cotacao": 14.2, "pl": 32.0, "pvp": 8.9, "dy": 0.0, "roe": 0.28, "liq2m": 75_000_000}),
    ("TSM", "Taiwan Semiconductor (TSMC ADR)", "Taiwan Semiconductor Manufacturing Company Limited", "Tecnologia e Semicondutores (EUA)", "acao_us", "USD", "TSM", {"cotacao": 188.0, "pl": 27.0, "pvp": 7.2, "dy": 0.013, "roe": 0.28, "roic": 0.22, "evebitda": 15.0, "liq2m": 85_000_000}),
    ("ASML", "ASML Holding N.V. (ADR)", "ASML Holding N.V.", "Tecnologia e Semicondutores (EUA)", "acao_us", "USD", "ASML", {"cotacao": 685.0, "pl": 34.0, "pvp": 17.5, "dy": 0.010, "roe": 0.52, "roic": 0.38, "evebitda": 25.0, "liq2m": 70_000_000}),
    ("NVO", "Novo Nordisk A/S (ADR)", "Novo Nordisk A/S", "Saúde e Farmacêuticas (EUA)", "acao_us", "USD", "NVO", {"cotacao": 108.0, "pl": 33.0, "pvp": 29.0, "dy": 0.013, "roe": 0.85, "roic": 0.62, "evebitda": 23.0, "liq2m": 65_000_000}),
    ("SPOT", "Spotify Technology S.A.", "Spotify Technology S.A.", "Comunicação e Streaming (EUA)", "acao_us", "USD", "SPOT", {"cotacao": 465.0, "pl": 54.0, "pvp": 18.0, "dy": 0.0, "roe": 0.28, "roic": 0.21, "evebitda": 38.0, "liq2m": 55_000_000}),
    ("COIN", "Coinbase Global Inc.", "Coinbase Global, Inc.", "Financeiro e Cripto (EUA)", "acao_us", "USD", "COIN", {"cotacao": 295.0, "pl": 38.0, "pvp": 8.2, "dy": 0.0, "roe": 0.24, "roic": 0.16, "evebitda": 24.0, "liq2m": 75_000_000}),
    ("BA", "The Boeing Company", "The Boeing Company", "Indústria e Aeroespacial (EUA)", "acao_us", "USD", "BA", {"cotacao": 152.0, "pl": -15.0, "pvp": -6.0, "dy": 0.0, "roe": -0.20, "evebitda": 25.0, "liq2m": 65_000_000}),
    ("LMT", "Lockheed Martin Corporation", "Lockheed Martin Corporation", "Indústria e Defesa (EUA)", "acao_us", "USD", "LMT", {"cotacao": 535.0, "pl": 20.0, "pvp": 18.5, "dy": 0.024, "roe": 0.85, "roic": 0.26, "evebitda": 14.5, "liq2m": 55_000_000}),
    ("RTX", "RTX Corporation", "RTX Corporation", "Indústria e Defesa (EUA)", "acao_us", "USD", "RTX", {"cotacao": 120.0, "pl": 22.0, "pvp": 2.6, "dy": 0.021, "roe": 0.12, "roic": 0.09, "evebitda": 14.0, "liq2m": 55_000_000}),
    ("DE", "Deere & Company (John Deere)", "Deere & Company", "Indústria e Agronegócio (EUA)", "acao_us", "USD", "DE", {"cotacao": 435.0, "pl": 16.5, "pvp": 5.2, "dy": 0.014, "roe": 0.34, "roic": 0.16, "evebitda": 12.5, "liq2m": 50_000_000}),
    ("UNP", "Union Pacific Corporation", "Union Pacific Corporation", "Transporte e Logística (EUA)", "acao_us", "USD", "UNP", {"cotacao": 242.0, "pl": 21.5, "pvp": 9.2, "dy": 0.022, "roe": 0.44, "roic": 0.17, "evebitda": 14.5, "liq2m": 50_000_000}),
    ("NEE", "NextEra Energy Inc.", "NextEra Energy, Inc.", "Energia Elétrica e Renováveis (EUA)", "acao_us", "USD", "NEE", {"cotacao": 76.0, "pl": 22.0, "pvp": 3.1, "dy": 0.027, "roe": 0.14, "roic": 0.08, "evebitda": 15.0, "liq2m": 55_000_000}),
    ("LIN", "Linde plc", "Linde plc", "Materiais e Química (EUA)", "acao_us", "USD", "LIN", {"cotacao": 455.0, "pl": 31.0, "pvp": 5.6, "dy": 0.012, "roe": 0.17, "roic": 0.14, "evebitda": 19.0, "liq2m": 50_000_000}),
    ("SBUX", "Starbucks Corporation", "Starbucks Corporation", "Consumo e Restaurantes (EUA)", "acao_us", "USD", "SBUX", {"cotacao": 98.0, "pl": 27.0, "pvp": -14.0, "dy": 0.024, "roe": 0.35, "roic": 0.22, "evebitda": 17.5, "liq2m": 55_000_000}),
    ("ABNB", "Airbnb Inc.", "Airbnb, Inc.", "Consumo e Viagens (EUA)", "acao_us", "USD", "ABNB", {"cotacao": 136.0, "pl": 29.0, "pvp": 10.5, "dy": 0.0, "roe": 0.36, "roic": 0.28, "evebitda": 21.0, "liq2m": 55_000_000}),
    ("PYPL", "PayPal Holdings Inc.", "PayPal Holdings, Inc.", "Financeiro e Pagamentos (EUA)", "acao_us", "USD", "PYPL", {"cotacao": 84.0, "pl": 19.5, "pvp": 4.2, "dy": 0.0, "roe": 0.21, "roic": 0.15, "evebitda": 13.5, "liq2m": 55_000_000}),
    ("VZ", "Verizon Communications Inc.", "Verizon Communications Inc.", "Telecomunicações (EUA)", "acao_us", "USD", "VZ", {"cotacao": 42.5, "pl": 9.8, "pvp": 1.8, "dy": 0.064, "roe": 0.18, "roic": 0.09, "evebitda": 7.5, "liq2m": 55_000_000}),
    ("T", "AT&T Inc.", "AT&T Inc.", "Telecomunicações (EUA)", "acao_us", "USD", "T", {"cotacao": 22.8, "pl": 10.5, "pvp": 1.5, "dy": 0.049, "roe": 0.14, "roic": 0.08, "evebitda": 7.2, "liq2m": 55_000_000}),
    ("PM", "Philip Morris International", "Philip Morris International Inc.", "Consumo Básico (EUA)", "acao_us", "USD", "PM", {"cotacao": 130.0, "pl": 20.5, "pvp": 14.0, "dy": 0.041, "roe": 0.65, "roic": 0.25, "evebitda": 15.5, "liq2m": 50_000_000}),
    ("MO", "Altria Group Inc.", "Altria Group, Inc.", "Consumo Básico (EUA)", "acao_us", "USD", "MO", {"cotacao": 54.0, "pl": 10.2, "pvp": 12.0, "dy": 0.075, "roe": 0.80, "roic": 0.30, "evebitda": 9.0, "liq2m": 50_000_000}),

    # ── REITs (Real Estate Investment Trusts nos EUA — USD) ─────────────────────────
    ("O", "Realty Income Corporation", "Realty Income Corporation (The Monthly Dividend Company)", "REIT · Varejo e Renda Mensal", "reit", "USD", "O", {"cotacao": 57.5, "pl": 38.0, "pvp": 1.32, "dy": 0.055, "roe": 0.04, "roic": 0.04, "evebitda": 16.5, "liq2m": 65_000_000}),
    ("PLD", "Prologis Inc.", "Prologis, Inc.", "REIT · Logística e Galpões", "reit", "USD", "PLD", {"cotacao": 116.0, "pl": 34.0, "pvp": 1.98, "dy": 0.033, "roe": 0.06, "roic": 0.05, "evebitda": 21.0, "liq2m": 70_000_000}),
    ("AMT", "American Tower Corporation", "American Tower Corporation", "REIT · Torres e Telecomunicações", "reit", "USD", "AMT", {"cotacao": 208.0, "pl": 36.0, "pvp": 24.0, "dy": 0.031, "roe": 0.35, "roic": 0.08, "evebitda": 20.5, "liq2m": 60_000_000}),
    ("EQIX", "Equinix Inc.", "Equinix, Inc.", "REIT · Data Centers e Cloud", "reit", "USD", "EQIX", {"cotacao": 915.0, "pl": 72.0, "pvp": 6.5, "dy": 0.019, "roe": 0.09, "roic": 0.06, "evebitda": 25.0, "liq2m": 60_000_000}),
    ("WELL", "Welltower Inc.", "Welltower Inc.", "REIT · Saúde e Senior Housing", "reit", "USD", "WELL", {"cotacao": 135.0, "pl": 68.0, "pvp": 2.85, "dy": 0.020, "roe": 0.04, "roic": 0.04, "evebitda": 28.0, "liq2m": 55_000_000}),
    ("SPG", "Simon Property Group Inc.", "Simon Property Group, Inc.", "REIT · Shoppings e Outlets", "reit", "USD", "SPG", {"cotacao": 176.0, "pl": 22.0, "pvp": 18.5, "dy": 0.047, "roe": 0.78, "roic": 0.11, "evebitda": 16.0, "liq2m": 55_000_000}),
    ("PSA", "Public Storage", "Public Storage", "REIT · Self-Storage", "reit", "USD", "PSA", {"cotacao": 335.0, "pl": 31.0, "pvp": 10.8, "dy": 0.036, "roe": 0.36, "roic": 0.14, "evebitda": 19.5, "liq2m": 50_000_000}),
    ("DLR", "Digital Realty Trust Inc.", "Digital Realty Trust, Inc.", "REIT · Data Centers e Cloud", "reit", "USD", "DLR", {"cotacao": 178.0, "pl": 65.0, "pvp": 2.9, "dy": 0.027, "roe": 0.05, "roic": 0.04, "evebitda": 26.0, "liq2m": 55_000_000}),
    ("CCI", "Crown Castle Inc.", "Crown Castle Inc.", "REIT · Torres e Fibra", "reit", "USD", "CCI", {"cotacao": 104.0, "pl": 36.0, "pvp": 7.8, "dy": 0.060, "roe": 0.21, "roic": 0.07, "evebitda": 17.5, "liq2m": 50_000_000}),
    ("VICI", "VICI Properties Inc.", "VICI Properties Inc.", "REIT · Entretenimento e Experiências", "reit", "USD", "VICI", {"cotacao": 31.8, "pl": 12.8, "pvp": 1.28, "dy": 0.054, "roe": 0.10, "roic": 0.07, "evebitda": 14.5, "liq2m": 55_000_000}),
    ("EXR", "Extra Space Storage Inc.", "Extra Space Storage Inc.", "REIT · Self-Storage", "reit", "USD", "EXR", {"cotacao": 164.0, "pl": 38.0, "pvp": 2.45, "dy": 0.039, "roe": 0.06, "roic": 0.05, "evebitda": 20.0, "liq2m": 45_000_000}),
    ("AVB", "AvalonBay Communities Inc.", "AvalonBay Communities, Inc.", "REIT · Residencial (Multifamily)", "reit", "USD", "AVB", {"cotacao": 228.0, "pl": 30.0, "pvp": 2.75, "dy": 0.030, "roe": 0.09, "roic": 0.06, "evebitda": 19.0, "liq2m": 45_000_000}),
    ("EQR", "Equity Residential", "Equity Residential", "REIT · Residencial (Multifamily)", "reit", "USD", "EQR", {"cotacao": 74.0, "pl": 28.0, "pvp": 2.50, "dy": 0.036, "roe": 0.09, "roic": 0.06, "evebitda": 18.0, "liq2m": 45_000_000}),
    ("INVH", "Invitation Homes Inc.", "Invitation Homes Inc.", "REIT · Residencial (Single-Family)", "reit", "USD", "INVH", {"cotacao": 33.5, "pl": 38.0, "pvp": 2.05, "dy": 0.034, "roe": 0.05, "roic": 0.04, "evebitda": 19.5, "liq2m": 42_000_000}),
    ("ARE", "Alexandria Real Estate Equities", "Alexandria Real Estate Equities, Inc.", "REIT · Life Science e Laboratórios", "reit", "USD", "ARE", {"cotacao": 108.0, "pl": 35.0, "pvp": 1.02, "dy": 0.048, "roe": 0.03, "roic": 0.03, "evebitda": 16.0, "liq2m": 42_000_000}),
    ("VTR", "Ventas Inc.", "Ventas, Inc.", "REIT · Saúde e Hospitais", "reit", "USD", "VTR", {"cotacao": 64.0, "pl": 58.0, "pvp": 2.65, "dy": 0.028, "roe": 0.04, "roic": 0.04, "evebitda": 21.0, "liq2m": 42_000_000}),
    ("MAA", "Mid-America Apartment Communities", "Mid-America Apartment Communities, Inc.", "REIT · Residencial (Multifamily)", "reit", "USD", "MAA", {"cotacao": 156.0, "pl": 33.0, "pvp": 3.0, "dy": 0.038, "roe": 0.09, "roic": 0.06, "evebitda": 18.5, "liq2m": 40_000_000}),
    ("ESS", "Essex Property Trust Inc.", "Essex Property Trust, Inc.", "REIT · Residencial (Multifamily)", "reit", "USD", "ESS", {"cotacao": 298.0, "pl": 31.0, "pvp": 3.4, "dy": 0.033, "roe": 0.11, "roic": 0.07, "evebitda": 19.0, "liq2m": 38_000_000}),
    ("IRM", "Iron Mountain Incorporated", "Iron Mountain Incorporated", "REIT · Data Centers e Gestão de Dados", "reit", "USD", "IRM", {"cotacao": 118.0, "pl": 65.0, "pvp": 45.0, "dy": 0.024, "roe": 0.45, "roic": 0.10, "evebitda": 22.0, "liq2m": 45_000_000}),
    ("SBAC", "SBA Communications Corp.", "SBA Communications Corporation", "REIT · Torres e Telecomunicações", "reit", "USD", "SBAC", {"cotacao": 228.0, "pl": 38.0, "pvp": -5.0, "dy": 0.017, "roe": 0.25, "roic": 0.09, "evebitda": 19.5, "liq2m": 40_000_000}),
    ("WY", "Weyerhaeuser Company", "Weyerhaeuser Company", "REIT · Florestal e Madeira", "reit", "USD", "WY", {"cotacao": 31.0, "pl": 32.0, "pvp": 2.3, "dy": 0.026, "roe": 0.07, "roic": 0.06, "evebitda": 16.5, "liq2m": 38_000_000}),
    ("KIM", "Kimco Realty Corporation", "Kimco Realty Corporation", "REIT · Varejo e Shopping Centers", "reit", "USD", "KIM", {"cotacao": 24.5, "pl": 28.0, "pvp": 1.55, "dy": 0.041, "roe": 0.06, "roic": 0.05, "evebitda": 16.0, "liq2m": 38_000_000}),
    ("REG", "Regency Centers Corporation", "Regency Centers Corporation", "REIT · Varejo e Shopping Centers", "reit", "USD", "REG", {"cotacao": 73.0, "pl": 32.0, "pvp": 2.05, "dy": 0.037, "roe": 0.06, "roic": 0.05, "evebitda": 17.5, "liq2m": 35_000_000}),
    ("FRT", "Federal Realty Investment Trust", "Federal Realty Investment Trust", "REIT · Varejo e Dividend King", "reit", "USD", "FRT", {"cotacao": 112.0, "pl": 34.0, "pvp": 3.1, "dy": 0.039, "roe": 0.09, "roic": 0.06, "evebitda": 17.0, "liq2m": 35_000_000}),
    ("BXP", "BXP Inc. (Boston Properties)", "BXP, Inc.", "REIT · Lajes Corporativas", "reit", "USD", "BXP", {"cotacao": 78.0, "pl": 35.0, "pvp": 2.2, "dy": 0.050, "roe": 0.06, "roic": 0.04, "evebitda": 15.5, "liq2m": 38_000_000}),
    ("GLPI", "Gaming and Leisure Properties", "Gaming and Leisure Properties, Inc.", "REIT · Entretenimento e Hospitalidade", "reit", "USD", "GLPI", {"cotacao": 50.5, "pl": 17.5, "pvp": 3.1, "dy": 0.060, "roe": 0.18, "roic": 0.09, "evebitda": 14.5, "liq2m": 38_000_000}),
    ("STAG", "STAG Industrial Inc.", "STAG Industrial, Inc.", "REIT · Logística e Industrial (Mensal)", "reit", "USD", "STAG", {"cotacao": 37.5, "pl": 33.0, "pvp": 2.0, "dy": 0.039, "roe": 0.06, "roic": 0.05, "evebitda": 16.5, "liq2m": 35_000_000}),
    ("NNN", "NNN REIT Inc.", "NNN REIT, Inc.", "REIT · Varejo Triple-Net", "reit", "USD", "NNN", {"cotacao": 43.5, "pl": 20.0, "pvp": 1.85, "dy": 0.053, "roe": 0.09, "roic": 0.06, "evebitda": 15.0, "liq2m": 35_000_000}),
    ("WPC", "W. P. Carey Inc.", "W. P. Carey Inc.", "REIT · Industrial e Triple-Net", "reit", "USD", "WPC", {"cotacao": 56.5, "pl": 22.0, "pvp": 1.45, "dy": 0.061, "roe": 0.07, "roic": 0.05, "evebitda": 15.5, "liq2m": 38_000_000}),
    ("ADC", "Agree Realty Corporation", "Agree Realty Corporation", "REIT · Varejo e Renda Mensal", "reit", "USD", "ADC", {"cotacao": 74.5, "pl": 36.0, "pvp": 1.55, "dy": 0.041, "roe": 0.04, "roic": 0.04, "evebitda": 18.0, "liq2m": 35_000_000}),
    ("AGNC", "AGNC Investment Corp.", "AGNC Investment Corp.", "REIT · Hipotecas e Recebíveis (mREIT)", "reit", "USD", "AGNC", {"cotacao": 9.6, "pl": 11.0, "pvp": 1.05, "dy": 0.145, "roe": 0.10, "liq2m": 50_000_000}),
    ("NLY", "Annaly Capital Management", "Annaly Capital Management, Inc.", "REIT · Hipotecas e Recebíveis (mREIT)", "reit", "USD", "NLY", {"cotacao": 19.5, "pl": 12.0, "pvp": 0.98, "dy": 0.132, "roe": 0.09, "liq2m": 45_000_000}),
    ("VNQ", "Vanguard Real Estate ETF", "Vanguard Real Estate Index Fund ETF", "ETF · REITs Americanos", "etf", "USD", "VNQ", {"cotacao": 93.0, "dy": 0.038, "liq2m": 65_000_000}),
    ("XLRE", "Real Estate Select Sector SPDR ETF", "The Real Estate Select Sector SPDR Fund", "REIT · ETF Setorial Imobiliário S&P 500", "reit", "USD", "XLRE", {"cotacao": 43.2, "pl": 34.0, "pvp": 2.6, "dy": 0.033, "liq2m": 50_000_000}),
    ("SCHH", "Schwab U.S. REIT ETF", "Schwab U.S. REIT ETF", "REIT · ETF de REITs Americanos", "reit", "USD", "SCHH", {"cotacao": 22.4, "pl": 33.0, "pvp": 2.4, "dy": 0.034, "liq2m": 40_000_000}),
    ("REET", "iShares Global REIT ETF", "iShares Global REIT ETF", "REIT · ETF Global de REITs", "reit", "USD", "REET", {"cotacao": 25.2, "pl": 28.0, "pvp": 1.7, "dy": 0.039, "liq2m": 35_000_000}),

    # ── ETFs e BDRs na B3 + ETFs Internacionais (EUA) ───────────────────────────────
    ("BOVA11", "iShares Ibovespa ETF", "iShares Ibovespa Fundo de Índice", "ETF · Ações Brasil (Ibovespa)", "etf", "BRL", "BOVA11.SA", {"cotacao": 125.0, "pl": 8.5, "pvp": 1.35, "dy": 0.0, "liq2m": 95_000_000}),
    ("IVVB11", "iShares S&P 500 B3 ETF", "iShares S&P 500 Fundo de Investimento em Cotas de Fundo de Índice", "ETF · Internacional na B3 (S&P 500)", "etf", "BRL", "IVVB11.SA", {"cotacao": 365.0, "pl": 26.0, "pvp": 4.8, "dy": 0.0, "liq2m": 65_000_000}),
    ("SMAL11", "iShares Small Cap B3 ETF", "iShares BM&FBOVESPA Small Cap Fundo de Índice", "ETF · Small Caps Brasil", "etf", "BRL", "SMAL11.SA", {"cotacao": 98.0, "pl": 11.0, "pvp": 1.15, "dy": 0.0, "liq2m": 45_000_000}),
    ("XFIX11", "Trend ETF IFIX (FIIs B3)", "Trend ETF IFIX Fundo de Índice", "ETF · Fundos Imobiliários B3", "etf", "BRL", "XFIX11.SA", {"cotacao": 10.8, "dy": 0.0, "pvp": 0.95, "liq2m": 3_000_000}),
    ("ALUG11", "Investo MSCI US IMI Real Estate ETF", "Investo MSCI US IMI Real Estate ETF (REITs na B3)", "ETF · REITs Americanos na B3", "etf", "BRL", "ALUG11.SA", {"cotacao": 46.5, "dy": 0.035, "pvp": 2.2, "liq2m": 2_500_000}),
    ("DIVO11", "It Now IDIV (Dividendos B3) ETF", "It Now IDIV Fundo de Índice", "ETF · Dividendos Brasil", "etf", "BRL", "DIVO11.SA", {"cotacao": 94.0, "pl": 7.5, "pvp": 1.25, "dy": 0.0, "liq2m": 12_000_000}),
    ("WRLD11", "Investo FTSE Global All Cap ETF", "Investo FTSE Global All Cap ETF (VT na B3)", "ETF · Ações Globais na B3", "etf", "BRL", "WRLD11.SA", {"cotacao": 118.0, "pl": 21.0, "pvp": 3.1, "dy": 0.0, "liq2m": 15_000_000}),
    ("NASD11", "Trend ETF NASDAQ 100", "Trend ETF NASDAQ 100 Fundo de Índice", "ETF · Tecnologia EUA na B3", "etf", "BRL", "NASD11.SA", {"cotacao": 16.5, "pl": 32.0, "pvp": 7.5, "dy": 0.0, "liq2m": 18_000_000}),
    ("GOLD11", "Trend ETF LBMA Ouro", "Trend ETF LBMA Ouro Fundo de Índice", "ETF · Ouro na B3", "etf", "BRL", "GOLD11.SA", {"cotacao": 16.8, "dy": 0.0, "liq2m": 22_000_000}),
    ("HASH11", "Hashdex Nasdaq Crypto Index ETF", "Hashdex Nasdaq Crypto Index Fundo de Índice", "ETF · Criptoativos na B3", "etf", "BRL", "HASH11.SA", {"cotacao": 64.0, "dy": 0.0, "liq2m": 35_000_000}),
    ("BITH11", "Hashdex Nasdaq Bitcoin Reference Price ETF", "Hashdex Bitcoin ETF", "ETF · Bitcoin na B3", "etf", "BRL", "BITH11.SA", {"cotacao": 112.0, "dy": 0.0, "liq2m": 18_000_000}),
    ("AAPL34", "Apple Inc. (BDR B3)", "Apple Inc. Brazilian Depositary Receipt", "BDR · Tecnologia EUA na B3", "etf", "BRL", "AAPL34.SA", {"cotacao": 65.0, "pl": 31.5, "pvp": 45.0, "dy": 0.005, "roe": 1.45, "liq2m": 18_000_000}),
    ("MSFT34", "Microsoft Corp. (BDR B3)", "Microsoft Corporation BDR", "BDR · Tecnologia EUA na B3", "etf", "BRL", "MSFT34.SA", {"cotacao": 104.0, "pl": 34.0, "pvp": 11.8, "dy": 0.007, "roe": 0.36, "liq2m": 16_000_000}),
    ("NVDC34", "NVIDIA Corp. (BDR B3)", "NVIDIA Corporation BDR", "BDR · Tecnologia EUA na B3", "etf", "BRL", "NVDC34.SA", {"cotacao": 16.2, "pl": 48.0, "pvp": 42.0, "dy": 0.0003, "roe": 0.91, "liq2m": 35_000_000}),
    ("GOGL34", "Alphabet Inc. (BDR B3)", "Alphabet Inc. BDR", "BDR · Tecnologia EUA na B3", "etf", "BRL", "GOGL34.SA", {"cotacao": 86.0, "pl": 24.0, "pvp": 7.1, "dy": 0.004, "roe": 0.31, "liq2m": 15_000_000}),
    ("AMZO34", "Amazon.com Inc. (BDR B3)", "Amazon.com Inc. BDR", "BDR · Consumo e Cloud na B3", "etf", "BRL", "AMZO34.SA", {"cotacao": 59.0, "pl": 41.0, "pvp": 8.2, "dy": 0.0, "roe": 0.22, "liq2m": 16_000_000}),
    ("M1TA34", "Meta Platforms Inc. (BDR B3)", "Meta Platforms Inc. BDR", "BDR · Tecnologia EUA na B3", "etf", "BRL", "M1TA34.SA", {"cotacao": 118.0, "pl": 27.5, "pvp": 9.1, "dy": 0.003, "roe": 0.35, "liq2m": 14_000_000}),
    ("TSLA34", "Tesla Inc. (BDR B3)", "Tesla Inc. BDR", "BDR · Automóveis e Energia na B3", "etf", "BRL", "TSLA34.SA", {"cotacao": 46.0, "pl": 68.0, "pvp": 11.5, "dy": 0.0, "roe": 0.20, "liq2m": 22_000_000}),
    ("BERK34", "Berkshire Hathaway (BDR B3)", "Berkshire Hathaway Inc. BDR", "BDR · Financeiro EUA na B3", "etf", "BRL", "BERK34.SA", {"cotacao": 132.0, "pl": 15.2, "pvp": 1.6, "dy": 0.0, "roe": 0.12, "liq2m": 12_000_000}),
    ("MELI34", "MercadoLibre Inc. (BDR B3)", "MercadoLibre Inc. BDR", "BDR · E-commerce na B3", "etf", "BRL", "MELI34.SA", {"cotacao": 95.0, "pl": 52.0, "pvp": 24.0, "dy": 0.0, "roe": 0.44, "liq2m": 18_000_000}),
    ("ROXO34", "Nu Holdings / Nubank (BDR B3)", "Nu Holdings Ltd. BDR", "BDR · Financeiro e Fintech na B3", "etf", "BRL", "ROXO34.SA", {"cotacao": 13.8, "pl": 32.0, "pvp": 8.9, "dy": 0.0, "roe": 0.28, "liq2m": 32_000_000}),
    ("SPY", "SPDR S&P 500 ETF Trust", "SPDR S&P 500 ETF Trust", "ETF · S&P 500 (EUA)", "etf", "USD", "SPY", {"cotacao": 585.0, "pl": 27.0, "pvp": 4.9, "dy": 0.012, "liq2m": 100_000_000}),
    ("VOO", "Vanguard S&P 500 ETF", "Vanguard S&P 500 ETF", "ETF · S&P 500 (EUA)", "etf", "USD", "VOO", {"cotacao": 538.0, "pl": 27.0, "pvp": 4.9, "dy": 0.013, "liq2m": 95_000_000}),
    ("IVV", "iShares Core S&P 500 ETF", "iShares Core S&P 500 ETF", "ETF · S&P 500 (EUA)", "etf", "USD", "IVV", {"cotacao": 588.0, "pl": 27.0, "pvp": 4.9, "dy": 0.013, "liq2m": 90_000_000}),
    ("QQQ", "Invesco QQQ Trust (Nasdaq-100)", "Invesco QQQ Trust Series 1", "ETF · Tecnologia Nasdaq-100 (EUA)", "etf", "USD", "QQQ", {"cotacao": 502.0, "pl": 34.0, "pvp": 7.8, "dy": 0.006, "liq2m": 100_000_000}),
    ("VTI", "Vanguard Total Stock Market ETF", "Vanguard Total Stock Market Index Fund ETF", "ETF · Mercado Total EUA", "etf", "USD", "VTI", {"cotacao": 290.0, "pl": 26.0, "pvp": 4.4, "dy": 0.013, "liq2m": 85_000_000}),
    ("VT", "Vanguard Total World Stock ETF", "Vanguard Total World Stock Index Fund ETF", "ETF · Ações Globais (Mundo)", "etf", "USD", "VT", {"cotacao": 120.0, "pl": 21.0, "pvp": 3.1, "dy": 0.019, "liq2m": 75_000_000}),
    ("VEA", "Vanguard FTSE Developed Markets ETF", "Vanguard FTSE Developed Markets ETF", "ETF · Mercados Desenvolvidos ex-EUA", "etf", "USD", "VEA", {"cotacao": 51.0, "pl": 15.5, "pvp": 1.8, "dy": 0.030, "liq2m": 70_000_000}),
    ("VWO", "Vanguard FTSE Emerging Markets ETF", "Vanguard FTSE Emerging Markets ETF", "ETF · Mercados Emergentes", "etf", "USD", "VWO", {"cotacao": 46.5, "pl": 14.0, "pvp": 1.8, "dy": 0.031, "liq2m": 70_000_000}),
    ("SCHD", "Schwab U.S. Dividend Equity ETF", "Schwab U.S. Dividend Equity ETF", "ETF · Dividendos EUA", "etf", "USD", "SCHD", {"cotacao": 28.8, "pl": 17.0, "pvp": 3.2, "dy": 0.035, "liq2m": 80_000_000}),
    ("VYM", "Vanguard High Dividend Yield ETF", "Vanguard High Dividend Yield Index Fund ETF", "ETF · Dividendos EUA", "etf", "USD", "VYM", {"cotacao": 130.0, "pl": 18.0, "pvp": 2.8, "dy": 0.028, "liq2m": 65_000_000}),
    ("JEPI", "JPMorgan Equity Premium Income ETF", "JPMorgan Equity Premium Income ETF", "ETF · Renda Mensal e Opções (EUA)", "etf", "USD", "JEPI", {"cotacao": 59.0, "pl": 22.0, "pvp": 3.5, "dy": 0.073, "liq2m": 70_000_000}),
    ("GLD", "SPDR Gold Shares ETF", "SPDR Gold Trust", "ETF · Ouro (EUA)", "etf", "USD", "GLD", {"cotacao": 252.0, "dy": 0.0, "liq2m": 90_000_000}),
    ("IAU", "iShares Gold Trust ETF", "iShares Gold Trust", "ETF · Ouro (EUA)", "etf", "USD", "IAU", {"cotacao": 51.5, "dy": 0.0, "liq2m": 65_000_000}),
    ("SLV", "iShares Silver Trust ETF", "iShares Silver Trust", "ETF · Prata (EUA)", "etf", "USD", "SLV", {"cotacao": 29.5, "dy": 0.0, "liq2m": 60_000_000}),
    ("EWZ", "iShares MSCI Brazil ETF", "iShares MSCI Brazil ETF", "ETF · Brasil em Dólar (NYSE)", "etf", "USD", "EWZ", {"cotacao": 27.5, "pl": 8.2, "pvp": 1.3, "dy": 0.068, "liq2m": 75_000_000}),

    # ── Criptoativos, Commodities e Câmbio (Global) ─────────────────────────────────
    ("BTC-USD", "Bitcoin (BTC / USD)", "Bitcoin em Dólar Americano", "Criptoativos e Digitais", "outros", "USD", "BTC-USD", {"cotacao": 92000.0, "dy": 0.0, "liq2m": 100_000_000}),
    ("ETH-USD", "Ethereum (ETH / USD)", "Ethereum em Dólar Americano", "Criptoativos e Digitais", "outros", "USD", "ETH-USD", {"cotacao": 3150.0, "dy": 0.0, "liq2m": 95_000_000}),
    ("SOL-USD", "Solana (SOL / USD)", "Solana em Dólar Americano", "Criptoativos e Digitais", "outros", "USD", "SOL-USD", {"cotacao": 215.0, "dy": 0.0, "liq2m": 85_000_000}),
    ("GC=F", "Ouro Futuro (COMEX USD)", "Gold Futures COMEX", "Commodities e Metais Preciosos", "outros", "USD", "GC=F", {"cotacao": 2680.0, "dy": 0.0, "liq2m": 90_000_000}),
    ("SI=F", "Prata Futuro (COMEX USD)", "Silver Futures COMEX", "Commodities e Metais Preciosos", "outros", "USD", "SI=F", {"cotacao": 31.5, "dy": 0.0, "liq2m": 70_000_000}),
    ("BZ=F", "Petróleo Brent Futuro (USD)", "Brent Crude Oil Futures", "Commodities e Energia", "outros", "USD", "BZ=F", {"cotacao": 74.0, "dy": 0.0, "liq2m": 85_000_000}),
    ("USDBRL=X", "Dólar Comercial (USD / BRL)", "Taxa de Câmbio Dólar Americano para Real Brasileiro", "Moedas e Câmbio", "outros", "BRL", "BRL=X", {"cotacao": 5.75, "dy": 0.0, "liq2m": 100_000_000}),
)

EXTRA_ASSETS_BY_TICKER: dict[str, dict[str, Any]] = {
    ticker: {
        "ticker": ticker,
        "company": company,
        "legal_name": legal_name,
        "sector": sector,
        "asset_class": asset_class,
        "currency": currency,
        "yahoo_symbol": yahoo_symbol,
        "ref_metrics": ref_metrics,
    }
    for ticker, company, legal_name, sector, asset_class, currency, yahoo_symbol, ref_metrics in _CATALOG_ENTRIES
}
for ticker, (title, maturity) in TREASURY_TITLES.items():
    company = f"{title} {maturity.year}"
    EXTRA_ASSETS_BY_TICKER[ticker] = {
        "ticker": ticker,
        "company": company,
        "legal_name": company,
        "sector": "Tesouro Direto · Preços oficiais",
        "asset_class": "titulo",
        "currency": "BRL",
        "yahoo_symbol": None,
        "share_step": TREASURY_SHARE_STEP,
        "ref_metrics": {},
    }

# Dynamic runtime registry for any user-searched or custom ticker discovered during the session
_DYNAMIC_ASSETS_LOCK = threading.Lock()
_DYNAMIC_ASSETS: dict[str, dict[str, Any]] = {}

_B3_STOCK_PATTERN = re.compile(r"^[A-Z]{4}(?:3|4|5|6|11|31|32|33|34|35|39)$")
_VALID_TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=+]{0,23}$")
# Yahoo exchange codes (and their display names) for US listings; every other venue is a global listing.
_US_EQUITY_EXCHANGES = frozenset({
    "NYQ", "NYS", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS", "PNK", "OQX", "OQB", "OQS",
    "NYSE", "NASDAQ", "NYSEARCA", "NYSE ARCA", "AMEX", "NYSE AMERICAN", "BATS",
    "NASDAQGS", "NASDAQGM", "NASDAQCM", "OTC",
})
# Foreign venue suffix such as 7203.T (Tokyo) or VOD.L (London); .SA is B3 and handled earlier.
_FOREIGN_LISTING_PATTERN = re.compile(r"\.[A-Z]{1,3}$")


def _is_us_equity_listing(quote: dict[str, Any]) -> bool:
    """True only when the payload names a USD quote on a US venue.

    Missing metadata keeps the previous behaviour (US equity) so hand-built
    quotes and dynamic entries without exchange data are not reclassified.
    """
    currency = str(quote.get("currency") or "").strip().upper()
    exchange = str(quote.get("exchange") or "").strip().upper()
    if currency and currency != "USD":
        return False
    if exchange and exchange not in _US_EQUITY_EXCHANGES:
        return False
    return True


def _is_foreign_listing(ticker: str) -> bool:
    return bool(_FOREIGN_LISTING_PATTERN.search(ticker)) and not ticker.upper().endswith(".SA")


def register_dynamic_asset(entry: dict[str, Any]) -> dict[str, Any]:
    """Store a discovered asset in the process-wide dynamic catalog."""
    ticker = str(entry["ticker"]).strip().upper()
    with _DYNAMIC_ASSETS_LOCK:
        _DYNAMIC_ASSETS[ticker] = entry
    return entry


def parse_fixed_income_asset(raw: str) -> dict[str, Any] | None:
    """Parse canonical or user-typed fixed-income tickers into calculation specs.

    Examples supported:
    - CDI, SELIC, POUPANCA
    - TESOURO-SELIC, TESOURO-SELIC-2029
    - TESOURO-IPCA, TESOURO-IPCA-2035, TESOURO-IPCA-2045, TESOURO-RENDA+, TESOURO-EDUCA+
    - TESOURO-PRE, TESOURO-PRE-2027, TESOURO-PRE-2031
    - CDB-110CDI, CDB-105CDI, LCI-95CDI, LCA-95CDI, CDI-108%
    - CRI-CDI+2, CRA-CDI+1.5, CDI+2, CDI+2.5
    - IPCA+6, IPCA+6.5, DEB-IPCA+7
    - PRE-12, PRE-13.5
    - TREASURY-4.5 (USD fixed rate)
    """
    text = str(raw or "").strip().upper().replace(",", ".").replace(" ", "")
    if not text:
        return None
    if text in {"CDI", "100%CDI", "CDI100%"}:
        return {"ticker": "CDI", "kind": "cdi_pct", "rate": 100.0, "duration": 0.05, "currency": "BRL",
                "company": "CDI 100% (Renda Fixa Pós-fixada)", "sector": "Renda Fixa · Pós-fixado (CDI)"}
    if text in {"SELIC", "TESOURO-SELIC", "LFT"}:
        return {"ticker": text if text != "LFT" else "TESOURO-SELIC", "kind": "cdi_pct", "rate": 100.0, "duration": 0.08, "currency": "BRL",
                "company": "Tesouro Selic (LFT · Pós-fixado)", "sector": "Títulos Públicos · Tesouro Direto"}
    m = re.match(r"^TESOURO-SELIC-(\d{4})$", text)
    if m:
        year = int(m.group(1))
        return {"ticker": f"TESOURO-SELIC-{year}", "kind": "cdi_pct", "rate": 100.2, "duration": 0.1, "currency": "BRL",
                "company": f"Tesouro Selic {year} (LFT)", "sector": "Títulos Públicos · Tesouro Direto"}
    if text in {"POUPANCA", "POUPANÇA"}:
        return {"ticker": "POUPANCA", "kind": "poupanca", "rate": 70.0, "duration": 0.02, "currency": "BRL",
                "company": "Caderneta de Poupança", "sector": "Renda Fixa · Poupança"}
    if text in {"TESOURO-IPCA", "NTNB", "TESOURO-RENDA+", "TESOURO-EDUCA+"}:
        label = "Tesouro Renda+ (NTN-B1)" if "RENDA" in text else "Tesouro Educa+ (NTN-B1)" if "EDUCA" in text else "Tesouro IPCA+ (NTN-B Principal)"
        return {"ticker": text if text != "NTNB" else "TESOURO-IPCA", "kind": "ipca_plus", "rate": 6.5, "duration": 6.0, "currency": "BRL",
                "company": label, "sector": "Títulos Públicos · Tesouro Direto"}
    m = re.match(r"^TESOURO-IPCA-(\d{4})$", text)
    if m:
        year = int(m.group(1))
        dur = max(1.5, min(18.0, (year - 2026) * 0.72))
        return {"ticker": f"TESOURO-IPCA-{year}", "kind": "ipca_plus", "rate": 6.5, "duration": dur, "currency": "BRL",
                "company": f"Tesouro IPCA+ {year} (NTN-B Principal)", "sector": "Títulos Públicos · Tesouro Direto"}
    if text in {"TESOURO-PRE", "LTN"}:
        return {"ticker": "TESOURO-PRE", "kind": "prefixado", "rate": 13.2, "duration": 3.0, "currency": "BRL",
                "company": "Tesouro Prefixado (LTN)", "sector": "Títulos Públicos · Tesouro Direto"}
    m = re.match(r"^TESOURO-PRE-(\d{4})$", text)
    if m:
        year = int(m.group(1))
        dur = max(1.0, min(10.0, (year - 2026) * 0.85))
        return {"ticker": f"TESOURO-PRE-{year}", "kind": "prefixado", "rate": 13.3, "duration": dur, "currency": "BRL",
                "company": f"Tesouro Prefixado {year} (LTN)", "sector": "Títulos Públicos · Tesouro Direto"}
    m = re.match(r"^(?:(CDB|LCI|LCA|CRI|CRA|DEB|RDB|LC)-?)?(\d{2,3}(?:\.\d+)?)%?CDI$", text)
    if m:
        prefix = m.group(1) or "CDB"
        pct = float(m.group(2))
        if 50.0 <= pct <= 250.0:
            pct_str = f"{int(pct)}" if pct.is_integer() else f"{pct:g}"
            ticker = f"{prefix}-{pct_str}CDI"
            sector = "Renda Fixa · Isentos (LCI/LCA)" if prefix in {"LCI", "LCA"} else "Renda Fixa · Pós-fixado (CDI)"
            return {"ticker": ticker, "kind": "cdi_pct", "rate": pct, "duration": 0.06, "currency": "BRL",
                    "company": f"{prefix} ({pct_str}% do CDI)", "sector": sector}
    m = re.match(r"^(?:(CRI|CRA|DEB|CDB)-?)?CDI\+(\d{1,2}(?:\.\d+)?)%?$", text)
    if m:
        prefix = m.group(1)
        spread = float(m.group(2))
        if 0.0 <= spread <= 20.0:
            sp_str = f"{int(spread)}" if spread.is_integer() else f"{spread:g}"
            ticker = f"{prefix}-CDI+{sp_str}" if prefix else f"CDI+{sp_str}"
            label_prefix = f"{prefix} " if prefix else "Renda Fixa "
            return {"ticker": ticker, "kind": "cdi_spread", "rate": spread, "duration": 1.2, "currency": "BRL",
                    "company": f"{label_prefix}(CDI + {sp_str.replace('.', ',')}% a.a.)", "sector": "Renda Fixa · Crédito Privado"}
    m = re.match(r"^(?:(DEB|CRI|CRA|NTNB|TESOURO)-?)?IPCA\+(\d{1,2}(?:\.\d+)?)%?$", text)
    if m:
        prefix = m.group(1)
        real_rate = float(m.group(2))
        if 0.0 <= real_rate <= 25.0:
            r_str = f"{int(real_rate)}" if real_rate.is_integer() else f"{real_rate:g}"
            ticker = f"{prefix}-IPCA+{r_str}" if prefix and prefix not in {"NTNB", "TESOURO"} else f"IPCA+{r_str}"
            return {"ticker": ticker, "kind": "ipca_plus", "rate": real_rate, "duration": 4.0, "currency": "BRL",
                    "company": f"Renda Fixa (IPCA + {r_str.replace('.', ',')}% a.a.)", "sector": "Renda Fixa · Inflação (IPCA+)"}
    m = re.match(r"^(?:PRE|PREFIXADO|LTN)-?(\d{1,2}(?:\.\d+)?)%?$", text)
    if m:
        nom_rate = float(m.group(1))
        if 1.0 <= nom_rate <= 35.0:
            r_str = f"{int(nom_rate)}" if nom_rate.is_integer() else f"{nom_rate:g}"
            return {"ticker": f"PRE-{r_str}", "kind": "prefixado", "rate": nom_rate, "duration": 2.5, "currency": "BRL",
                    "company": f"Renda Fixa Prefixada ({r_str.replace('.', ',')}% a.a.)", "sector": "Renda Fixa · Prefixado"}
    m = re.match(r"^(?:TREASURY|TBILL|BOND)-(\d{1,2}(?:\.\d+)?)%?$", text)
    if m:
        usd_rate = float(m.group(1))
        if 0.5 <= usd_rate <= 20.0:
            r_str = f"{int(usd_rate)}" if usd_rate.is_integer() else f"{usd_rate:g}"
            return {"ticker": f"TREASURY-{r_str}", "kind": "usd_fixed", "rate": usd_rate, "duration": 2.0, "currency": "USD",
                    "company": f"US Treasury / Bond ({r_str.replace('.', ',')}% a.a. USD)", "sector": "Títulos EUA · Renda Fixa USD"}
    return None


def normalize_ticker(raw: str) -> str:
    """Normalize a user or API ticker string into its canonical B3Lab ticker."""
    cleaned = str(raw or "").strip().upper().removesuffix(".SA")
    if cleaned == "BRL=X":
        return "USDBRL=X"
    fi = parse_fixed_income_asset(cleaned)
    if fi is not None:
        return fi["ticker"]
    return cleaned


def is_synthetic_fixed_income(symbol_or_ticker: str) -> bool:
    """Whether the ticker is calculated from BCB/Tesouro curves instead of a Yahoo symbol."""
    cleaned = str(symbol_or_ticker or "").strip().upper().removesuffix(".SA")
    entry = EXTRA_ASSETS_BY_TICKER.get(cleaned)
    if entry is not None:
        return entry["yahoo_symbol"] is None and cleaned not in TREASURY_TITLES
    return parse_fixed_income_asset(cleaned) is not None


def _is_quoted_instrument(symbol: str) -> bool:
    """Indices, FX, futures, crypto and modeled titles are quoted, not traded in share lots."""
    cleaned = str(symbol or "").strip().upper()
    return (cleaned.startswith("^") or cleaned.endswith(("=F", "=X"))
            or cleaned.endswith("-USD") or "-USD" in cleaned)


def supports_share_allocation(symbol_or_ticker: str) -> bool:
    """Whether share-count allocation and purchase estimates apply to this asset.

    Foreign exchange-listed equities are `outros` (Global) but still trade in
    shares; only quoted instruments such as indices, FX, futures and crypto do not.
    """
    if is_synthetic_fixed_income(symbol_or_ticker):
        return False
    meta = asset_metadata(symbol_or_ticker)
    if meta["asset_class"] != "outros":
        return True
    return not _is_quoted_instrument(meta.get("yahoo_symbol") or symbol_or_ticker)


def asset_metadata(ticker: str, b3_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return full metadata for any B3, US, FII, REIT, Fixed Income, ETF, or dynamic ticker."""
    canonical = normalize_ticker(ticker)
    if canonical in EXTRA_ASSETS_BY_TICKER:
        return dict(EXTRA_ASSETS_BY_TICKER[canonical])
    if b3_frame is not None and not b3_frame.empty and canonical in set(b3_frame["Ticker"]):
        row = b3_frame[b3_frame["Ticker"] == canonical].iloc[0]
        return {
            "ticker": canonical,
            "company": str(row.get("Empresa", "") or canonical),
            "legal_name": str(row.get("RazaoSocial", "") or row.get("Empresa", "") or canonical),
            "sector": str(row.get("Setor", "") or "Ações B3"),
            "asset_class": "acao_br",
            "currency": "BRL",
            "yahoo_symbol": f"{canonical}.SA",
        }
    with _DYNAMIC_ASSETS_LOCK:
        if canonical in _DYNAMIC_ASSETS:
            return dict(_DYNAMIC_ASSETS[canonical])
    fi = parse_fixed_income_asset(canonical)
    if fi is not None:
        return {
            "ticker": fi["ticker"],
            "company": fi["company"],
            "legal_name": fi["company"],
            "sector": fi["sector"],
            "asset_class": "titulo",
            "currency": fi["currency"],
            "yahoo_symbol": None,
        }
    if _B3_STOCK_PATTERN.match(canonical):
        is_fii = canonical.endswith("11")
        is_bdr = canonical.endswith(("31", "32", "33", "34", "35", "39"))
        return {
            "ticker": canonical,
            "company": canonical,
            "legal_name": canonical,
            "sector": "FII · Fundo Imobiliário B3" if is_fii else "BDR · Ações Globais na B3" if is_bdr else "Ações B3",
            "asset_class": "fii" if is_fii else "etf" if is_bdr else "acao_br",
            "currency": "BRL",
            "yahoo_symbol": f"{canonical}.SA",
        }
    other_market = (canonical.startswith("^") or canonical.endswith(("=F", "=X", "-USD"))
                    or _is_foreign_listing(canonical))
    return {
        "ticker": canonical,
        "company": canonical,
        "legal_name": canonical,
        "sector": "Ativo Internacional / Global",
        "asset_class": "outros" if other_market else "acao_us",
        "currency": "BRL" if canonical == "^BVSP" else "USD",
        "yahoo_symbol": canonical,
    }


def modeled_history_notice(tickers) -> str | None:
    modeled = [ticker for ticker in tickers if is_synthetic_fixed_income(ticker)]
    if modeled:
        return f"{', '.join(modeled)}: retornos de renda fixa gerados pelo modelo B3Lab, não por cotações históricas."
    return None


def provider_symbol(ticker: str, b3_universe: set[str] | None = None) -> str:
    """Return the price-table column key / Yahoo symbol for a canonical ticker."""
    canonical = normalize_ticker(ticker)
    if canonical in EXTRA_ASSETS_BY_TICKER:
        sym = EXTRA_ASSETS_BY_TICKER[canonical]["yahoo_symbol"]
        return sym if sym is not None else canonical
    if b3_universe is not None and canonical in b3_universe:
        return f"{canonical}.SA"
    with _DYNAMIC_ASSETS_LOCK:
        if canonical in _DYNAMIC_ASSETS:
            sym = _DYNAMIC_ASSETS[canonical].get("yahoo_symbol")
            return sym if sym else canonical
    if parse_fixed_income_asset(canonical) is not None:
        return canonical
    if _B3_STOCK_PATTERN.match(canonical):
        return f"{canonical}.SA"
    return canonical


# Yahoo exchange suffixes whose quotes are not in US dollars. The suffix is the only
# reliable signal when a searched asset is not registered in this process yet, because
# asset_metadata() falls back to USD for unknown tickers.
_FOREIGN_SUFFIX_CURRENCIES = {
    ".T": "JPY",
    ".SW": "CHF",
    ".TO": "CAD",
    ".V": "CAD",
    ".AX": "AUD",
    ".PA": "EUR",
    ".AS": "EUR",
    ".DE": "EUR",
    ".F": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".BR": "EUR",
    ".HE": "EUR",
    ".IR": "EUR",
    ".LS": "EUR",
    ".VI": "EUR",
    ".L": "GBP",
    ".HK": "HKD",
    ".SI": "SGD",
    ".KS": "KRW",
    ".KQ": "KRW",
    ".NS": "INR",
    ".BO": "INR",
    ".SS": "CNY",
    ".SZ": "CNY",
    ".TW": "TWD",
    ".MX": "MXN",
}


def asset_currency(symbol: str) -> str:
    """Return the native ISO currency of a Yahoo symbol; BRL when it already trades in reais."""
    cleaned = str(symbol or "").strip().upper()
    if not cleaned:
        return "BRL"
    if cleaned.endswith(".SA") or cleaned in {"BRL=X", "USDBRL=X", "^BVSP"}:
        return "BRL"
    if is_synthetic_fixed_income(cleaned):
        spec = parse_fixed_income_asset(cleaned)
        return str((spec or {}).get("currency") or "BRL").upper()
    meta = asset_metadata(cleaned)
    reported = str(meta.get("currency") or "").strip()
    # Yahoo lists London shares in pence, not pounds; never scale them by the GBP rate.
    if reported in {"GBp", "GBX"}:
        return "GBX"
    suffix = "." + cleaned.rsplit(".", 1)[-1] if "." in cleaned else None
    suffix_currency = _FOREIGN_SUFFIX_CURRENCIES.get(suffix)
    if suffix_currency == "GBP":
        # Without reported metadata, a .L quote is assumed to be pence and excluded.
        return "GBP" if reported == "GBP" else "GBX"
    return suffix_currency or reported.upper() or "USD"


def is_usd_symbol(symbol: str) -> bool:
    """Return True when a Yahoo symbol is natively priced in USD and should convert to BRL."""
    return asset_currency(symbol) == "USD"


def is_valid_asset_ticker(ticker: str, b3_universe: set[str] | None = None) -> bool:
    """Validate that a ticker is either in the B3/multi-asset catalog, fixed income, or a valid market ticker."""
    canonical = normalize_ticker(ticker)
    if not canonical or len(canonical) > 24:
        return False
    if b3_universe is not None and canonical in b3_universe:
        return True
    if canonical in EXTRA_ASSETS_BY_TICKER:
        return True
    with _DYNAMIC_ASSETS_LOCK:
        if canonical in _DYNAMIC_ASSETS:
            return True
    if parse_fixed_income_asset(canonical) is not None:
        return True
    return bool(_VALID_TICKER_PATTERN.match(canonical))


def get_full_universe(b3_frame: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Combine the B3 CSV universe with all multi-asset classes (FIIs, US Stocks, REITs, Títulos, ETFs, BDRs, Global)."""
    if b3_frame is None:
        b3_frame = get_listed_stocks()
    seen: set[str] = set()
    stocks: list[dict[str, Any]] = []
    for row in b3_frame.to_dict("records"):
        ticker = str(row["Ticker"]).strip().upper()
        if ticker in EXTRA_ASSETS_BY_TICKER or ticker in seen:
            continue
        seen.add(ticker)
        stocks.append({
            "ticker": ticker,
            "company": row.get("Empresa", ""),
            "legal_name": row.get("RazaoSocial", ""),
            "sector": row.get("Setor", ""),
            "asset_class": "acao_br",
            "currency": "BRL",
            "supports_shares": True,
        })
    for ticker, item in EXTRA_ASSETS_BY_TICKER.items():
        if ticker in seen:
            continue
        seen.add(ticker)
        stocks.append({
            "ticker": ticker,
            "company": item["company"],
            "legal_name": item["legal_name"],
            "sector": item["sector"],
            "asset_class": item["asset_class"],
            "currency": item["currency"],
            "supports_shares": supports_share_allocation(ticker),
            **({"share_step": item["share_step"]} if "share_step" in item else {}),
        })
    with _DYNAMIC_ASSETS_LOCK:
        for ticker, item in _DYNAMIC_ASSETS.items():
            if ticker in seen:
                continue
            seen.add(ticker)
            stocks.append({
                "ticker": ticker,
                "company": item.get("company", ticker),
                "legal_name": item.get("legal_name", item.get("company", ticker)),
                "sector": item.get("sector", "Ativo Global"),
                "asset_class": item.get("asset_class", "acao_us"),
                "currency": item.get("currency", "USD"),
                "supports_shares": item.get("asset_class") != "outros",
            })
    return stocks


def build_synthetic_fixed_income_prices(
    symbols: list[str] | tuple[str, ...],
    start_date: dt.date,
    index: pd.DatetimeIndex | None = None,
    cdi_rates: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate daily price trajectories (base 100.0) for synthetic fixed-income instruments."""
    today = dt.date.today()
    if index is None or len(index) == 0:
        index = pd.bdate_range(start=start_date, end=today)
    if len(index) == 0:
        index = pd.bdate_range(end=today, periods=60)

    # Align daily CDI rates or fall back to 13.75% a.a.
    default_daily_cdi = (1.0 + 0.1375) ** (1.0 / 252.0) - 1.0
    if cdi_rates is not None and not cdi_rates.empty:
        aligned_cdi = cdi_rates.reindex(cdi_rates.index.union(index)).sort_index().ffill().bfill().reindex(index)
        aligned_cdi = aligned_cdi.fillna(default_daily_cdi)
    else:
        aligned_cdi = pd.Series(default_daily_cdi, index=index, dtype=float)

    ipca_daily = (1.0 + 0.0462) ** (1.0 / 252.0) - 1.0
    t = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(index=index, dtype=float)

    for idx_sym, sym in enumerate(symbols):
        spec = parse_fixed_income_asset(sym) or {
            "ticker": sym, "kind": "cdi_pct", "rate": 100.0, "duration": 0.05, "currency": "BRL"
        }
        kind = spec["kind"]
        rate = float(spec["rate"])
        duration = float(spec.get("duration", 0.1))
        phase = (idx_sym + 1) * 0.73

        if kind == "cdi_pct":
            daily_ret = aligned_cdi.to_numpy(dtype=float) * (rate / 100.0)
            # Micro-jitter guarantees non-zero variance for HRPOpt / covariance matrices when CDI is constant
            jitter = 1.5e-5 * np.sin(t * 0.19 + phase)
            daily_ret = daily_ret + jitter
        elif kind == "cdi_spread":
            daily_spread = (1.0 + rate / 100.0) ** (1.0 / 252.0) - 1.0
            daily_ret = (1.0 + aligned_cdi.to_numpy(dtype=float)) * (1.0 + daily_spread) - 1.0
            daily_ret = daily_ret + 2.5e-5 * np.sin(t * 0.23 + phase)
        elif kind == "poupanca":
            cdi_arr = aligned_cdi.to_numpy(dtype=float)
            poup_base = (1.0 + 0.0617) ** (1.0 / 252.0) - 1.0 + 0.12 * cdi_arr
            daily_ret = np.where(cdi_arr * 252 <= 0.085, 0.70 * cdi_arr, poup_base)
            daily_ret = daily_ret + 1.0e-5 * np.sin(t * 0.17 + phase)
        elif kind == "ipca_plus":
            real_daily = (1.0 + rate / 100.0) ** (1.0 / 252.0) - 1.0
            carry = (1.0 + ipca_daily) * (1.0 + real_daily) - 1.0
            # Duration mark-to-market oscillation around real-rate carry
            mtm = 0.00035 * duration * np.sin(t * 0.29 + phase) + 0.0002 * duration * np.cos(t * 0.13 + phase)
            daily_ret = carry + mtm
        elif kind == "prefixado":
            nom_daily = (1.0 + rate / 100.0) ** (1.0 / 252.0) - 1.0
            mtm = 0.00032 * duration * np.cos(t * 0.27 + phase) + 0.00018 * duration * np.sin(t * 0.11 + phase)
            daily_ret = nom_daily + mtm
        else:  # usd_fixed
            usd_daily = (1.0 + rate / 100.0) ** (1.0 / 252.0) - 1.0
            daily_ret = usd_daily + 0.0002 * duration * np.sin(t * 0.21 + phase)

        # Ensure first row starts at base 100.0 and subsequent rows compound daily_ret[1:]
        prices = np.empty(len(index), dtype=float)
        prices[0] = 100.0
        if len(index) > 1:
            prices[1:] = 100.0 * np.cumprod(1.0 + daily_ret[1:])
        frame[sym] = prices

    return frame


def _classify_yahoo_quote(quote: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Yahoo Finance search quote item into a B3Lab asset entry."""
    symbol = str(quote.get("symbol") or "").strip().upper()
    if not symbol or "=" in symbol and symbol not in {"GC=F", "SI=F", "CL=F", "BZ=F", "BRL=X", "USDBRL=X"}:
        if not symbol.endswith(("=F", "=X")):
            return None
    qtype = str(quote.get("quoteType") or "").upper()
    exchange = str(quote.get("exchange") or quote.get("exchDisp") or "").upper()
    long_name = quote.get("longname") or quote.get("longName")
    name = str(quote.get("shortname") or quote.get("shortName") or long_name or symbol).strip()
    sector = str(quote.get("sectorDisp") or quote.get("sector") or "").strip()
    description = f"{name} {long_name or ''} {quote.get('industryDisp') or quote.get('industry') or ''}".upper()

    is_b3 = symbol.endswith(".SA") or exchange in {"SAO", "BVMF", "B3"}
    canonical = normalize_ticker(symbol)
    if not _VALID_TICKER_PATTERN.match(canonical):
        return None
    if canonical in EXTRA_ASSETS_BY_TICKER:
        return dict(EXTRA_ASSETS_BY_TICKER[canonical])

    is_fund = qtype in {"ETF", "MUTUALFUND"}
    is_bond = qtype == "BOND" or (is_fund and bool(re.search(
        r"\b(TREASUR(?:Y|IES)|BONDS?|T-BILLS?|TIPS|FIXED INCOME|RENDA FIXA|IMA-B)\b", description)))

    if is_b3:
        currency = "BRL"
        yahoo_symbol = f"{canonical}.SA"
        if qtype not in {"ETF", "BOND"} and canonical.endswith("11") and re.search(r"\b(FII|FIAGRO|FI.INFRA|FUNDO IMOBILIÁRIO)\b", description):
            asset_class = "fii"
            sector = sector or "FII · Fundo Imobiliário B3"
        elif is_bond:
            asset_class = "titulo"
            sector = "Renda Fixa · Ativo listado"
        elif is_fund:
            asset_class = "etf"
            sector = sector or "ETF / Fundo B3"
        elif canonical.endswith(("31", "32", "33", "34", "35", "39")):
            asset_class = "etf"
            sector = sector or "BDR · Ações Globais na B3"
        elif qtype == "EQUITY":
            asset_class = "acao_br"
            sector = sector or "Ações B3"
        else:
            asset_class = "outros"
            sector = sector or "Outros ativos B3"
    else:
        currency = "USD"
        yahoo_symbol = canonical
        if is_bond:
            asset_class = "titulo"
            sector = "Títulos EUA · Treasuries e Bonds"
        elif is_fund:
            asset_class = "etf"
            sector = f"ETF · {sector or 'Internacional'}"
        elif qtype == "EQUITY" and re.search(r"\bREITS?\b", description):
            asset_class = "reit"
            sector = f"REIT · {sector or 'Imobiliário EUA'}"
        elif qtype == "EQUITY" and _is_us_equity_listing(quote):
            asset_class = "acao_us"
            sector = f"{sector} (EUA)" if sector else "Ações Americanas (EUA)"
        elif qtype == "EQUITY":
            # Tokyo, London, Toronto and other venues are global listings, never US equities.
            asset_class = "outros"
            sector = f"{sector} · Global" if sector else "Ações Internacionais (Global)"
        else:
            asset_class = "outros"
            sector = sector or "Criptoativos, moedas e índices"

    return {
        "ticker": canonical,
        "company": name,
        "legal_name": str(long_name or name),
        "sector": sector,
        "asset_class": asset_class,
        "currency": str(quote.get("currency") or currency).upper(),
        "yahoo_symbol": yahoo_symbol,
        "quote_type": qtype,
    }


@cache_data(ttl=1800)
def search_yahoo_assets(query: str) -> list[dict[str, Any]]:
    """Query Yahoo Finance's autocomplete search endpoint for any global ticker or company name."""
    cleaned = str(query or "").strip()
    if len(cleaned) < 2:
        return []
    try:
        response = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": cleaned, "quotesCount": 12, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=4,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes") or []
    except Exception:
        logger.debug("Yahoo search unavailable for q=%s", cleaned, exc_info=True)
        return []

    results: list[dict[str, Any]] = []
    for quote in quotes:
        classified = _classify_yahoo_quote(quote)
        if classified is not None:
            register_dynamic_asset(classified)
            results.append(classified)
    return results


_NASDAQ_HEADERS = {
    "Accept": "application/json",
    "Referer": "https://www.nasdaq.com/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}


def _finite_num(val: Any) -> float | None:
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_yahoo_info(symbol: str) -> dict[str, Any]:
    """Shared Yahoo payload or {} — a failed lookup never blocks the quote."""
    try:
        return get_yahoo_info(symbol) or {}
    except Exception:
        logger.warning("Yahoo Finance info unavailable for %s", symbol, exc_info=True)
        return {}


def _quote_metadata(symbol: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Explicit quote unit and the conventions Yahoo actually publishes.

    Indices are quoted in points, never currency. FX pairs expose the base/quote
    convention, crypto and futures their own volume window, and futures the
    contract expiry only when Yahoo returns it.
    """
    info = info if isinstance(info, dict) else {}
    upper = str(symbol or "").upper()
    pair = None
    if upper.endswith("=X"):
        code = upper[:-2]
        if len(code) == 6 and code.isalpha():
            pair = f"{code[:3]}/{code[3:]}"
        elif len(code) == 3 and code.isalpha():
            pair = f"USD/{code}"
    volume_24h = _finite_num(info.get("volume24Hr"))
    if volume_24h is not None and volume_24h <= 0:
        volume_24h = None
    session_volume = _finite_num(info.get("regularMarketVolume")) or _finite_num(info.get("volume"))
    if session_volume is not None and session_volume <= 0:
        session_volume = None
    contract = None
    if upper.endswith("=F"):
        expiry = str(info.get("expireIsoDate") or "").strip()[:10] or None
        if expiry is None:
            epoch = _finite_num(info.get("expireDate"))
            if epoch is not None and epoch > 0:
                try:
                    expiry = dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date().isoformat()
                except (OSError, OverflowError, ValueError):
                    expiry = None
        contract = {
            "symbol": str(info.get("underlyingSymbol") or "").strip() or None,
            "expiry": expiry,
            "unit": str(info.get("contractUnit") or info.get("contractSize") or "").strip() or None,
            "open_interest": _finite_num(info.get("openInterest")),
        }
    return {
        "price_unit": "points" if upper.startswith("^") else "money",
        "quote_pair": pair,
        "volume_24h": volume_24h,
        "session_volume": session_volume,
        "volume_window": "24h" if volume_24h is not None else ("sessão" if session_volume is not None else None),
        "contract": contract,
    }


def _is_financial_institution(sector: str | None, industry: str | None, *, ebitda: float | None = None, current_ratio: float | None = None) -> bool:
    text = f"{sector or ''} {industry or ''}".casefold()
    if any(term in text for term in ("bank", "insurance", "capital markets", "credit services", "asset management", "financial conglomerates")):
        return True
    return ("financial" in (sector or "").casefold() or (sector or "").casefold() == "finance") and ebitda is None and current_ratio is None


def _yahoo_pvp_and_equity(info: dict[str, Any], price: float | None, market_cap: float | None) -> tuple[float | None, float | None]:
    pvp = _finite_num(info.get("priceToBook"))
    book_value = _finite_num(info.get("bookValue"))
    # Yahoo mixes Class A book value with Class B price on dual-class shares like BRK-B.
    if pvp is not None and 0 < pvp < 0.05 and price and book_value and book_value > price * 100:
        total_debt = _finite_num(info.get("totalDebt"))
        debt_to_equity = _finite_num(info.get("debtToEquity"))
        if total_debt is not None and total_debt > 0 and debt_to_equity is not None and debt_to_equity > 0:
            equity = total_debt / (debt_to_equity / 100.0)
            if market_cap is not None and market_cap > 0 and equity > 0:
                return market_cap / equity, equity
        return None, None
    equity = (market_cap / pvp) if (market_cap is not None and market_cap > 0 and pvp is not None and pvp > 0) else None
    return pvp, equity


def _yahoo_dy_percent(info: dict[str, Any], price: float | None) -> float | None:
    field = "yield" if info.get("quoteType") in {"ETF", "MUTUALFUND"} else "trailingAnnualDividendYield"
    raw = _finite_num(info.get(field))
    raw_dy = raw * 100.0 if raw is not None else None
    if raw_dy is None or raw_dy == 0.0:
        # Yahoo Finance often returns trailingAnnualDividendYield=0.0 for FIIs or omits it
        # while reporting the 12m yield in percentage points under `dividendYield`.
        fallback_dy = _finite_num(info.get("dividendYield"))
        if fallback_dy is not None and fallback_dy > 0:
            raw_dy = fallback_dy
    if raw_dy is None and price is not None and price > 0:
        rate = _finite_num(info.get("trailingAnnualDividendRate"))
        if rate is not None and rate >= 0:
            raw_dy = (rate / price) * 100.0
    return raw_dy


def yahoo_equity_screener_row(ticker: str, meta: dict[str, Any], info: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Yahoo Finance info dict into a normalized US-equity Screener row (fractions)."""
    if not info:
        return None
    price = _finite_num(info.get("regularMarketPrice")) or _finite_num(info.get("currentPrice")) or _finite_num(info.get("previousClose"))
    if price is None or price <= 0:
        return None
    market_cap = _finite_num(info.get("marketCap"))
    pvp, _ = _yahoo_pvp_and_equity(info, price, market_cap)
    pl = _finite_num(info.get("trailingPE"))
    if pl is None:
        eps = _finite_num(info.get("trailingEps"))
        if eps is not None and eps != 0:
            pl = price / eps
    ev_ebitda = _finite_num(info.get("enterpriseToEbitda"))
    ev = _finite_num(info.get("enterpriseValue"))
    ebitda = _finite_num(info.get("ebitda"))
    if ev_ebitda is None and ev is not None and ev > 0 and ebitda is not None and ebitda != 0:
        ev_ebitda = ev / ebitda
    if ev is not None and ev <= 0 and ev_ebitda is not None and ev_ebitda < 0:
        ev_ebitda = None
    roe = _finite_num(info.get("returnOnEquity"))
    if pvp is not None and pvp < 0:
        pvp = None
        roe = None
    dy_pct = _yahoo_dy_percent(info, price)
    dy = dy_pct / 100.0 if dy_pct is not None else None
    avg_vol = _finite_num(info.get("averageDailyVolume3Month")) or _finite_num(info.get("averageVolume"))
    liq2m = avg_vol * price if avg_vol is not None and avg_vol > 0 else None
    return {
        "ticker": ticker,
        "currency": str(info.get("currency") or meta.get("currency") or "USD").upper(),
        "asset_class": meta["asset_class"],
        "setor": meta["sector"],
        "cotacao": price,
        "pl": pl,
        "pvp": pvp,
        "dy": dy,
        "roe": roe,
        "evebitda": ev_ebitda,
        "mrgliq": _finite_num(info.get("profitMargins")),
        "mrgebit": _finite_num(info.get("operatingMargins")),
        "liq2m": liq2m,
    }


def _parse_nasdaq_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"--", "N/A", "NaN", "∞", "-∞"}:
        return None
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip()
    return _finite_num(cleaned)


def _nasdaq_table_latest(table: dict[str, Any] | None) -> dict[str, float]:
    rows = (table or {}).get("rows") if isinstance(table, dict) else None
    if not isinstance(rows, list):
        return {}
    values: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("value1") or "").strip()
        if not label or label in values:
            continue
        number = _parse_nasdaq_number(row.get("value2"))
        if number is not None:
            values[label] = number
    return values


def _fetch_nasdaq_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    try:
        response = requests.get(
            f"https://api.nasdaq.com/api/{path}",
            params=params,
            headers=_NASDAQ_HEADERS,
            timeout=5,
        )
        response.raise_for_status()
        data = response.json().get("data")
        return data if isinstance(data, dict) else {}
    except (requests.RequestException, AttributeError, TypeError, ValueError):
        logger.debug("Nasdaq endpoint %s unavailable", path, exc_info=True)
        return {}


@cache_data(ttl=60, cache_if=bool)
def _get_nasdaq_quote(symbol: str, asset_class: str, currency: str) -> dict[str, Any]:
    """Get a USD quote from Nasdaq for a US stock, REIT, or listed ETF."""
    if currency != "USD" or not re.fullmatch(r"[A-Z]+(?:-[A-Z])?", symbol):
        return {}
    if asset_class in {"acao_us", "reit"}:
        instrument = "stocks"
    elif asset_class in {"etf", "titulo"}:
        instrument = "etf"
    else:
        return {}

    nasdaq_symbol = symbol.replace("-", ".")
    try:
        response = requests.get(
            f"https://api.nasdaq.com/api/quote/{nasdaq_symbol}/info",
            params={"assetclass": instrument},
            headers=_NASDAQ_HEADERS,
            timeout=4,
        )
        response.raise_for_status()
        data = response.json().get("data") or {}
        if data.get("symbol") != nasdaq_symbol or data.get("assetClass") != instrument.upper():
            return {}
        primary = data.get("primaryData") or {}
        sale_price = primary.get("lastSalePrice") or ""
        if not sale_price.startswith("$") or primary.get("currency") not in (None, "", "USD"):
            return {}
        price = float(sale_price[1:].replace(",", ""))
        if not math.isfinite(price) or price <= 0:
            return {}
    except (requests.RequestException, AttributeError, TypeError, ValueError):
        logger.debug("Nasdaq quote unavailable for %s", symbol, exc_info=True)
        return {}

    quote = {"price": price, "currency": "USD", "source": "Nasdaq"}
    stats = data.get("keyStats")
    week_range = stats.get("fiftyTwoWeekHighLow") if isinstance(stats, dict) else None
    if isinstance(week_range, dict):
        range_text = str(week_range.get("value")).replace(",", "")
        bounds = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", range_text)
        if bounds:
            low, high = map(float, bounds.groups())
            if 0 < low <= high and math.isfinite(high):
                quote.update(low_52w=low, high_52w=high)
    quote_date = re.search(r"[A-Za-z]{3} \d{1,2}, \d{4}", str(primary.get("lastTradeTimestamp")))
    if quote_date:
        try:
            quote["last_quote_date"] = dt.datetime.strptime(quote_date[0], "%b %d, %Y").date().isoformat()
        except ValueError:
            pass
    return quote


def _nasdaq_summary_value(summary: dict[str, Any], key: str) -> Any:
    summary_data = summary.get("summaryData") if isinstance(summary.get("summaryData"), dict) else {}
    entry = summary_data.get(key)
    return entry.get("value") if isinstance(entry, dict) else None


@cache_data(ttl=900, max_entries=128, cache_if=bool)
def nasdaq_equity_indicators(symbol: str) -> dict[str, float]:
    """Annual Nasdaq ratios in Yahoo units: multiples, and ROE/margins as fractions.

    Dividend yield is left out. Nasdaq publishes it only for Nasdaq-listed names,
    so the screener uses trailing dividends from Yahoo's public chart instead.
    Banks and insurers do not get an EV/EBITDA; that multiple does not describe them.
    """
    if not re.fullmatch(r"[A-Z]+(?:-[A-Z])?", symbol):
        return {}
    nasdaq_symbol = symbol.replace("-", ".")
    try:
        summary = _fetch_nasdaq_json(f"quote/{nasdaq_symbol}/summary", {"assetclass": "stocks"})
        financials = _fetch_nasdaq_json(f"company/{nasdaq_symbol}/financials", {"frequency": "1"})
    except Exception:
        logger.debug("Nasdaq screener indicators unavailable for %s", symbol, exc_info=True)
        return {}
    if not summary and not financials:
        return {}

    market_cap = _parse_nasdaq_number(_nasdaq_summary_value(summary, "MarketCap"))
    sector = str(_nasdaq_summary_value(summary, "Sector") or "").strip()
    industry = str(_nasdaq_summary_value(summary, "Industry") or "").strip()
    ratios = _nasdaq_table_latest(financials.get("financialRatiosTable"))
    income = _nasdaq_table_latest(financials.get("incomeStatementTable"))
    balance = _nasdaq_table_latest(financials.get("balanceSheetTable"))
    cashflow = _nasdaq_table_latest(financials.get("cashFlowTable"))

    net_income_k = income.get("Net Income Applicable to Common Shareholders", income.get("Net Income"))
    equity_k = balance.get("Total Equity")
    info: dict[str, float] = {}
    if market_cap and net_income_k and net_income_k != 0:
        info["trailingPE"] = market_cap / (net_income_k * 1000.0)
    if market_cap and equity_k and equity_k != 0:
        info["priceToBook"] = market_cap / (equity_k * 1000.0)
    if market_cap:
        info["marketCap"] = market_cap

    def fraction(label: str) -> float | None:
        value = ratios.get(label)
        return None if value is None else value / 100.0

    for label, key in (("After Tax ROE", "returnOnEquity"), ("Profit Margin", "profitMargins"), ("Operating Margin", "operatingMargins")):
        value = fraction(label)
        if value is not None:
            info[key] = value

    if not _is_financial_institution(sector, industry):
        ebit_k = income.get("Operating Income")
        depreciation_k = cashflow.get("Depreciation")
        cash_k = balance.get("Cash and Cash Equivalents")
        short_debt = balance.get("Short-Term Debt / Current Portion of Long-Term Debt")
        long_debt = balance.get("Long-Term Debt")
        if (market_cap and ebit_k is not None and depreciation_k is not None and cash_k is not None
                and (short_debt is not None or long_debt is not None)):
            ebitda = (ebit_k + depreciation_k) * 1000.0
            net_debt = ((short_debt or 0.0) + (long_debt or 0.0) - cash_k - (balance.get("Short-Term Investments") or 0.0)) * 1000.0
            enterprise = market_cap + net_debt
            if ebitda != 0 and enterprise > 0:
                info["enterpriseToEbitda"] = enterprise / ebitda
    return info


@cache_data(ttl=300, cache_if=bool)
def _get_nasdaq_fundamentals(symbol: str, asset_class: str, currency: str) -> dict[str, Any]:
    """Fetch quote, summary, and annual financial statements from Nasdaq in parallel."""
    if currency != "USD" or asset_class not in {"acao_us", "reit"} or not re.fullmatch(r"[A-Z]+(?:-[A-Z])?", symbol):
        return {}
    nasdaq_symbol = symbol.replace("-", ".")
    with ThreadPoolExecutor(max_workers=3) as pool:
        quote_future = pool.submit(_get_nasdaq_quote, symbol, asset_class, currency)
        summary_future = pool.submit(_fetch_nasdaq_json, f"quote/{nasdaq_symbol}/summary", {"assetclass": "stocks"})
        financials_future = pool.submit(_fetch_nasdaq_json, f"company/{nasdaq_symbol}/financials", {"frequency": "1"})
        quote = quote_future.result()
        if not quote:
            return {}
        summary = summary_future.result()
        financials = financials_future.result()

    summary_data = summary.get("summaryData") if isinstance(summary.get("summaryData"), dict) else {}
    def _summary_val(key: str) -> Any:
        entry = summary_data.get(key)
        return entry.get("value") if isinstance(entry, dict) else None

    price = quote["price"]
    market_cap = _parse_nasdaq_number(_summary_val("MarketCap"))
    share_volume = _parse_nasdaq_number(_summary_val("ShareVolume"))
    volume_2m = share_volume * price if share_volume is not None and share_volume > 0 else None
    dy = _parse_nasdaq_number(_summary_val("Yield"))
    sector_text = str(_summary_val("Sector") or "").strip()
    industry_text = str(_summary_val("Industry") or "").strip()

    ratios = _nasdaq_table_latest(financials.get("financialRatiosTable"))
    income = _nasdaq_table_latest(financials.get("incomeStatementTable"))
    balance = _nasdaq_table_latest(financials.get("balanceSheetTable"))

    # Nasdaq financial statement dollar lines are reported in $ thousands.
    net_income_k = income.get("Net Income Applicable to Common Shareholders")
    if net_income_k is None:
        net_income_k = income.get("Net Income")
    ebit_k = income.get("Operating Income")
    equity_k = balance.get("Total Equity")

    pl = market_cap / (net_income_k * 1000.0) if market_cap and net_income_k and net_income_k != 0 else None
    pvp = market_cap / (equity_k * 1000.0) if market_cap and equity_k and equity_k != 0 else None
    roe = ratios.get("After Tax ROE")
    net_margin = ratios.get("Profit Margin")
    ebit_margin = ratios.get("Operating Margin")
    current_ratio_pct = ratios.get("Current Ratio")
    current_ratio = current_ratio_pct / 100.0 if current_ratio_pct is not None else None

    candidates = {
        "pl": pl,
        "pvp": pvp,
        "ev_ebitda": None,
        "roe": roe,
        "net_margin": net_margin,
        "ebit_margin": ebit_margin,
        "dy": dy,
    }
    indicators = {key: value for key, value in candidates.items() if key in ASSET_INDICATORS[asset_class]}

    health_data: dict[str, float | None] = {}
    if asset_class in EQUITY_CLASSES:
        if current_ratio is not None:
            health_data["current_ratio"] = current_ratio
        if not _is_financial_institution(sector_text, industry_text, current_ratio=current_ratio):
            cash_k = balance.get("Cash and Cash Equivalents")
            st_inv_k = balance.get("Short-Term Investments") or 0.0
            st_debt_k = balance.get("Short-Term Debt / Current Portion of Long-Term Debt")
            lt_debt_k = balance.get("Long-Term Debt")
            total_debt_k = (st_debt_k or 0.0) + (lt_debt_k or 0.0) if (st_debt_k is not None or lt_debt_k is not None) else None
            total_cash_k = (cash_k + st_inv_k) if cash_k is not None else None
            if total_debt_k is not None and total_cash_k is not None and equity_k is not None and equity_k > 0:
                health_data["debt_equity"] = (total_debt_k - total_cash_k) / equity_k
            if market_cap and total_debt_k is not None and total_cash_k is not None and ebit_k is not None and ebit_k != 0:
                ev = market_cap + (total_debt_k - total_cash_k) * 1000.0
                if ev > 0:
                    health_data["ev_ebit"] = ev / (ebit_k * 1000.0)

    detail = {
        **quote,
        "market_cap": market_cap,
        "volume_2m": volume_2m,
        "indicators": indicators,
        "health": health_data,
    }
    if industry_text and industry_text not in {"N/A", "--"}:
        detail["subsector"] = industry_text
    return detail


def get_extra_asset_fundamentals(
    ticker: str, *, fii_table: pd.DataFrame | None = None, b3_stock: dict | None = None,
) -> dict[str, Any]:
    """Load the market and fundamental data applicable to a non-Fundamentus asset."""
    canonical = normalize_ticker(ticker)
    meta = b3_stock or asset_metadata(canonical)
    if canonical in TREASURY_TITLES:
        try:
            price, quote_date = latest_treasury_quote(canonical)
        except ValueError:
            logger.warning("Tesouro Transparente data unavailable for %s", canonical, exc_info=True)
            price, quote_date = None, None
        try:
            treasury = latest_treasury_detail(canonical)
        except (ValueError, KeyError):
            logger.warning("Tesouro Transparente detail unavailable for %s", canonical, exc_info=True)
            treasury = None
        result = {
            "ticker": canonical,
            "available": price is not None,
            "company": meta["company"],
            "sector": meta["sector"],
            "asset_class": "titulo",
            "currency": "BRL",
            "price": price,
            "last_quote_date": quote_date,
            "source": "Tesouro Transparente · PU Compra Manhã",
            **_quote_metadata(canonical),
        }
        if treasury:
            result["treasury"] = treasury
        return result
    spec = parse_fixed_income_asset(canonical)
    if spec is not None:
        terms = {
            "cdi_pct": ("Percentual do CDI", "% do CDI"),
            "cdi_spread": ("Adicional sobre o CDI", "% a.a."),
            "ipca_plus": ("Taxa real acima do IPCA", "% a.a."),
            "prefixado": ("Taxa prefixada", "% a.a."),
            "poupanca": ("Poupança", None),
            "usd_fixed": ("Taxa prefixada em USD", "% a.a."),
        }
        label, unit = terms[spec["kind"]]
        return {
            "ticker": canonical,
            "available": True,
            "company": meta["company"],
            "sector": meta["sector"],
            "asset_class": "titulo",
            "currency": meta["currency"],
            "fixed_income": {
                "label": label,
                "rate": spec["rate"] if unit else None,
                "unit": unit,
                **fund_data.synthetic_income_reference(spec),
            },
            "indicators": {},
            "source": "Premissas do modelo B3Lab",
            **_quote_metadata(canonical),
        }

    symbol = provider_symbol(canonical)
    if uses_market_quote(canonical):
        return market_asset_detail(canonical, get_yahoo_quotes((symbol,)).get(symbol), info=_safe_yahoo_info(symbol))

    if meta["asset_class"] == "fii":
        data = _fii_detail(canonical, meta, load_fii_market_data() if fii_table is None else fii_table)
        if data:
            return data

    info: dict[str, Any] = {}
    try:
        info = get_yahoo_info(symbol)
    except Exception:
        logger.warning("Yahoo Finance data unavailable for %s (%s)", canonical, symbol, exc_info=True)

    if info.get("quoteType") and not b3_stock and canonical not in EXTRA_ASSETS_BY_TICKER:
        classified = _classify_yahoo_quote({**info, "symbol": symbol})
        if classified is not None:
            meta = register_dynamic_asset(classified)
    asset_class = meta["asset_class"]

    def _pct(field: str) -> float | None:
        value = _finite_num(info.get(field))
        return value * 100.0 if value is not None else None

    current_price = _finite_num(info.get("regularMarketPrice")) or _finite_num(info.get("currentPrice"))
    price = current_price or _finite_num(info.get("previousClose"))
    quote_time = _finite_num(info.get("regularMarketTime")) if current_price is not None else None
    try:
        quote_date = dt.datetime.fromtimestamp(quote_time, dt.timezone.utc).date().isoformat() if quote_time else None
    except (OSError, ValueError, OverflowError):
        quote_date = None
    low_52w = _finite_num(info.get("fiftyTwoWeekLow"))
    high_52w = _finite_num(info.get("fiftyTwoWeekHigh"))
    market_cap = _finite_num(info.get("marketCap")) if asset_class in EQUITY_CLASSES | {"reit"} else None
    avg_vol = _finite_num(info.get("averageDailyVolume3Month")) or _finite_num(info.get("averageVolume"))
    volume_2m = avg_vol * price if price is not None and price > 0 and avg_vol is not None and avg_vol > 0 else None
    raw_dy = _yahoo_dy_percent(info, price)

    pl = _finite_num(info.get("trailingPE"))
    if pl is None and price is not None and price > 0:
        eps = _finite_num(info.get("trailingEps"))
        if eps is not None and eps != 0:
            pl = price / eps
    pvp, book_equity = _yahoo_pvp_and_equity(info, price, market_cap)
    enterprise_value = _finite_num(info.get("enterpriseValue"))
    ebitda = _finite_num(info.get("ebitda"))
    ev_ebitda = _finite_num(info.get("enterpriseToEbitda"))
    if ev_ebitda is None and enterprise_value is not None and enterprise_value > 0 and ebitda is not None and ebitda != 0:
        ev_ebitda = enterprise_value / ebitda
    if enterprise_value is not None and enterprise_value <= 0 and ev_ebitda is not None and ev_ebitda < 0:
        ev_ebitda = None

    candidates = {
        "pl": pl, "pvp": pvp, "ev_ebitda": ev_ebitda, "roe": _pct("returnOnEquity"),
        "net_margin": _pct("profitMargins"), "ebit_margin": _pct("operatingMargins"),
        "dy": raw_dy,
    }
    indicators = {key: value for key, value in candidates.items() if key in ASSET_INDICATORS[asset_class]}

    currency = str(info.get("currency") or meta.get("currency") or "USD").upper()
    company_name = str(info.get("longName") or info.get("shortName") or meta.get("company") or canonical)
    sector = str(meta.get("sector") or info.get("sector") or "Mercado Global")
    subsector = str(info.get("industry") or ASSET_CLASS_LABELS.get(meta.get("asset_class", ""), ""))
    current_ratio = _finite_num(info.get("currentRatio")) if asset_class in EQUITY_CLASSES else None

    health_data: dict[str, float | None] = {}
    if asset_class in EQUITY_CLASSES:
        if current_ratio is not None:
            health_data["current_ratio"] = current_ratio
        if not _is_financial_institution(info.get("sector"), info.get("industry"), ebitda=ebitda, current_ratio=current_ratio):
            total_debt = _finite_num(info.get("totalDebt"))
            total_cash = _finite_num(info.get("totalCash"))
            if total_debt is not None and total_cash is not None and book_equity is not None and book_equity > 0:
                health_data["debt_equity"] = (total_debt - total_cash) / book_equity
            total_revenue = _finite_num(info.get("totalRevenue"))
            operating_margins = _finite_num(info.get("operatingMargins"))
            if (enterprise_value is not None and enterprise_value > 0
                    and total_revenue is not None and total_revenue > 0
                    and operating_margins is not None and operating_margins != 0):
                health_data["ev_ebit"] = enterprise_value / (total_revenue * operating_margins)

    result = {
        "ticker": canonical,
        "available": any(value is not None for value in (price, low_52w, high_52w, market_cap, current_ratio, *indicators.values())),
        "company": company_name,
        "asset_class": asset_class,
        "sector": sector,
        "subsector": subsector or None,
        "currency": currency,
        "price": price,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "volume_2m": volume_2m,
        "market_cap": market_cap,
        "last_quote_date": quote_date,
        "indicators": indicators,
        "health": health_data,
        "source": "Yahoo Finance",
    }
    if not result["available"]:
        fallback = (
            _get_nasdaq_fundamentals(symbol, asset_class, currency)
            if asset_class in {"acao_us", "reit"}
            else _get_nasdaq_quote(symbol, asset_class, currency)
        )
        if fallback:
            result.update(fallback, available=True)
    result.update(_quote_metadata(symbol, info))
    fund_profile = fund_data.yahoo_fund_block(info, asset_class, distribution_yield=raw_dy)
    if fund_profile:
        result["fund"] = fund_profile
    if asset_class in EQUITY_CLASSES | {"reit"}:
        result["history"] = get_equity_history(canonical, symbol)
    return result


def uses_market_quote(ticker: str) -> bool:
    """Identify instruments that have market prices but no corporate indicators."""
    return (ticker not in TREASURY_TITLES and not is_synthetic_fixed_income(ticker)
            and asset_metadata(ticker)["asset_class"] in {"titulo", "outros"})


def market_asset_detail(ticker: str, quote: dict | None, *, allow_nasdaq: bool = True, b3_stock: dict | None = None,
                        info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a quote-only result; do not substitute hard-coded catalog figures."""
    meta = b3_stock or asset_metadata(ticker)
    if not quote and allow_nasdaq and meta["asset_class"] == "titulo" and meta.get("quote_type") != "BOND":
        quote = _get_nasdaq_quote(provider_symbol(ticker), meta["asset_class"], meta["currency"])
    info = info if isinstance(info, dict) else {}
    result = {
        "ticker": ticker,
        "available": bool(quote),
        "company": meta["company"],
        "asset_class": meta["asset_class"],
        "sector": meta["sector"],
        "currency": meta["currency"],
        "price": None,
        "source": "Yahoo Finance",
        **(quote or {}),
    }
    if info.get("currency") and not quote:
        result["currency"] = str(info["currency"]).upper()
    result.update(_quote_metadata(provider_symbol(ticker), info))
    fund_profile = fund_data.yahoo_fund_block(info, meta["asset_class"], distribution_yield=_yahoo_dy_percent(info, result.get("price")))
    if fund_profile:
        result["fund"] = fund_profile
    return result


def load_fii_market_data() -> pd.DataFrame:
    """Allow analysis to fall back without retrying an outage for every fund."""
    try:
        return get_fii_market_data()
    except Exception:
        logger.warning("Fundamentus FII data unavailable", exc_info=True)
        return pd.DataFrame()


def _fii_detail(ticker: str, meta: dict, table: pd.DataFrame) -> dict[str, Any] | None:
    if ticker not in table.index:
        return None
    row = table.loc[ticker, ["price", "pvp", "dy", "liq2m"]].dropna().to_dict()
    if not row:
        return None
    result = {
        "ticker": ticker, "available": True, "company": meta["company"],
        "asset_class": "fii", "sector": meta["sector"], "currency": "BRL", "volume_2m": row.get("liq2m"),
        "price": row.get("price"), "indicators": {key: row.get(key) for key in ("pvp", "dy")},
        "source": "Fundamentus · FIIs",
        **_quote_metadata(ticker),
    }
    # Vacância, cap rate, FFO yield, imóveis, segmento e gestão vivem na página de detalhes.
    # A falha na página detalhada mantém o resumo acima intacto.
    detail = fund_data.get_fii_detail(ticker)
    if detail:
        result["fund"] = detail
        result["detail_source"] = detail["source"]
    return result


def catalog_screener_rows(asset_class: str, fields_by_class: dict[str, set[str]]) -> pd.DataFrame:
    """Build reference-only screener rows; modeled exposures have no market quotes or turnover."""
    rows = []
    for ticker, item in EXTRA_ASSETS_BY_TICKER.items():
        category = item["asset_class"]
        if asset_class not in {"all", category}:
            continue
        ref = item["ref_metrics"] if item["yahoo_symbol"] else {}
        rows.append({"ticker": ticker, "currency": item["currency"], "asset_class": category,
                     "setor": item["sector"], **{field: ref.get(field) for field in sorted(fields_by_class[category]) if field != "setor"}})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
