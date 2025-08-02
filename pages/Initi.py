import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

st.set_page_config(
    page_title="Análise de Ações B3",
    page_icon="📈"
)

st.write("# **B3 Explorer 📈**")

data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)

tickers = st.multiselect('Escolha ações para explorar! (2 ou mais ações)', stocks)

period_dict = {'diário':'1d', 'semanal':'1wk', 'mensal':'1mo', 'trimestral':'3mo', 'semestral':'6mo', 'anual':'1y'}
interval_dict = {'dia':'1d', 'semana':'1wk', 'mês':'1mo', '3 meses':'3mo', 'hora':'1h', 'minuto':'1m'}

period_selected = st.sidebar.selectbox('Período ⏰', list(period_dict.keys()))
interval_selected = st.sidebar.selectbox('Intervalo 📊', list(interval_dict.keys()))
data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2023, 1, 1), min_value=datetime.date(2000, 1, 1))

if len(tickers) < 2:
    st.warning("Selecione ao menos 2 ações para análise.")
else:
    tickers_full = [t + ".SA" for t in tickers]

    try:
        all_data = {}
        for t in tickers_full:
            df_temp = yf.download(t,
                                  start=data_inicio,
                                  end=datetime.datetime.now(),
                                  interval=interval_dict[interval_selected],
                                  progress=False)
            if not df_temp.empty:
                all_data[t] = df_temp['Close']
            else:
                st.warning(f"Dados não disponíveis para {t}")

        if all_data:
            data = pd.DataFrame(all_data)

            st.subheader("Cotação (Fechamento)")
            st.dataframe(data.tail())

            returns = data.pct_change().dropna() * 100
            returns_rounded = returns.round(2)
            st.subheader("Retornos (%)")
            st.dataframe(returns_rounded.style.format("{:.2f}%"))

            st.subheader("Gráfico das Cotações")
            st.line_chart(data)

            # Só mostrar os tickers selecionados (sem descrição)
            df_descr = pd.DataFrame({'Ticker': tickers_full})
            st.subheader("Ações Selecionadas")
            st.table(df_descr)

        else:
            st.error("Nenhum dado válido foi baixado para os tickers selecionados.")

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")



