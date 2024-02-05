import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt

st.sidebar.header("Opções Simulação 👨‍🔬")
mu_selected = st.sidebar.slider("Média",-4.00,4.00,0.01)
sigma_selected = st.sidebar.slider("Volatilidade",-4.00,4.00,0.001)
n_simulations = st.sidebar.slider("Número de Simulações",10,1000,10)
valor = st.sidebar.number_input("Capital Inicial", min_value=10)
periodos = int(st.sidebar.number_input("Período", min_value=2))

try:
    def simulate_1d_gbm(nsteps=periodos, t=1, mu=mu_selected, sigma=sigma_selected):
        steps = [ (mu_selected - (sigma_selected**2)/2) + np.random.randn()*sigma for i in range(periodos) ]
        y = valor+np.exp(np.cumsum(steps))
        y[0] = valor
        x = [ t*i for i in range(periodos) ]
        return x, y
except:
    pass            



try:
    st.header("Simulação 💻")
    st.subheader("Método de Monte Carlo")
    import random
    import plotly.express as px
    nsims = n_simulations
    simulation_data = {}
    values=[]
    for i in range(nsims):
        x, y = simulate_1d_gbm()
        simulation_data['y{col}'.format(col=i)] = y
        values.append(y)
        simulation_data['x'] = x

    ycols = [ 'y{col}'.format(col=i) for i in range(nsims) ]
    fig, ax = plt.subplots()
    ax.plot(x, ycols)
    st.pyplot(ax)
    terminal_values=[values[i][-1] for i in range(nsims)]
    terminal_values = np.array(terminal_values)
    terminal_values = np.reshape(terminal_values,(-1,1))
    df = pd.DataFrame([terminal_values.mean(), terminal_values.std(), terminal_values.max(), 
                       terminal_values.min()])
    df = df.rename({0:"Média", 1:"Desvio Padrão", 2:"Máximo", 3:"Mínimo"}, axis=0)
    df = df.rename({0:"Estatísticas"}, axis=1)
    st.subheader("Resultados")
    st.dataframe(df.T)
    





except:
    pass



