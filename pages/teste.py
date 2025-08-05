from bcb import sgs
from bcb import sgs

# Buscar dados da Selic a partir de 2010-01-01
selic = sgs.get({'selic': 432}, start='2020-01-01')
st.write(selic)
