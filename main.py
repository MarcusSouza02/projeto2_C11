import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. CARREGAR DATASET DO IPCA
df = pd.read_csv('bcdata.sgs.4449.csv', sep=';')

# Ajustar data
df['data'] = pd.to_datetime(df['data'], dayfirst=True)

# Converter valores
df['valor'] = pd.to_numeric(df['valor'].str.replace(',', '.'), errors='coerce')

# Plotar série temporal completa
plt.figure(figsize=(12,6))
plt.plot(df['data'], df['valor'])
plt.xlabel('Data')
plt.ylabel('IPCA')
plt.title('Série Temporal do IPCA (BCB)')
plt.grid(True)
plt.show()

# 2. DEFINIR ÍNDICE DE DATA E RECORTAR APÓS 2020
df = df.set_index('data')

df_recorte = df.loc['2020':]

plt.figure(figsize=(12,6))
plt.plot(df_recorte.index, df_recorte['valor'])
plt.xlabel('Data')
plt.ylabel('IPCA')
plt.title('Série Temporal do IPCA — Após 2020')
plt.grid(True)
plt.show()

# 3. DECOMPOSIÇÃO (Tendência, Sazonalidade, Ruído)
decomp = seasonal_decompose(df_recorte['valor'], model='additive', period=12)

# Tendência
plt.figure(figsize=(12,4))
plt.plot(decomp.trend, color='orange')
plt.title('Tendência — IPCA (Após 2020)')
plt.grid(True)
plt.show()

# Sazonalidade
plt.figure(figsize=(12,4))
plt.plot(decomp.seasonal, color='green')
plt.title('Sazonalidade — IPCA (Após 2020)')
plt.grid(True)
plt.show()

# Ruído
plt.figure(figsize=(12,4))
plt.scatter(decomp.resid.index, decomp.resid, color='red', s=15)
plt.title('Ruído — IPCA (Após 2020)')
plt.grid(True)
plt.show()

# 4. FILTRAR APÓS 2020 (AGORA CORRIGIDO)
df_filtrado = df.loc['2020':]   # <-- CORRIGIDO

# 5. TREINAR O MODELO HOLT-WINTERS
modelo = ExponentialSmoothing(
    df_filtrado['valor'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
)

ajuste = modelo.fit()

# 6. FAZER PREVISÃO PARA 12 MESES
previsao = ajuste.forecast(12)


# 7. PLOTAR RESULTADO
plt.figure(figsize=(12,5))
plt.plot(df.loc['2025':].index, df.loc['2025':]['valor'], label='Série Real (2025+)')
plt.plot(previsao, label='Previsão Holt-Winters', linestyle='--')
plt.title('Previsão Holt-Winters - IPCA (após 2025)')
plt.legend()
plt.grid()
plt.show()
