import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar dataset do IPCA
df = pd.read_csv('bcdata.sgs.4449.csv', sep=';')

# Ajustar data
df['data'] = pd.to_datetime(df['data'], dayfirst=True)

# Converter valores
df['valor'] = pd.to_numeric(df['valor'].str.replace(',', '.'), errors='coerce')

# Plotar série temporal
plt.figure(figsize=(12,6))
plt.plot(df['data'], df['valor'])
plt.xlabel('Data')
plt.ylabel('IPCA')
plt.title('Série Temporal do IPCA (BCB)')
plt.grid(True)
plt.show()

# Definir índice
df = df.set_index('data')

df_recorte = df.loc['2020':]
plt.figure(figsize=(12,6))
plt.plot(df_recorte.index, df_recorte['valor'])
plt.xlabel('Data')
plt.ylabel('IPCA')
plt.title('Série Temporal do IPCA — Após 2020')
plt.grid(True)
plt.show()

decomp = seasonal_decompose(df_recorte['valor'], model='additive', period=12)
# tendencias
plt.figure(figsize=(12,4))
plt.plot(decomp.trend, color='orange')
plt.title('Tendência — IPCA (Após 2020)')
plt.grid(True)
plt.show()

# sazonalidade
plt.figure(figsize=(12,4))
plt.plot(decomp.seasonal, color='green')
plt.title('Sazonalidade — IPCA (Após 2020)')
plt.grid(True)
plt.show()

# ruido
plt.figure(figsize=(12,4))
plt.scatter(decomp.resid.index, decomp.resid, color='red', s=15)
plt.title('Ruído — IPCA (Após 2020)')
plt.grid(True)
plt.show()



