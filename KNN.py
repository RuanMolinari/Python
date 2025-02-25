import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download latest version
path = kagglehub.dataset_download("yasserh/wine-quality-dataset")

print("Path to dataset files:", path)

df = pd.read_csv(f'{path}\\WineQT.csv', index_col='Id')
#df.head()
#df['quality'].value_counts()
dataplot = sns.heatmap(df.corr(), annot=True)
df.info()
sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 40})

# Criando um jitter plot
sns.stripplot(x="quality", y="fixed acidity", data=df, jitter=True)
plt.title('Jitter Plot using Seaborn')
plt.xlabel('Quality')
plt.ylabel('Fixed Acidity')
plt.show()

# Obter todas as colunas do DataFrame
columns = df.columns

# Definir o número de linhas e colunas para os subplots
n = len(columns)
ncols = 2
nrows = (n + 1) // ncols  # Calcular o número de linhas necessárias

fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
fig.suptitle('Scatter plots de todas as variáveis comparadas a target', y=1.02)

# Iterar sobre as colunas e plotar os gráficos
for i, col in enumerate(columns):
    row = i // ncols
    col_pos = i % ncols
    sns.stripplot(x="quality", y=col, data=df, jitter=True, ax=axs[row, col_pos])
    axs[row, col_pos].set_title(f'{col} vs Quality')

# Remover subplots vazios, se houver
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axs.flatten()[j])

# Ajustar layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()