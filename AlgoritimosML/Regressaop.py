import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
from scipy import stats
import seaborn as sns

base = pd.read_csv(r'C:\Users\Jaum\Desktop\download\download\3.Algoritmos de Machine Learning\mt_cars.csv')

print(base.head())
print(base.shape)

base = base.drop(['Unnamed: 0'], axis=1)

print(base.head())

corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')

column_pairs = [('mpg', 'cyl'),('mpg','disp'),('mpg', 'hp'),('mpg','drat'),('mpg', 'wt'),('mpg','vs')]
n_plots = len(column_pairs)

fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6,4*n_plots))

for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')

#aic 156.6 bic 162.5
modelo1 = sm.ols(formula='mpg ~ wt + disp + hp', data=base).fit()
print(modelo1.summary())
print(" ")

#aic 165.1 bic 169.5
modelo2 = sm.ols(formula='mpg ~ disp + cyl', data=base).fit()
print(modelo2.summary())
print(" ")

#aic 179.1 bic 183.5
modelo3 = sm.ols(formula='mpg ~ drat + vs', data=base).fit()
print(modelo2.summary())
print(" ")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,18))

residuos1 = modelo1.resid
axes[0].hist(residuos1,20)
axes[0].set_xlabel("Residuos")
axes[0].set_ylabel("Frequencia")
axes[0].set_title("Histograma de Resíduos 1")

residuos2 = modelo2.resid
axes[1].hist(residuos2,20)
axes[1].set_xlabel("Residuos")
axes[1].set_ylabel("Frequencia")
axes[1].set_title("Histograma de Resíduos 2")

residuos3 = modelo3.resid
axes[2].hist(residuos3,20)
axes[2].set_xlabel("Residuo")
axes[2].set_ylabel("Frequencia")
axes[2].set_title("Histograma de Resíduos 3")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,18))

stats.probplot(residuos1,dist="norm",plot=axes[0])
axes[0].set_title("Q-Q Plot de Residuos 1")

stats.probplot(residuos2,dist="norm",plot=axes[1])
axes[1].set_title("Q-Q Plot de Residuos 2")

stats.probplot(residuos3,dist="norm",plot=axes[2])
axes[2].set_title("Q-Q Plot de Residuos 3")
plt.tight_layout()
plt.show()

#h0 - dados estão normalmente distribuídos
#p <= 0.05 rejeito a hipótese nula, (não estão normalmente distribuídos)
#p > 0.05 não é possível rejeitar a h0
stat, pval = stats.shapiro(residuos1)
print(f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')
stat, pval = stats.shapiro(residuos2)
print(f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')
stat, pval = stats.shapiro(residuos3)
print(f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')

