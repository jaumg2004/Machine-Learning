import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset = pd.read_csv(r'C:\Users\Jaum\Desktop\CursoIA\4.Tópicos Avançados de Machine Learning\credit_simple.csv', sep=';')
print(dataset.shape)

print(dataset.head())

y = dataset['CLASSE']
X = dataset.drop(columns='CLASSE')

print(X)

print(X.isnull().sum())

mediana = X['SALDO_ATUAL'].median()
print(mediana)

X['SALDO_ATUAL'] = X['SALDO_ATUAL'].fillna(mediana)
print(X.isnull().sum())

agrupado = X.groupby(['ESTADOCIVIL']).size()
print(agrupado)

X['ESTADOCIVIL'] = X['ESTADOCIVIL'].fillna('masculino solteiro')
print(X.isnull().sum())

desv = X['SALDO_ATUAL'].std()
print(desv)

print(X.loc[X['SALDO_ATUAL'] >= 2 * desv, 'SALDO_ATUAL'])

X.loc[X['SALDO_ATUAL'] >= 2 * desv, 'SALDO_ATUAL'] = mediana

print(X.loc[X['SALDO_ATUAL'] >= 2 * desv])

agrupado = X.groupby(['PROPOSITO']).size()
print(agrupado)

X.loc[X['PROPOSITO'] == 'Eletrodomésticos', 'PROPOSITO'] = 'outros'
X.loc[X['PROPOSITO'] == 'qualificação', 'PROPOSITO'] = 'outros'
agrupado = X.groupby(['PROPOSITO']).size()
print(agrupado)

X['DATA'] = pd.to_datetime(X['DATA'], format='%d/%m/%Y')
print(X['DATA'])

X['ANO'] = X['DATA'].dt.year
X['MES'] = X['DATA'].dt.month
X['DIADASEMANA'] = X['DATA'].dt.day_name()

print(X['ESTADOCIVIL'].unique())
print(X['PROPOSITO'].unique())
print(X['DIADASEMANA'].unique())

labelencoder1 = LabelEncoder()
X['ESTADOCIVIL'] = labelencoder1.fit_transform(X['ESTADOCIVIL'])
X['PROPOSITO'] = labelencoder1.fit_transform(X['PROPOSITO'])
X['DIADASEMANA'] = labelencoder1.fit_transform(X['DIADASEMANA'])

print(X.head())

outros = X['OUTROSPLANOSPGTO'].unique()
print(outros)

z = pd.get_dummies(X['OUTROSPLANOSPGTO'], prefix='OUTROS')

print(z)
print(X)

sc = StandardScaler()
m = sc.fit_transform(X.iloc[: , 0:3])

X = pd.concat([X, z, pd.DataFrame(m,columns=['SALDO_ATUAL_N', 'RESIDENCIA_N', 'IDADE_n'])], axis=1)

X.drop(columns=['SALDO_ATUAL', 'RESIDENCIADESDE', 'IDADE', 'OUTROSPLANOSPGTO', 'DATA', 'OUTROS_banco'], inplace=True)

print(X)