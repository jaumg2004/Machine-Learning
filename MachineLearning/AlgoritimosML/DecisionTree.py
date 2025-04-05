import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

# Carregar a base de dados
base = pd.read_csv(r'C:\Users\Jaum\Desktop\download\download\3.Algoritmos de Machine Learning\insurance.csv')

# Remover a coluna indesejada
base = base.drop(['Unnamed: 0'], axis=1)

# Exibir os primeiros registros e a forma do DataFrame
print(base.head())
print(base.shape)

# Definir X (variáveis independentes) e y (variável dependente)
y = base.iloc[:, 7].values
X = base.iloc[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

# Codificar variáveis categóricas em X
labelencoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = labelencoder.fit_transform(X[:, i])

# Dividir o dataset em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)

modelo = DecisionTreeClassifier(random_state=1, max_depth=8, max_leaf_nodes=8)
modelo.fit(X_treinamento, y_treinamento)

dot_data = export_graphviz(modelo, out_file=None, filled=True, freature_names=base.columns[:-1], class_names=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree",format="png")
graph.view()

# Fazer previsões
previsoes = modelo.predict(X_teste)
print(previsoes)

# Avaliar o modelo
accuracy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')

print(f'Acurácia: {accuracy:.2f}')
print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1: {f1:.2f}')

report = classification_report(y_teste, previsoes)
print(report)