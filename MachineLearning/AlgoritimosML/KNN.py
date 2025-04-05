from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

mtcars = pd.read_csv(r'C:\Users\Jaum\Desktop\download\download\3.Algoritmos de Machine Learning\mt_cars.csv')
mtcars.head()

X = mtcars[['mpg','hp']].values
y = mtcars['cyl'].values

knn = KNeighborsClassifier(n_neighbors=3)
modelo = knn.fit(X,y)

y_prev = modelo.predict(X)

accuracy = accuracy_score(y, y_prev)
precision = precision_score(y, y_prev, average='weighted')
recall = recall_score(y, y_prev, average='weighted')
f1 = f1_score(y, y_prev, average='weighted')
cm = confusion_matrix(y, y_prev)

print(f'Acurácia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1: {f1}')
print('Matriz de confussão:\n', cm)

#'mpg','hp'
new_data = np.array([[19.3, 105]])

previsao = modelo.predict(new_data)
print(f'Previão: {previsao}')

distancia, indices = modelo.kneighbors(new_data)
print(f'Distância: {distancia}')
print(f'Indices: {indices}')

print(mtcars.loc[[1, 5, 31], ['cyl', 'mpg', 'hp']])