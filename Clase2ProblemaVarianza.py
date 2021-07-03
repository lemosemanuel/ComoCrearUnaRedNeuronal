# k-fold cross validation me va a ayudar a resolver este problema de la varianza
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.engine import sequential


# 7 pasos simples para crear una Red Neuronal Por Emanuel LEmos 
# Paso 1: Inicializar los pesos aleatoriamente con valores cercanos a 0 (pero no 0).
# Paso 2: Introducir la primera observación del dataset en la Capa de Entrada, cada característica es un nodo de entrada.
# Paso 3: Propagación hacia adelante: de izquierda a derecha, las neuronas se activan de modo que la activación de
# cada una se limita por los pesos. Propaga las activaciones hasta obtener la predicción y.
# Paso 4: Comparamos la predicción con el resultado real. Se mide entonces el error generado.
# Paso 5: Propagación hacia atrás: de derecha a izquierda, propagando el error hacia atrás. Se actualizan los pesos
# según lo responsables que Sean del error. El ratio de aprendizaje gobierna cuánto deben actualizarse los pesos.
# Paso 6: Se repiten los Pasos 1 a 5 y se actualizan los pesos después de cada observación (Reinforcement Learning). O:
# Se repiten los Pasos 1 a 5 pero actualiza los pesos después de un conjunto de observaciones (Batch Learning).
# Paso 7: Cuando todo el Conjunto de Entrenamiento ha pasado por la RNA, se completa un epoch. Hacer más epochs.



df = pd.read_csv('Churn_Modelling.csv')
# me quedo solo con lo que me sirve del df
#me quedo desde la columna 3 hasta la 12
X = df.iloc[:, 3:13].values
Y = df.iloc[:, 13].values

#tengo variables categoricas , las modifico
#las paso en dummies (con OneHotEncoder y LabelEncoder)

# el LabelEconder lo que hace es modifica la variable en numerico , ej: si hay 3 variables lo modifica en :[1,2,3]
#creo el label encoder para el pais
'''labelencoder_X_1 = LabelEncoder()
X.iloc[:,1]=labelencoder_X_1.fit_transform(X.iloc[:,1])
#ahora con el genero
labelenconder_X_2=LabelEncoder()
X.iloc[:,2]=labelenconder_X_2.fit_transform(X.iloc[:,2])
'''


# podria hacerlo mas facil con una funcion que me busque las columnas que son string y me las convierta

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# hago el train test 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#hago el escalado de variables (ya que hay mucha diferencia entre variables)
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)



#CONSTRUIR LA RNA
# recordar los 7 pasos
#importo librerias Keras y adicionales
import keras
# inicializar los parametros con Sequential
from keras.models import Sequential
# Dense para crear las capas intermedias de la red neuronal
from keras.layers import Dense


#hago una matris de confucion
from sklearn.metrics import confusion_matrix
# como habia hecho una sigmoide tengo que marcar un umbral que me dice si el cliente abandona o no
y_pred=(y_pred>0.5)
cm= confusion_matrix(Y_test,y_pred)

cm

(1515+204)/2000


### PERO OJO HAY QUE INTENTAR RESOLVER EL PROBLEMA DE LA VARIANZA ,
#  ya que sino cada vez que lo entrene me va a dar diferente

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",  
                        activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    #devuelvo el clasificador
    return classifier
# definimos los mismos parametros que teniamos en el fit pero esta vez va a ir directamente con el cross validation
# classifier.fit(X_train, Y_train,  batch_size = 10, epochs = 100)
classifier= KerasClassifier(build_fn=build_classifier,  batch_size = 10, epochs = 100)
# cv es el numero de cross validation que quieran poner
accuracies=cross_val_score(estimator=classifier,X=X_train,y= Y_train,cv=10,n_jobs=-1)

# vamos a calcular la varianza
mean= accuracies.mean()
variance= accuracies.std()
print(mean)
print(variance)
# >>> print(mean)
# 0.8423750042915344
# >>> print(variance)
# 0.011435284603257717
#  como se ve el valor de la varianza es baja 
#  aunque la media del accuraci no es muy buena . vamos a intentar mejorar eso

# ver clase 3

